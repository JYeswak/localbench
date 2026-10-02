"""localbench omp freeze and run|ab --omp-frozen (localbench/ompfreeze.py) against a fake global install: a stdlib-made
package tree laid out like ~/.bun/install/global/node_modules, an `omp` symlink to its dist entry, and a fake `bun` on
PATH that runs the entry with /bin/sh. The entry prints `omp/<version>` read from the package.json next to it, so a
snapshot that still reads the live install shows up as the live version."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import audit, backends, ompfreeze, workloads

ROOT = Path(__file__).resolve().parent.parent

ENTRY = """#!/usr/bin/env bun
here=$(dirname "$(realpath "$0")")
v=$(sed -n 's/.*"version": *"\\([^"]*\\)".*/\\1/p' "$here/../package.json")
echo "omp/$v{suffix}"
"""
BUN = """#!/bin/sh
if [ "$1" = --version ]; then echo 1.4.0-fake; exit 0; fi
exec /bin/sh "$@"
"""


def _write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def _package(path: Path, name: str, version: str = "1.0.0", **deps) -> None:
    _write(path / "package.json", json.dumps({"name": name, "version": version, **deps}))


class FakeInstall:
    """tmp/global/node_modules: @fake/coding-agent (the omp package) -> dep-a -> dep-b (nested under dep-a, while a
    different hoisted dep-b exists too); `unrelated` is installed but not depended on. tmp/bin holds `omp` (symlink to
    the entry, as ~/.bun/bin/omp is) and `bun`."""

    def __init__(self, tmp: Path, suffix: str = ""):
        self.tmp, self.nm = tmp, tmp / "global" / "node_modules"
        self.pkg = self.nm / "@fake" / "coding-agent"
        _package(self.pkg, "@fake/coding-agent", bin={"omp": "dist/cli.js"}, dependencies={"dep-a": "^1"},
                 optionalDependencies={"not-installed-opt": "^1"}, peerDependencies={"not-installed-peer": "^1"})
        _write(self.pkg / "src" / "config" / "model-resolver.ts", "export {}\n")
        _write(self.pkg / "dist" / "cli.js", ENTRY.replace("{suffix}", suffix), 0o755)
        _package(self.nm / "dep-a", "dep-a", dependencies={"dep-b": "^2"})
        _write(self.nm / "dep-a" / "value.txt", "a1\n")
        # A symlink into the live tree: a snapshot that kept it would read the live install.
        (self.nm / "dep-a" / "linked.txt").symlink_to(self.nm / "dep-a" / "value.txt")
        _package(self.nm / "dep-a" / "node_modules" / "dep-b", "dep-b", "2.0.0")
        _package(self.nm / "dep-b", "dep-b", "9.9.9")
        _package(self.nm / "unrelated", "unrelated")
        self.bin = tmp / "bin"
        self.bin.mkdir()
        (self.bin / "omp").symlink_to(self.pkg / "dist" / "cli.js")
        _write(self.bin / "bun", BUN, 0o755)

    def env(self) -> dict:
        return {"LOCALBENCH_OMP": str(self.bin / "omp"), "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}"}

    def update(self, version: str) -> None:
        """What uca does: a new package.json and a new dist entry in the live install."""
        _package(self.pkg, "@fake/coding-agent", version, bin={"omp": "dist/cli.js"}, dependencies={"dep-a": "^1"})
        _write(self.pkg / "dist" / "cli.js", ENTRY.replace("{suffix}", "") + f"# build {version}\n", 0o755)

    def live_version(self) -> str:
        out = subprocess.run(["omp", "--version"], env={**os.environ, **self.env()}, capture_output=True, text=True,
                             check=False)
        return out.stdout.strip()


class FreezeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()     # /var -> /private/var: realpaths compare equal
        self.addCleanup(self._cleanup)
        self.fake = FakeInstall(self.tmp)
        self.frozen = self.tmp / "frozen"
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(ompfreeze, "FROZEN_ROOT", self.frozen))
        stack.enter_context(mock.patch.dict(os.environ, self.fake.env()))

    def _cleanup(self):
        for dirpath, _dirs, _files in os.walk(self.tmp):
            os.chmod(dirpath, 0o755)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def freeze(self) -> dict:
        return ompfreeze.freeze(ompfreeze.plan())


class Freeze(FreezeCase):
    def test_snapshot_runs_the_frozen_version_after_the_live_install_updates(self):
        manifest = self.freeze()
        target = self.frozen / manifest["snapshot_id"]
        entry = ompfreeze.entry_path(target)
        self.assertTrue(manifest["created"])
        self.assertEqual(manifest["snapshot_id"], f"1.0.0-{backends.sha16(str(self.fake.pkg / 'dist' / 'cli.js'))}")
        self.assertEqual(ompfreeze.entry_version(target), "1.0.0")
        sha_before = backends.sha16(str(entry))
        self.fake.update("2.0.0")
        self.assertEqual(self.fake.live_version(), "omp/2.0.0")          # the live omp moved
        self.assertEqual(ompfreeze.entry_version(target), "1.0.0")       # the snapshot did not
        self.assertEqual(backends.sha16(str(entry)), sha_before)
        self.assertEqual(manifest["bun_version"], "1.4.0-fake")

    def test_snapshot_holds_the_dependency_closure_and_nothing_points_back_into_the_install(self):
        target = self.frozen / self.freeze()["snapshot_id"]
        nm = target / "node_modules"
        self.assertEqual((nm / "dep-a" / "node_modules" / "dep-b" / "package.json").read_text(),
                         (self.fake.nm / "dep-a" / "node_modules" / "dep-b" / "package.json").read_text())
        self.assertFalse((nm / "dep-b").exists(), "dep-a resolves its nested dep-b; the hoisted one is not needed")
        self.assertFalse((nm / "unrelated").exists())
        self.assertTrue((nm / "@fake" / "coding-agent" / "src" / "config" / "model-resolver.ts").is_file())
        outside = []
        for dirpath, dirs, files in os.walk(target):
            for name in dirs + files:
                p = Path(dirpath) / name
                if p.is_symlink() and not Path(os.path.realpath(p)).is_relative_to(target):
                    outside.append(str(p))
        self.assertEqual(outside, [])
        self.assertTrue((target / "bin" / "bun").is_file() and not (target / "bin" / "bun").is_symlink())

    def test_snapshot_is_read_only_and_resolves_to_its_frozen_package(self):
        from localbench import park

        target = self.frozen / self.freeze()["snapshot_id"]
        entry = ompfreeze.entry_path(target)
        with self.assertRaises(PermissionError):
            (target / "node_modules" / "@fake" / "coding-agent" / "dist" / "cli.js").write_text("changed")
        # features/presets/prove find omp's package from LOCALBENCH_OMP: through the entry, the frozen one.
        self.assertEqual(park.omp_package(str(entry)), target / "node_modules" / "@fake" / "coding-agent")

    def test_a_second_freeze_of_the_same_omp_reuses_the_snapshot_without_copying(self):
        first = self.freeze()
        plan = ompfreeze.plan()
        self.assertTrue(plan.exists)
        self.assertIsNone(plan.refuse)
        with mock.patch.object(shutil, "copytree", side_effect=AssertionError("copied again")):
            again = ompfreeze.freeze(plan)
        self.assertFalse(again["created"])
        self.assertEqual(again["made_at"], first["made_at"])

    def test_an_update_during_the_copy_is_refused_and_leaves_no_snapshot(self):
        plan = ompfreeze.plan()
        self.fake.update("2.0.0")
        with self.assertRaisesRegex(RuntimeError, "changed during the copy"):
            ompfreeze.freeze(plan)
        self.assertEqual(sorted(p.name for p in self.frozen.iterdir()), [])

    def test_an_entry_that_does_not_report_the_frozen_version_is_removed(self):
        # A runtime file the closure misses (here: the version comes from a package nothing declares): the copy runs
        # but answers differently from the live omp, so it must not be offered as that omp.
        _package(self.fake.nm / "undeclared", "undeclared", "1.0.0")
        _write(self.fake.pkg / "dist" / "cli.js", "#!/usr/bin/env bun\nhere=$(dirname \"$(realpath \"$0\")\")\n"
               "cat \"$here/../../../undeclared/VERSION\" 2>/dev/null || echo omp/missing\n", 0o755)
        _write(self.fake.nm / "undeclared" / "VERSION", "omp/1.0.0\n")
        self.assertEqual(self.fake.live_version(), "omp/1.0.0")
        plan = ompfreeze.plan()
        with self.assertRaisesRegex(RuntimeError, "--version printed 'missing'"):
            ompfreeze.freeze(plan)
        self.assertFalse(plan.target.exists())

    def test_a_foreign_directory_at_the_target_is_refused_and_left_alone(self):
        plan = ompfreeze.plan()
        _write(plan.target / "mine.txt", "not localbench's\n")
        plan = ompfreeze.plan()
        self.assertIn("not a snapshot localbench made", plan.refuse or "")
        with self.assertRaises(RuntimeError):
            ompfreeze.freeze(plan)
        self.assertEqual((plan.target / "mine.txt").read_text(), "not localbench's\n")


class Retention(FreezeCase):
    def _versions(self, *versions: str) -> list[str]:
        ids = []
        for v in versions:
            self.fake.update(v)
            ids.append(self.freeze()["snapshot_id"])
            ompfreeze.prune(ids[-1])
        return ids

    def test_keeps_the_newest_three_snapshots_and_never_touches_other_directories(self):
        _write(self.frozen / "user-notes" / "keep.txt", "x\n")
        _write(self.frozen / "0.0.1-copied" / ompfreeze.MARKER, json.dumps({"snapshot_id": "something-else"}))
        ids = self._versions("1.0.1", "1.0.2", "1.0.3", "1.0.4", "1.0.5")
        names = sorted(p.name for p in self.frozen.iterdir() if not p.name.startswith("."))
        self.assertEqual(names, sorted([*ids[-3:], "user-notes", "0.0.1-copied"]))
        self.assertTrue((self.frozen / "user-notes" / "keep.txt").is_file())

    def test_a_snapshot_a_run_holds_is_not_pruned(self):
        ids = self._versions("1.0.1")
        with ompfreeze.hold(ids[0]):
            more = self._versions("1.0.2", "1.0.3", "1.0.4")
            self.assertTrue((self.frozen / ids[0]).is_dir())
        ompfreeze.prune(more[-1])
        self.assertEqual(sorted(p.name for p in self.frozen.iterdir() if not p.name.startswith(".")), sorted(more))

    def test_the_snapshot_being_reused_is_kept_even_when_it_is_the_oldest(self):
        ids = self._versions("1.0.1", "1.0.2", "1.0.3")
        self.assertEqual(ompfreeze.prune_candidates(ids[0]), [])
        self.assertEqual(ompfreeze.prune_candidates("1.0.4-new"), [self.frozen / ids[0]])


def _fake_leg(update_after_first=None):
    """A leg that pins omp the way omp_pins does (omp_bin() --version, sha16 of omp_bin()); after the first leg the
    live install is updated, as uca did on 2026-10-02."""
    calls = []

    def execute(backend, model, *, label="run", **_):
        binary = workloads.omp_bin()
        version = backends._first_line(binary, "--version").removeprefix("omp/").strip()
        calls.append((label, binary, version, backends.sha16(binary)))
        if update_after_first and len(calls) == 1:
            update_after_first()
        return {"provenance": {"label": label, "created": "t",
                               "pins": {"model": "m", "omp_version": version, "omp_sha": backends.sha16(binary)}},
                "verdicts": {"contended": False, "must_fail": [], "preflight_problems": []},
                "metrics": {"e2e.ok.first_startup_s": {"value": 1.0, "better": "lower"}},
                "conformance": {}, "run_dir": "r", "results": [],
                "system": {"before": {"live": {}, "host": {}}, "during": {"resident_unknown_samples": 0},
                           "cpu": {"busy_pct": {"mean": 5.0}}}}
    return execute, calls


@contextlib.contextmanager
def _fake_backend(spec, args=()):
    yield None, "m"


class Cli(FreezeCase):
    def _main(self, argv, execute=None):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as receipts, \
                mock.patch.object(cli, "execute", execute or _fake_leg()[0]), \
                mock.patch.object(cli, "open_backend", _fake_backend), mock.patch.object(cli, "RECEIPTS", Path(receipts)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(argv)
            receipt = Path(receipts) / "t.json"
            doc = json.loads(receipt.read_text()) if receipt.is_file() else None
        return rc, out.getvalue(), err.getvalue(), doc

    def test_omp_freeze_prints_the_entry_and_writes_an_audit_row(self):
        rows_before = len(audit.rows())
        rc, out, _err, _ = self._main(["omp", "freeze"])
        self.assertEqual(rc, 0)
        entry = ompfreeze.entry_path(ompfreeze.plan().target)
        self.assertEqual(out, f"LOCALBENCH_OMP={entry}\n")
        row = audit.rows()[-1]
        self.assertEqual(len(audit.rows()), rows_before + 1)
        self.assertEqual((row["verb"], row["outcome"], row["detail"]["created"]), ("omp freeze", "done", True))
        rc, out, _err, _ = self._main(["omp", "freeze"])                  # again: reuse, same entry, a no-op row
        self.assertEqual((rc, out), (0, f"LOCALBENCH_OMP={entry}\n"))
        self.assertIn("reused", audit.rows()[-1]["detail"]["noop"])

    def test_omp_freeze_dry_run_changes_nothing(self):
        rows_before = len(audit.rows())
        rc, out, _err, _ = self._main(["omp", "freeze", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("copy 3 packages", out)                            # the package, dep-a, dep-a's nested dep-b
        self.assertFalse(self.frozen.exists())
        self.assertEqual(len(audit.rows()), rows_before)

    def test_ab_omp_frozen_pins_one_omp_across_legs_while_the_live_install_updates(self):
        execute, calls = _fake_leg(lambda: self.fake.update("2.0.0"))
        rc, _out, _err, receipt = self._main(["ab", "ollama:m", "ollama:m", "--tiers", "e2e", "--pairs", "2",
                                              "--bank", "t", "--omp-frozen"], execute)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 5)
        self.assertEqual({(binary, version, sha) for _label, binary, version, sha in calls},
                         {calls[0][1:]})
        self.assertTrue(calls[0][1].startswith(str(self.frozen)))
        self.assertEqual(calls[0][2], "1.0.0")
        self.assertEqual(receipt["pin_drift"], {})
        self.assertEqual(os.environ["LOCALBENCH_OMP"], str(self.fake.bin / "omp"))   # restored after the invocation
        self.assertEqual(self.fake.live_version(), "omp/2.0.0")

    def test_without_omp_frozen_the_same_update_splits_the_a_legs(self):
        # The known-bad leg: what --omp-frozen prevents, so the test above cannot pass by an unchanging fake.
        execute, calls = _fake_leg(lambda: self.fake.update("2.0.0"))
        rc, _out, _err, receipt = self._main(["ab", "ollama:m", "ollama:m", "--tiers", "e2e", "--pairs", "2",
                                              "--bank", "t"], execute)
        self.assertEqual([c[2] for c in calls], ["1.0.0", "2.0.0", "2.0.0", "2.0.0", "2.0.0"])
        self.assertIn("e2e", receipt["pin_drift"])
        self.assertEqual(rc, 0)   # the drift voids the e2e rows; it is reported, not an exit status

    def test_run_omp_frozen_runs_its_leg_through_the_snapshot(self):
        execute, calls = _fake_leg()
        with mock.patch.object(cli, "judge", lambda s, **_: 0):
            rc, _out, _err, _ = self._main(["run", "ollama:m", "--tiers", "e2e", "--omp-frozen"], execute)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1].startswith(str(self.frozen)))
        self.assertEqual(calls[0][2], "1.0.0")


if __name__ == "__main__":
    unittest.main()
