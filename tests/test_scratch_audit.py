import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("scratch_audit", Path(__file__).parents[1] / "scripts" / "scratch-audit.py")
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class ScratchAudit(unittest.TestCase):
    def test_clean_owned_subtree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "var" / "agent-tmp" / "owned"
            d.mkdir(parents=True)
            (d / ".owner").write_text("pid=1 label=test repo=" + str(root) + "\n")
            self.assertEqual(mod.scan(root)["ownerless"], [])
            self.assertEqual(mod.scan(root)["nested_git"], [])
            self.assertEqual(mod.main([str(root), "--json"]), 0)

    def test_ownerless_and_nested_git_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "var" / "agent-tmp" / "bad"
            (d / ".git").mkdir(parents=True)
            result = mod.scan(root)
            self.assertEqual(result["ownerless"], ["var/agent-tmp/bad"])
            self.assertEqual(result["nested_git"], ["var/agent-tmp/bad/.git"])
            self.assertEqual(mod.main([str(root), "--json"]), 1)

    def test_foreign_owner_repo_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "var" / "agent-tmp" / "foreign"
            d.mkdir(parents=True)
            (d / ".owner").write_text("repo=/elsewhere\n")
            result = mod.scan(root)
            self.assertEqual(len(result["foreign"]), 1)


if __name__ == "__main__":
    unittest.main()
