import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import models
from tests.test_features import QWEN, OmpHome


LOCAL_SHA = "a" * 40
UPSTREAM_SHA = "b" * 40


class OllamaManifestFreshness(unittest.TestCase):
    def test_installed_digest_is_compared_to_same_tag_manifest_or_reported_unavailable(self):
        manifest = b'{"schemaVersion":2,"layers":[]}'
        digest = models.hashlib.sha256(manifest).hexdigest()
        tags = json.dumps({"models": [{"name": "example:latest", "digest": digest, "size": 42}]}).encode()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(models.park, "STATE", Path(tmp) / "absent"):
            for code, upstream, expected in ((200, manifest, "current"),
                                             (200, b"different manifest", "update available"),
                                             (0, b"", "unreachable")):
                with self.subTest(expected=expected):
                    def fetch(url, *args, **kwargs):
                        return (200, tags) if url.endswith("/api/tags") else (code, upstream)

                    with mock.patch.object(models, "_fetch", side_effect=fetch):
                        row, = models.ollama_models()
                    self.assertEqual((row["name"], row["source"], row["digest"]),
                                     ("example:latest", "example:latest", digest[:12]))
                    self.assertIn(expected, row["freshness"])
                    self.assertIsNone(row["upstream_modified"])
                    self.assertEqual(row["upstream_date_source"], "unavailable")
                    if code == 200:
                        self.assertEqual(row["upstream_digest"], models.hashlib.sha256(upstream).hexdigest()[:12])
                    else:
                        self.assertNotIn("upstream_digest", row)


class HuggingFaceFreshness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "publisher" / "model"
        self.directory.mkdir(parents=True)
        (self.directory / "config.json").write_text('{"architecture":"example"}')

    def row(self, info=None, code=200, repo="publisher/model"):
        if info is None:
            info = {"id": repo, "sha": UPSTREAM_SHA,
                    "lastModified": "2026-09-28T12:34:00.000Z",
                    "createdAt": "2026-01-02T03:04:00.000Z"}
        body = json.dumps(info).encode() if not isinstance(info, bytes) else info
        with mock.patch.object(models, "_fetch", return_value=(code, body)):
            return models._hf_row(self.directory, "mlx-serve", "publisher/model", repo)

    def test_newer_local_mtime_does_not_override_changed_upstream_commit(self):
        (self.directory / ".hf_commit").write_text(LOCAL_SHA + "\n")
        os.utime(self.directory / "config.json", (2_000_000_000, 2_000_000_000))
        row = self.row()
        self.assertEqual(row["installed_artifact"], LOCAL_SHA[:12])
        self.assertEqual(row["installed_artifact_source"], ".hf_commit (recorded HF commit; bytes not verified)")
        self.assertEqual(row["upstream_sha"], UPSTREAM_SHA[:12])
        self.assertIn("update available", row["freshness"])
        self.assertNotIn("current", row["freshness"])
        self.assertEqual(row["upstream_modified"], "2026-09-28T12:34")
        self.assertEqual(row["upstream_date_source"], "lastModified")
        self.assertEqual((row["server"], row["name"], row["repo"]),
                         ("mlx-serve", "publisher/model", "publisher/model"))

    def test_matching_recorded_commit_is_current_without_claiming_byte_verification(self):
        (self.directory / "refs").mkdir()
        (self.directory / "refs" / "main").write_text(UPSTREAM_SHA + "\n")
        row = self.row()
        self.assertEqual(row["installed_artifact"], UPSTREAM_SHA[:12])
        self.assertEqual(row["installed_artifact_source"], "refs/main (recorded HF commit; bytes not verified)")
        self.assertIn("current", row["freshness"])
        self.assertIn("bytes not verified", row["freshness"])

    def test_matching_prefix_is_not_a_matching_commit(self):
        local_sha = UPSTREAM_SHA[:12] + "c" * 28
        (self.directory / ".hf_commit").write_text(local_sha)
        row = self.row()
        self.assertEqual(row["installed_artifact"], row["upstream_sha"])
        self.assertIn("update available", row["freshness"])
        self.assertNotIn("current", row["freshness"])

    def test_hf_download_metadata_commit_is_recorded_without_hashing_weights(self):
        metadata = self.directory / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata / ".gitattributes.metadata").write_text(f"{UPSTREAM_SHA}\nfile-etag\n0\n")
        row = self.row()
        self.assertEqual(row["installed_artifact"], UPSTREAM_SHA[:12])
        self.assertIn(".gitattributes.metadata", row["installed_artifact_source"])
        self.assertIn("current", row["freshness"])

    def test_missing_commit_only_identifies_cheap_files_and_cannot_be_current(self):
        weights = self.directory / "model.safetensors"
        weights.write_bytes(b"initial weights")
        before = self.row()
        weights.write_bytes(b"changed weights")
        after = self.row()
        self.assertEqual(before["installed_artifact"], after["installed_artifact"])
        self.assertTrue(after["installed_artifact"].startswith("files:"))
        self.assertIn("config", after["installed_artifact_source"])
        self.assertIn("unknown", after["freshness"])
        self.assertNotIn("current", after["freshness"])
        (self.directory / "config.json").write_text('{"architecture":"changed"}')
        self.assertNotEqual(after["installed_artifact"], self.row()["installed_artifact"])

    def test_unreachable_api_keeps_installed_identity_but_marks_freshness_unavailable(self):
        (self.directory / ".hf_commit").write_text(LOCAL_SHA)
        row = self.row(code=0)
        self.assertEqual(row["installed_artifact"], LOCAL_SHA[:12])
        self.assertIn("unavailable", row["freshness"])
        self.assertIn("unreachable", row["freshness"])
        self.assertNotIn("current", row["freshness"])
        self.assertIsNone(row["upstream_modified"])
        self.assertEqual(row["upstream_date_source"], "unavailable")
        self.assertNotIn("upstream_sha", row)

    def test_invalid_api_sha_and_body_explain_unknown_freshness(self):
        (self.directory / ".hf_commit").write_text(LOCAL_SHA)
        for info in ({"sha": "invalid", "createdAt": "2026-01-02T03:04:00Z"}, b"not json"):
            with self.subTest(info=info):
                row = self.row(info)
                self.assertIn("invalid Hugging Face response", row["freshness"])
                self.assertNotIn("current", row["freshness"])

    def test_wrong_repo_metadata_and_conflicting_records_cannot_claim_current(self):
        (self.directory / ".hf_commit").write_text(UPSTREAM_SHA)
        wrong_repo = self.row(repo="other/model")
        self.assertIn("repo", wrong_repo["freshness"])
        self.assertNotIn("current", wrong_repo["freshness"])
        (self.directory / "refs").mkdir()
        (self.directory / "refs" / "main").write_text(LOCAL_SHA)
        conflicting = self.row()
        self.assertIn("conflicting", conflicting["freshness"])
        self.assertNotIn("current", conflicting["freshness"])

    def test_created_at_is_labeled_as_creation_not_modification(self):
        row = self.row({"id": "publisher/model", "sha": UPSTREAM_SHA,
                        "createdAt": "2026-01-02T03:04:00.000Z"})
        self.assertEqual(row["upstream_modified"], "2026-01-02T03:04")
        self.assertEqual(row["upstream_date_source"], "createdAt")
        self.assertNotIn("current", row["freshness"])


    def test_missing_upstream_date_is_explicitly_unavailable(self):
        row = self.row({"id": "publisher/model", "sha": UPSTREAM_SHA})
        self.assertIsNone(row["upstream_modified"])
        self.assertEqual(row["upstream_date_source"], "unavailable")
        self.assertIn("unknown", row["freshness"])


class DisabledAgents(unittest.TestCase):
    """`localbench models`/`gpu` route listing over a fake omp under a temp HOME (tests.test_features.OmpHome)."""

    def setUp(self):
        self.omp = OmpHome(Path(tempfile.mkdtemp(prefix="models-")))
        self.omp.profile("default", {"smol": QWEN}, disabled=("scout",))
        self.omp.profile("flow", {"smol": QWEN}, disabled=("scout", "sonic"), flow=True)
        self.omp.profile("lab", {"smol": QWEN})

    def test_disabled_scout_is_not_listed_as_a_qwen_route_but_its_other_smol_uses_are(self):
        disabled: dict[str, list[str]] = {}
        with self.omp.env():
            uses = models.routes_by_model(["default", "flow", "lab"], disabled)
        qwen = uses[QWEN.split("/", 1)[1]]
        scout = [f for f in qwen if f.startswith("scout ")]
        self.assertEqual([qwen[f] for f in scout], [["lab"]])
        self.assertEqual(disabled, {scout[0]: ["default", "flow"]})
        others = {p for f, who in qwen.items() if f not in scout for p in who}
        self.assertEqual(others, {"default", "flow", "lab"})


if __name__ == "__main__":
    unittest.main()
