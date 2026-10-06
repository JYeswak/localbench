import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts import rehearse


class FakeSuite:
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    @staticmethod
    def pin():
        return {"items_sha256": "x"}


class Rehearsal(unittest.TestCase):
    def test_endpoint_failure_covers_the_suite_with_one_probe(self):
        from localbench import decision

        suite = FakeSuite()
        suite.items = [{"id": f"item-{i}"} for i in range(128)]
        with patch.object(decision, "post", wraps=decision.post) as post:
            receipt = rehearse._run_suite({}, {}, {}, suite, "residency-loss")
        outcomes = receipt["run"]["decision"]["local"]["outcomes"]
        self.assertEqual([x["id"] for x in outcomes], [item["id"] for item in suite.items])
        self.assertEqual(post.call_count, 1)
        self.assertTrue(all(x["error"] == "http" and x["detail"] for x in outcomes))
        self.assertTrue(all(x["latency_s"] is None for x in outcomes))

    def test_stall_fault_times_out_without_dropping_items(self):
        receipt = rehearse._run_suite({}, {}, {}, FakeSuite(), "stall")
        outcomes = receipt["run"]["decision"]["local"]["outcomes"]
        self.assertEqual([x["id"] for x in outcomes], ["a", "b", "c"])
        self.assertTrue(all(x["error"] == "timeout" and x["detail"] for x in outcomes))
        self.assertTrue(all(x["latency_s"] is None for x in outcomes))


    def test_decision_rehearsal_receipt_passes_localbench_validate(self):
        from localbench.__main__ import validate_doc

        receipt = rehearse._run_suite({}, {}, {}, FakeSuite(), "residency-loss")
        self.assertEqual(validate_doc(receipt), [])

    def test_memory_rehearsal_receipts_pass_localbench_validate(self):
        from localbench import prove
        from localbench.__main__ import validate_doc

        spec = prove.load_spec(rehearse.ROOT / "registries/proofs/mnemopi-extraction__proof.json")
        # The module sha is read from the installed omp package; CI has no omp (hermetic suite, cb05f21), and the
        # receipt contract under test does not depend on which sha it is.
        with patch.object(prove, "_module_sha", return_value="0" * 16):
            receipts = prove.run_memory_verdicts(spec, rehearse._legs(spec, "residency-loss"))
        self.assertTrue(receipts)
        self.assertTrue(all(not validate_doc(item["receipt"]) for item in receipts))


    def test_http_fault_server_exercises_real_loopback_path(self):
        for fault, expected in (("5xx", 503), ("residency-loss", 503), ("slow-first-byte", 200)):
            with self.subTest(fault=fault), rehearse.FakeFaultServer(fault) as url:
                try:
                    with urlopen(Request(url, data=b"{}", method="POST"), timeout=5) as response:
                        status = response.status
                except HTTPError as exc:
                    status = exc.code
                    exc.close()
                self.assertEqual(status, expected)

    def test_healthy_decision_keeps_every_item_without_error(self):
        receipt = rehearse._run_suite({}, {}, {}, FakeSuite(), "")
        outcomes = receipt["run"]["decision"]["local"]["outcomes"]
        self.assertEqual([x["id"] for x in outcomes], [item["id"] for item in FakeSuite.items])
        self.assertTrue(all(x["ok"] for x in outcomes))


if __name__ == "__main__":
    unittest.main()
