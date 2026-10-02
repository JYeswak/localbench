"""External-input checks for the planning and claim gates."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


class GateScripts(unittest.TestCase):
    def setUp(self):
        RUNS.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="gate-selftest-", dir=RUNS)
        self.addCleanup(self.scratch.cleanup)
        self.tmp = Path(self.scratch.name)

    def gate(self, script, *args, cwd=ROOT):
        return subprocess.run(
            ["sh", str(ROOT / "scripts" / script), *map(str, args)],
            cwd=cwd,
            env={**os.environ, "TMPDIR": str(self.tmp)},
            text=True,
            capture_output=True,
            check=False,
        )

    def claim_fixture(self):
        proof = self.tmp / "claim-receipt.txt"
        proof.write_text('"signed": true\n')
        readme = self.tmp / "README.md"
        readme.write_text("The report was signed.\n")
        claims = self.tmp / "claims.tsv"
        claims.write_text(
            "label\treadme_pattern\tcapability_key\texpected_substr\tproof_path\tenforce\tnotes\n"
            f'report_signed\tThe report was signed.\t\t"signed": true\t{proof.relative_to(ROOT)}'
            "\tyes\tvalidated test receipt\n"
        )
        return claims.relative_to(ROOT), readme.relative_to(ROOT)


    def test_missing_inputs_fail_and_name_the_path(self):
        for script, missing in (
            ("check-readiness.sh", self.tmp / "missing-packet.md"),
            ("check-claim-discipline.sh", self.tmp / "missing-claims.tsv"),
        ):
            with self.subTest(script=script):
                result = self.gate(script, missing)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(str(missing), result.stdout + result.stderr)

    def test_relative_inputs_have_same_result_from_another_cwd(self):
        claims, readme = self.claim_fixture()
        for script, args, expected in (
            ("check-readiness.sh", ("docs/planning/packet.md",), r"^READY:"),
            ("check-claim-discipline.sh", (claims, readme),
             r"check-claim-discipline: 1 passed, 0 failed"),
        ):
            with self.subTest(script=script):
                inside = self.gate(script, *args)
                outside = self.gate(script, *args, cwd=self.tmp)
                self.assertEqual(inside.returncode, 0, inside.stdout + inside.stderr)
                self.assertEqual(outside.returncode, inside.returncode, outside.stdout + outside.stderr)
                self.assertEqual(outside.stdout, inside.stdout)
                self.assertRegex(outside.stdout, expected)

    def test_sign_off_requires_real_marker_not_design_substring(self):
        packet = (ROOT / "docs" / "planning" / "packet.md").read_text()
        marker = "<!-- CHECK: SIGN-OFF -->"
        prefix = packet[: packet.index(marker) + len(marker)]
        design = self.tmp / "design.md"
        design.write_text(prefix + "\nDesign reviewed 2026-09-22.\nDesign dated 2026-09-22.\n")
        unsigned = self.gate("check-readiness.sh", design)
        self.assertNotEqual(unsigned.returncode, 0, unsigned.stdout)
        self.assertIn("execution sign-off", unsigned.stdout)
        self.assertIn("vocabulary", unsigned.stdout)

        signed = self.tmp / "signed.md"
        signed.write_text(prefix + "\nSigned by agent on 2026-09-22.\nSign-off accepted on 2026-09-22.\n")
        valid = self.gate("check-readiness.sh", signed)
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
        self.assertIn("READY:", valid.stdout)

    def test_enforced_readme_claim_drift_fails_loudly(self):
        claims, readme = self.claim_fixture()
        drifted = self.tmp / "README-drift.md"
        drifted.write_text((ROOT / readme).read_text().replace("was signed.", "was unsigned."))

        good = self.gate("check-claim-discipline.sh", claims, readme)
        bad = self.gate("check-claim-discipline.sh", claims, drifted)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)
        self.assertIn("report_signed", bad.stdout)
        self.assertIn("pattern", bad.stdout)



if __name__ == "__main__":
    unittest.main()
