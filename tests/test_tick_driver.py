"""The tick driver's decisions: when it may wake the agent pane, and when it must stop."""

import importlib.util
import sys
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("tick_driver", Path(__file__).resolve().parent.parent / "scripts" / "tick_driver.py")
assert SPEC and SPEC.loader, "scripts/tick_driver.py not found"
td = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = td      # dataclasses resolve postponed annotations through sys.modules
SPEC.loader.exec_module(td)

# Footers captured from the pane on 2026-09-23.
WORKING = " π > x\n  ⎋ Working…     Take Over V11 Loop\n ⠼ 5m > ◕ Grok 4.7 > 📁 ~/Developer/localbench > ⑂ main\n╰─\n"
SPINNING = "some output\n\n ⠦ 9s > ◕ Grok 4.7 > 📁 ~/Developer/localbench > ⑂ main ?2\n╰─\n"
IDLE = "  ⠼ 3m old spinner far up in the scrollback\nreply text\n reply ends here\n π > ◕ Grok 4.7 > 📁 ~/Developer/localbench\n╰─\n"


def step(s, **kw):
    facts = {"now": 10_000.0, "run_alive": False, "working": False, "stop": False, "head": "h1"} | kw
    return td.decide(s, **facts)


class Footer(unittest.TestCase):
    def test_a_mid_turn_footer_is_working_and_an_idle_one_is_not(self):
        self.assertTrue(td.pane_working(WORKING))
        self.assertTrue(td.pane_working(SPINNING))
        self.assertFalse(td.pane_working(IDLE))


class Decide(unittest.TestCase):
    def test_a_live_run_is_never_interrupted_however_long_the_pane_idles(self):
        s = td.State()
        for _ in range(50):
            action, s = step(s, run_alive=True)
            self.assertEqual(action, "wait")

    def test_a_mid_turn_pane_resets_the_idle_count(self):
        _, s = step(td.State())
        action, s = step(s, working=True)
        self.assertEqual((action, s.idle_checks), ("wait", 0))
        action, _ = step(s)
        self.assertEqual(action, "wait")                       # one idle check again, two are needed

    def test_the_handoff_goes_first_once_two_idle_checks_pass(self):
        action, s = step(td.State())
        self.assertEqual(action, "wait")
        action, s = step(s)
        self.assertEqual(action, "send-first")
        s = td.sent(s, now=10_000.0, head="h1")
        _, s = step(s, now=20_000.0)
        action, _ = step(s, now=20_060.0)
        self.assertEqual(action, "send-tick")

    def test_no_tick_inside_the_cooldown(self):
        s = td.sent(td.State(idle_checks=5), now=10_000.0, head="h1")
        s = td.State(**{**s.__dict__, "idle_checks": 5})
        self.assertEqual(step(s, now=10_899.0, head="h2")[0], "wait")
        self.assertEqual(step(s, now=10_901.0, head="h2")[0], "send-tick")

    def ticks(self, heads):
        """Deliver the handoff at h1, then one tick per later head; return the action at the check after them."""
        s = td.sent(td.State(), now=0.0, head="h1")
        t = 0.0
        for head in heads:
            t += 1000
            action, s = step(td.State(**{**s.__dict__, "idle_checks": 1}), now=t, head=head)
            if action != "send-tick":
                return action
            s = td.sent(s, now=t, head=head)
        return step(td.State(**{**s.__dict__, "idle_checks": 1}), now=t + 1000, head=heads[-1])[0]

    def test_three_ticks_that_move_nothing_end_the_driver(self):
        self.assertEqual(self.ticks(["h1", "h1"]), "exit-stale")

    def test_a_commit_between_ticks_keeps_the_loop_alive(self):
        self.assertEqual(self.ticks(["h1", "h2", "h2", "h3", "h3"]), "send-tick")

    def test_the_stop_file_wins_over_an_idle_pane(self):
        s = td.sent(td.State(idle_checks=9), now=0.0, head="h0")
        self.assertEqual(step(s, stop=True, now=99_999.0)[0], "stop")


if __name__ == "__main__":
    unittest.main()
