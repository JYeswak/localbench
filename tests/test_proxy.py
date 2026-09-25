"""The proxy's side-call labels (proxy.purpose): which omp request is main work, which is the effort classifier, the
judge, or mnemopi's memory LLM. Bodies here quote omp 18.2.11's prompts as sent, not proxy.SIDE_CALLS."""

import os
import unittest
from pathlib import Path

from localbench.proxy import purpose

EXTRACT = ("You are a precise long-term memory extractor.\n\nExtract only persistent information explicitly stated in "
           "the user message: stable facts, explicit instructions to the assistant, stable preferences, ...")
CONSOLIDATE = ("Summarize memories in 1-3 concise sentences.\n\nPreserve every fact, name, number, version, date, and "
               "decision exactly. ...\n\nMemories:\n- the port is 4411\n- the owner is Okafor")
EFFORT = "Choose the reasoning effort this turn needs. Answer with one of: minimal, low, medium, high."

OMP_PROMPTS = Path(os.path.expanduser(
    "~/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/prompts/system"))


class Purpose(unittest.TestCase):
    def test_tools_mean_main_even_when_text_quotes_a_marker(self):
        self.assertEqual(purpose({"tools": [{}], "messages": [{"role": "user", "content": CONSOLIDATE}]}), "main")

    def test_extraction_is_matched_in_the_system_turn(self):
        body = {"messages": [{"role": "system", "content": EXTRACT}, {"role": "user", "content": "port is 4411"}]}
        self.assertEqual(purpose(body), "memory-extract")

    def test_extraction_marker_in_a_user_turn_is_not_extraction(self):
        self.assertEqual(purpose({"messages": [{"role": "user", "content": EXTRACT}]}), "aux")

    def test_consolidation_is_matched_in_the_user_turn(self):
        self.assertEqual(purpose({"messages": [{"role": "user", "content": CONSOLIDATE}]}), "memory-consolidate")

    def test_effort_classifier_wins_over_quoted_memory_text(self):
        body = {"messages": [{"role": "system", "content": EFFORT}, {"role": "user", "content": EXTRACT}]}
        self.assertEqual(purpose(body), "auto-thinking")

    def test_content_parts_are_read(self):
        body = {"messages": [{"role": "system", "content": [{"type": "text", "text": EXTRACT}]}]}
        self.assertEqual(purpose(body), "memory-extract")

    def test_unknown_side_call_stays_aux(self):
        self.assertEqual(purpose({"messages": [{"role": "system", "content": "Write a short title."}]}), "aux")


@unittest.skipUnless(OMP_PROMPTS.is_dir(), "omp is not installed under ~/.bun; the drift check needs its prompt files")
class InstalledOmpPrompts(unittest.TestCase):
    """omp ships most days: when it rewords a memory prompt, the classifier silently files the call under `aux`."""

    def test_installed_extraction_prompt_is_recognised(self):
        text = (OMP_PROMPTS / "memory-extraction-system.md").read_text()
        body = {"messages": [{"role": "system", "content": text}, {"role": "user", "content": "x"}]}
        self.assertEqual(purpose(body), "memory-extract")

    def test_installed_consolidation_prompt_is_recognised(self):
        text = (OMP_PROMPTS / "memory-consolidation-system.md").read_text()
        self.assertEqual(purpose({"messages": [{"role": "user", "content": text + "- fact"}]}), "memory-consolidate")


if __name__ == "__main__":
    unittest.main()
