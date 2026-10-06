import tempfile
import unittest
from pathlib import Path

from localbench.response_cache import ResponseCache, response_key


class ResponseCacheContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = ResponseCache(Path(self.tmp.name))
        self.kw = dict(request_bytes=b'{"prompt":"x"}', model_digest="model-sha",
                       backend_version="backend-1", suite_pin="suite-sha",
                       repeat_namespace="repeat-1", template_bytes=b"template-v1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_key_includes_every_semantic_input(self):
        key = response_key(**self.kw)
        variants = (
            ("request_bytes", b'{"prompt":"y"}'),
            ("model_digest", "other-model-sha"),
            ("backend_version", "backend-2"),
            ("suite_pin", "other-suite-sha"),
            ("repeat_namespace", "repeat-2"),
            ("template_bytes", b"template-v2"),
        )
        for field, value in variants:
            with self.subTest(field=field):
                self.assertNotEqual(key, response_key(**{**self.kw, field: value}))

    def test_hit_is_quality_only_not_latency_or_availability(self):
        key = response_key(**self.kw)
        self.assertTrue(self.cache.put(key, b'{"answer":"ok"}'))
        hit = self.cache.get(key)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.body, b'{"answer":"ok"}')
        self.assertEqual(hit.flags.as_dict(), {"cache_hit": True,
                                               "latency_eligible": False,
                                               "availability_eligible": False})

    def test_errors_timeouts_and_truncation_are_never_cached(self):
        key = response_key(**self.kw)
        for kwargs in ({"error_kind": "invalid"}, {"timed_out": True}, {"truncated": True}):
            self.assertFalse(self.cache.put(key, b"bad", **kwargs))
            self.assertIsNone(self.cache.get(key))

    def test_corrupt_entry_is_a_miss(self):
        key = response_key(**self.kw)
        path = self.cache._path(key)
        path.parent.mkdir(parents=True)
        path.write_text("not json")
        self.assertIsNone(self.cache.get(key))

    def test_live_flags_are_countable(self):
        self.assertEqual(ResponseCache.live_flags().as_dict(), {"cache_hit": False,
                                                                 "latency_eligible": True,
                                                                 "availability_eligible": True})


if __name__ == "__main__":
    unittest.main()
