import unittest

from src.utils.repetition_detector import RepetitionDetector, RepetitionDetectorConfig


class RepetitionDetectorTests(unittest.TestCase):
    def test_ngram_counter_does_not_recount_entire_buffer_each_chunk(self):
        detector = RepetitionDetector(
            RepetitionDetectorConfig(
                ngram_size=2,
                ngram_threshold=3,
                window_size=1000,
                min_windows=10,
                check_interval=1,
                min_text_length=0,
                char_repeat_threshold=1000,
            )
        )

        # Before the fix, these incremental unique chunks falsely pushed the
        # first n-gram over threshold because the whole buffer was recounted on
        # every call.
        chunks = [
            "alpha beta gamma",
            " delta",
            " epsilon",
            " zeta",
        ]

        detections = [detector.add_chunk(chunk) for chunk in chunks]

        self.assertTrue(all(result is None for result in detections))

    def test_repeated_ngram_still_detects_real_loop(self):
        detector = RepetitionDetector(
            RepetitionDetectorConfig(
                ngram_size=2,
                ngram_threshold=3,
                window_size=1000,
                min_windows=10,
                check_interval=1,
                min_text_length=0,
                char_repeat_threshold=1000,
            )
        )

        result = None
        for chunk in ["alpha beta ", "alpha beta ", "alpha beta "]:
            result = detector.add_chunk(chunk)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_repetitive)
        self.assertEqual(result.detection_type, "ngram")


if __name__ == "__main__":
    unittest.main()
