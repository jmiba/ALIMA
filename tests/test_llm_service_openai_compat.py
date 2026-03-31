import unittest

try:
    from src.llm.llm_service import LlmService
    PYQT_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    LlmService = None
    PYQT_IMPORT_ERROR = exc


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6-backed LLM service unavailable: {PYQT_IMPORT_ERROR}")
class TestOpenAICompatibleSamplingRetry(unittest.TestCase):
    def test_retries_when_provider_rejects_custom_temperature(self):
        params = {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_p": 0.1,
        }
        exc = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"Unsupported value: 'temperature' does not support 0.7 with this model. "
            "Only the default (1) value is supported.\", "
            "'type': 'invalid_request_error', 'param': 'temperature', 'code': 'unsupported_value'}}"
        )

        should_retry = LlmService._should_retry_openai_without_sampling(exc, params)

        self.assertTrue(should_retry)

    def test_does_not_retry_for_unrelated_provider_error(self):
        params = {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_p": 0.1,
        }
        exc = Exception("Error code: 401 - invalid_api_key")

        should_retry = LlmService._should_retry_openai_without_sampling(exc, params)

        self.assertFalse(should_retry)


if __name__ == "__main__":
    unittest.main()
