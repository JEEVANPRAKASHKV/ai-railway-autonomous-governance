"""Test isolation:
- durable governance state (audit trail) goes to a throw-away directory;
- no test calls the internet: live weather is off (WEATHER_LIVE=0);
- no test ever calls a real (paid) LLM: real keys from a developer's .env are blanked here. Environment variables
  take precedence over .env in pydantic-settings, so the router starts in rule-based mode. Tests that need a provider
  configure a stubbed one explicitly (see tests/test_llm_providers.py)."""
import os
import tempfile

os.environ.setdefault("RB_STATE_DIR", tempfile.mkdtemp(prefix="rb_test_state_"))
for _k in ("SARVAM_API_KEY", "GEMINI_API_KEY", "AZURE_OPENAI_KEY"):
    os.environ[_k] = ""
# no network in tests: live weather falls back to climatology (tests/test_real_data.py stubs Open-Meteo explicitly)
os.environ["WEATHER_LIVE"] = "0"
