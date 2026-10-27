"""Test isolation:
- durable governance state (audit trail) goes to a throw-away directory;
- no test ever calls a real (paid) LLM: real keys from a developer's .env are blanked here. Environment variables
  take precedence over .env in pydantic-settings, so the router starts in rule-based mode. Tests that need a provider
  configure a stubbed one explicitly (see tests/test_llm_providers.py)."""
import os
import tempfile

os.environ.setdefault("RB_STATE_DIR", tempfile.mkdtemp(prefix="rb_test_state_"))
for _k in ("SARVAM_API_KEY", "GEMINI_API_KEY", "AZURE_OPENAI_KEY"):
    os.environ[_k] = ""
