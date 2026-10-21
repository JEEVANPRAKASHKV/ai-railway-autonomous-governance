"""Test isolation: durable governance state (audit trail) goes to a throw-away directory."""
import os
import tempfile

os.environ.setdefault("RB_STATE_DIR", tempfile.mkdtemp(prefix="rb_test_state_"))
