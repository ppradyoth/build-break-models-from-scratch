from ctf.providers.keys import redact, resolve_key

_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
]


def test_redact_masks_api_keys():
    s = "Authorization: Bearer sk-abcdEFGH1234567890abcdEFGH"
    out = redact(s)
    assert "sk-abcd" not in out
    assert "REDACTED" in out


def test_redact_leaves_prose_alone():
    s = "the quick brown fox jumps over the lazy dog"
    assert redact(s) == s


def test_resolve_key_none_without_config(monkeypatch, tmp_path):
    for e in _ENV_VARS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    assert resolve_key("openai") is None
