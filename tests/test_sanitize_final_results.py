import json

from scripts.sanitize_final_results import KEEP, sanitize


def test_sanitizer_is_deterministic_and_excludes_provider_payloads(tmp_path):
    first = sanitize(output_dir=tmp_path / "a", mapping_path=tmp_path / "a" / "map.json")
    second = sanitize(output_dir=tmp_path / "b", mapping_path=tmp_path / "b" / "map.json")
    assert first["providers"].keys() == {"openai", "anthropic", "deepseek", "gemini", "kimi", "grok"}
    assert set(first["providers"]) == set(second["providers"])
    for provider in first["providers"]:
        assert first["providers"][provider]["private_source_sha256"] == second["providers"][provider]["private_source_sha256"]
        assert first["providers"][provider]["row_count"] == second["providers"][provider]["row_count"]
    for provider in first["providers"]:
        left = (tmp_path / "a" / f"{provider}.jsonl").read_bytes()
        right = (tmp_path / "b" / f"{provider}.jsonl").read_bytes()
        assert left == right
        row = json.loads(left.splitlines()[0])
        assert set(row) <= set(KEEP)
        assert "raw_response" not in row
        assert "request_id" not in row
