import pytest
from core.config_validation import validate_config


def test_validate_config_missing_files(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    # validate_config should raise if files missing
    with pytest.raises(FileNotFoundError):
        validate_config()
