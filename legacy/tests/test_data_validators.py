import pandas as pd
import pytest
import phoenix_config as config
from core.data_validators import validate_features, validate_no_nulls, validate_target


def test_validate_features_missing():
    df = pd.DataFrame({"RSI": [1], "Target": [0]})
    with pytest.raises(ValueError):
        validate_features(df)


def test_validate_no_nulls():
    data = {f: [1] for f in config.FEATURES}
    data[config.FEATURES[0]] = [None]
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        validate_no_nulls(df)


def test_validate_target():
    df = pd.DataFrame({f: [1] for f in config.FEATURES})
    with pytest.raises(ValueError):
        validate_target(df)
