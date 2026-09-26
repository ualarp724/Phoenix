import phoenix_config as config
from phoenix_parameter_optimizer_v2 import FastParameterOptimizer


def test_optimizer_runs_minimal():
    optimizer = FastParameterOptimizer(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    results = optimizer.optimizar()
    assert results is not None
    assert len(results) > 0
