import json
import joblib
import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model

def test_config_loading():
    config = load_config()
    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)

def test_model_training():
    config = load_config()
    digits = load_digits()
    model = train_model(digits.data, digits.target, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_model_accuracy_threshold():
    config = load_config()
    digits = load_digits()
    model = train_model(digits.data, digits.target, config)
    acc = model.score(digits.data, digits.target)
    assert acc > 0.8
