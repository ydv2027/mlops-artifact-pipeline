import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_config(config_path='config/config.json'):
    with open(config_path) as f:
        return json.load(f)

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

def main():
    config = load_config()
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    joblib.dump(model, "model_train.pkl")

if __name__ == "__main__":
    main()
