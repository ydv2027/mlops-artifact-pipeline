import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

def inference():
    model = joblib.load("model_train.pkl")
    digits = load_digits()
    X, y = digits.data, digits.target
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"Inference Accuracy: {acc:.4f}")

if __name__ == "__main__":
    inference()
