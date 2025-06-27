import pickle
import os
from model import load_or_generate_data, preprocess_data, train_model

def save_model():
    """
    Train and save the model, scaler, and feature names as pickle files
    """
    print("Loading data...")
    data = load_or_generate_data("Housing.csv")  # Dosya yoksa sentetik veri üretir

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)

    print("Training model...")
    model = train_model(X_train, y_train)

    # Save directory
    os.makedirs("models", exist_ok=True)

    with open("models/house_price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model saved to models/house_price_model.pkl")

    with open("models/feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved to models/feature_scaler.pkl")

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("✅ Feature names saved to models/feature_names.pkl")

    return True

if __name__ == "__main__":
    save_model()
