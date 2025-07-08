import pickle
import os
from model import load_or_generate_data, preprocess_data, train_model

def save_model():
    print("Loading data...")
    data = load_or_generate_data("HousingPrice.csv")

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names, scaler_city, scaler_district = preprocess_data(data)

    print("Training model...")
    model = train_model(X_train, y_train)

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

    with open("models/city_scaler.pkl", "wb") as f:
        pickle.dump(scaler_city, f)
    print("✅ City scaler saved to models/city_scaler.pkl")

    with open("models/district_scaler.pkl", "wb") as f:
        pickle.dump(scaler_district, f)
    print("✅ District scaler saved to models/district_scaler.pkl")

    return True

if __name__ == "__main__":
    save_model()
