import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore

# 1. Veri yükleme veya sentetik üretim
def load_or_generate_data(filepath="Housing.csv"):
    try:
        data = pd.read_csv(filepath)
        print(f"Veri dosyası yüklendi: {filepath}")
    except:
        print("Veri bulunamadı, sentetik veri üretiliyor...")
        np.random.seed(42)
        n_samples = 1000
        sqft = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples) + np.random.random(n_samples)
        age = np.random.randint(0, 50, n_samples)
        lot_size = np.random.normal(8000, 2000, n_samples)
        price = (
            100000 + 150 * sqft + 15000 * bedrooms + 25000 * bathrooms -
            1000 * age + 2 * lot_size + np.random.normal(0, 50000, n_samples)
        )
        data = pd.DataFrame({
            "area": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": np.random.randint(1, 4, n_samples),
            "mainroad": np.random.choice(["yes", "no"], n_samples),
            "guestroom": np.random.choice(["yes", "no"], n_samples),
            "basement": np.random.choice(["yes", "no"], n_samples),
            "hotwaterheating": np.random.choice(["yes", "no"], n_samples),
            "airconditioning": np.random.choice(["yes", "no"], n_samples),
            "parking": np.random.randint(0, 4, n_samples),
            "prefarea": np.random.choice(["yes", "no"], n_samples),
            "furnishingstatus": np.random.choice(["furnished", "semi-furnished", "unfurnished"], n_samples),
            "price": price
        })
    return data

# 2. Veri ön işleme
def preprocess_data(data):
    data = data.dropna()

    # Yeni kombinasyon özellikleri (interaction terms)
    data["area_x_bedrooms"] = data["area"] * data["bedrooms"]
    data["bath_per_bed"] = data["bathrooms"] / (data["bedrooms"] + 0.1)
    data["area_x_bath"] = data["area"] * data["bathrooms"]
    data["stories_x_bath"] = data["stories"] * data["bathrooms"]
    data["parking_x_bed"] = data["parking"] * data["bedrooms"]

    # Aykırı değer temizleme
    num_cols = data.select_dtypes(include=[np.number]).columns
    data = data[(np.abs(zscore(data[num_cols])) < 3).all(axis=1)]

    # Kategorik sadeleştirme
    cat_cols = data.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        top = data[col].value_counts().nlargest(2).index
        data[col] = data[col].apply(lambda x: x if x in top else "other")

    # Fiyatın karekök dönüşümü
    data["price"] = np.sqrt(data["price"])

    X = data.drop(columns=["price"])
    y = data["price"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# 3. Model eğitimi
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 4. Değerlendirme
def evaluate_model(model, X_test, y_test, feature_names):
    y_pred_sqrt = model.predict(X_test)
    y_test_orig = y_test ** 2
    y_pred_orig = y_pred_sqrt ** 2

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)

    print(f"\n📊 Model Performansı:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    print("\n🧮 Katsayılar:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept (sqrt space): {model.intercept_:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, color="teal")
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel("Gerçek Fiyat")
    plt.ylabel("Tahmin Edilen Fiyat")
    plt.title("Gerçek vs Tahmin Fiyatları")
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png")
    print("📷 Grafik kaydedildi: prediction_vs_actual.png")

    return y_pred_orig, rmse, r2

# 5. Yeni veri ile tahmin
def predict_house_price(model, scaler, input_dict, feature_columns):
    df = pd.DataFrame([input_dict])
    df["area_x_bedrooms"] = df["area"] * df["bedrooms"]
    df["bath_per_bed"] = df["bathrooms"] / (df["bedrooms"] + 0.1)
    df["area_x_bath"] = df["area"] * df["bathrooms"]
    df["stories_x_bath"] = df["stories"] * df["bathrooms"]
    df["parking_x_bed"] = df["parking"] * df["bedrooms"]

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].iloc[0] not in ["furnished", "semi-furnished"]:
            df[col] = "other"

    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    scaled = scaler.transform(df)
    pred_sqrt = model.predict(scaled)[0]
    return pred_sqrt ** 2

# 6. Ana fonksiyon
def main():
    data = load_or_generate_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    model = train_model(X_train, y_train)
    y_pred, rmse, r2 = evaluate_model(model, X_test, y_test, feature_names)

    new_house = {
        "area": 7500,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "no",
        "furnishingstatus": "semi-furnished"
    }

    predicted_price = predict_house_price(model, scaler, new_house, feature_names)
    print("\n🏠 Örnek Tahmin:")
    print(f"Girdi: {new_house}")
    print(f"Tahmin Edilen Fiyat: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
