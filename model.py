import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import zscore

def load_or_generate_data(filepath="HousingPrice.csv"):
    data = pd.read_csv(filepath)
    print(f"Veri dosyası yüklendi: {filepath}")
    return data

def preprocess_data(data):
    data = data.rename(columns={
        "İl": "city", "İlçe": "district", "Alan (m2)": "area",
        "Oda Sayısı": "bedrooms", "Bina Yaşı": "building_age", "Kat": "floor",
        "Isıtma": "heating", "Balkon": "balcony", "Asansör": "elevator",
        "Eşyalı mı?": "furnished", "Fiyat (TL)": "price"
    })

    data["bedrooms"] = data["bedrooms"].astype(str).str.extract(r"(\d+)").astype(float)
    for col in ["area", "bedrooms", "building_age", "floor", "price"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data["furnishingstatus"] = data["furnished"].map({"Evet": "furnished", "Hayır": "unfurnished"})
    data = data.dropna()
    data = data[(np.abs(zscore(data[["price", "area"]])) < 3).all(axis=1)]

    data["city_mean_price"] = data.groupby("city")["price"].transform("mean")
    log_city_prices = np.log1p(data["city_mean_price"])
    scaler_city = StandardScaler()
    data["city_encoded"] = scaler_city.fit_transform(log_city_prices.to_frame()) * 0.000001

    data["district_mean_price"] = data.groupby("district")["price"].transform("mean")
    log_district_prices = np.log1p(data["district_mean_price"])
    scaler_district = StandardScaler()
    data["district_encoded"] = scaler_district.fit_transform(log_district_prices.to_frame()) * 0.5

    data = data.drop(columns=["city_mean_price", "district_mean_price", "district", "city"])

    data["price"] = np.log1p(data["price"])
    data["area"] = np.log1p(data["area"])
    data["building_age"] = np.log1p(data["building_age"] + 1)
    data["floor"] = np.log1p(data["floor"] + 1)
    data["log_bedrooms"] = np.log1p(data["bedrooms"])

    data["area_x_bed"] = data["area"] * data["bedrooms"]
    data["room_density"] = data["area"] / (data["bedrooms"] + 0.1)
    data["floor_x_age"] = data["floor"] * data["building_age"]
    data["area_per_floor"] = data["area"] / (data["floor"] + 1)

    data = pd.get_dummies(data, columns=["heating", "balcony", "elevator", "furnishingstatus"], drop_first=False)

    if "balcony_Var" in data.columns:
        data["balcony_var_weighted"] = data["balcony_Var"] * 10
    if "elevator_Var" in data.columns:
        data["elevator_var_weighted"] = data["elevator_Var"] * 10
    if "furnishingstatus_furnished" in data.columns:
        data["furnished_weighted"] = data["furnishingstatus_furnished"] * 10

    X = data.drop(columns=["price", "furnished"])
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(pd.DataFrame(X_train, columns=X.columns))
    X_test_scaled = scaler.transform(pd.DataFrame(X_test, columns=X.columns))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns, scaler_city, scaler_district

def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100],
        'max_depth': [7],
        'learning_rate': [0.05],
        'subsample': [0.8],
        'colsample_bytree': [1]
    }
    base_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(base_model, param_grid, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("\n🔍 En iyi XGBoost hiperparametreleri:")
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    y_pred_log = model.predict(X_test)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig)
    accuracy_percent = np.mean(np.abs(y_pred_orig - y_test_orig) / y_test_orig <= 0.20) * 100

    print(f"\n📊 Model Performansı:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE (Ortalama Hata Yüzdesi): {mape:.2%}")
    print(f"%20 toleransla doğruluk: %{accuracy_percent:.2f}")

    print("\n🧮 Özellik Önem Skoru (RandomForest):")
    for name, score in sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"{name}: {score:.4f}")

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

def predict_house_price(model, scaler, input_dict, feature_columns, city_avg_price_map, scaler_city, district_avg_price_map, scaler_district):
    df = pd.DataFrame([input_dict])
    df["furnishingstatus"] = df["furnished"].map({"Evet": "furnished", "Hayır": "unfurnished"})

    # Şehir ve ilçe log fiyatlarının hesaplanması (dict'e mean uygulanamaz hatası giderildi)
    default_city_price = np.mean(list(city_avg_price_map.values()))
    city_log_price = np.log1p(city_avg_price_map.get(df["city"].iloc[0], default_city_price))
    df["city_encoded"] = scaler_city.transform([[city_log_price]]) * 0.000001

    default_district_price = np.mean(list(district_avg_price_map.values()))
    district_log_price = np.log1p(district_avg_price_map.get(df["district"].iloc[0], default_district_price))
    df["district_encoded"] = scaler_district.transform([[district_log_price]]) * 0.5

    # Sayısal dönüşümler
    df["area"] = np.log1p(df["area"])
    df["building_age"] = np.log1p(df["building_age"] + 1)
    df["floor"] = np.log1p(df["floor"] + 1)
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    df["log_bedrooms"] = np.log1p(df["bedrooms"])

    # Yeni öznitelikler
    df["area_x_bed"] = df["area"] * df["bedrooms"]
    df["room_density"] = df["area"] / (df["bedrooms"] + 0.1)
    df["floor_x_age"] = df["floor"] * df["building_age"]
    df["area_per_floor"] = df["area"] / (df["floor"] + 1)

    # Kategorik değişkenleri one-hot encode et
    df = pd.get_dummies(df)

    # Eksik sütunları doldur
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        filler_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, filler_df], axis=1)

    # Opsiyonel: ağırlıklı sütunlar
    if "balcony_Var" in df.columns:
        df["balcony_var_weighted"] = df["balcony_Var"] * 10
    if "elevator_Var" in df.columns:
        df["elevator_var_weighted"] = df["elevator_Var"] * 10
    if "furnishingstatus_furnished" in df.columns:
        df["furnished_weighted"] = df["furnishingstatus_furnished"] * 10

    # Sıralama
    df = df[feature_columns]
    df = pd.DataFrame(df, columns=feature_columns)

    # Ölçeklendirme ve tahmin
    scaled = scaler.transform(df)
    pred_log = model.predict(scaled)[0]
    return np.expm1(pred_log)

def main():
    data = load_or_generate_data()
    city_avg_price_map = data.groupby("İl")["Fiyat (TL)"].mean()
    district_avg_price_map = data.groupby("İlçe")["Fiyat (TL)"].mean()

    X_train, X_test, y_train, y_test, scaler, feature_names, scaler_city, scaler_district = preprocess_data(data)
    model = train_model(X_train, y_train)
    y_pred, rmse, r2 = evaluate_model(model, X_test, y_test, feature_names)

    new_house = {
        "city": "Adana", "district": "Seyhan", "area": 130, "bedrooms": 3,
        "building_age": 10, "floor": 2, "heating": "Kombi", "balcony": "Var",
        "elevator": "Yok", "furnished": "Evet"
    }
    predicted_price = predict_house_price(model, scaler, new_house, feature_names, city_avg_price_map, scaler_city, district_avg_price_map, scaler_district)
    print("\n🏠 Örnek Tahmin:")
    print(f"Girdi: {new_house}")
    print(f"Tahmin Edilen Fiyat: ₺{predicted_price:,.2f}")

if __name__ == "__main__":
    main()
