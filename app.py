import streamlit as st
import pandas as pd
import pickle
import os
from model import predict_house_price  # Kendi model dosyan olmalı

# Sayfa başlığı
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.markdown("Aşağıdaki ev özelliklerini doldurun, tahmini fiyatı hesaplayalım.")

# Model ve yardımcı dosyaları yükle
@st.cache_resource
def load_model():
    paths = {
        "model": "models/house_price_model.pkl",
        "scaler": "models/feature_scaler.pkl",
        "features": "models/feature_names.pkl",
        "city_scaler": "models/city_scaler.pkl",
        "district_scaler": "models/district_scaler.pkl"
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            st.error(f"Eksik dosya: {path}")
            st.stop()

    with open(paths["model"], "rb") as f:
        model = pickle.load(f)
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["features"], "rb") as f:
        feature_names = pickle.load(f)
    with open(paths["city_scaler"], "rb") as f:
        scaler_city = pickle.load(f)
    with open(paths["district_scaler"], "rb") as f:
        scaler_district = pickle.load(f)

    return model, scaler, feature_names, scaler_city, scaler_district

# İl-İlçe verisini yükle
@st.cache_data
def load_city_district_data():
    df = pd.read_csv("il_ilce.csv")
    df["il"] = df["il"].str.strip().str.title()
    df["ilce"] = df["ilce"].str.strip().str.title()
    return df.groupby("il")["ilce"].apply(list).to_dict()

# Verileri yükle
city_district_map = load_city_district_data()
model, scaler, feature_names, scaler_city, scaler_district = load_model()

# Tahmin geçmişi
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Kullanıcıdan veri al
city = st.selectbox("İl", sorted(city_district_map.keys()))
district_options = city_district_map.get(city, [])
district = st.selectbox("İlçe", district_options, key=f"district_select_{city}")

area = st.slider("Ev Alanı (m2)", 30, 500, 100)
bedrooms = st.slider("Oda Sayısı", 1, 10, 3)
building_age = st.slider("Bina Yaşı", 0, 30, 10)
floor = st.slider("Kat", 0, 20, 2)
heating = st.selectbox("Isıtma", ["Kombi", "Merkezi", "Soba"])
balcony = st.selectbox("Balkon", ["Var", "Yok"])
elevator = st.selectbox("Asansör", ["Var", "Yok"])
furnished = st.selectbox("Eşyalı mı?", ["Evet", "Hayır"])

# Tahmin butonu
if st.button("Tahmini Fiyatı Göster"):
    input_data = {
        "city": city,
        "district": district,
        "area": area,
        "bedrooms": bedrooms,
        "building_age": building_age,
        "floor": floor,
        "heating": heating,
        "balcony": balcony,
        "elevator": elevator,
        "furnished": furnished
    }

    # Ortalama fiyat haritaları
    df = pd.read_csv("HousingPrice.csv")
    city_avg_price_map = df.groupby("İl")["Fiyat (TL)"].mean().to_dict()
    district_avg_price_map = df.groupby("İlçe")["Fiyat (TL)"].mean().to_dict()

    # Tahmin
    predicted_price = predict_house_price(
        model, scaler, input_data, feature_names,
        city_avg_price_map, scaler_city, district_avg_price_map, scaler_district
    )

    st.success(f"🏷️ Tahmini Ev Fiyatı: ₺{predicted_price:,.2f}")

    # Grafik varsa göster
    if os.path.exists("prediction_vs_actual.png"):
        st.image("prediction_vs_actual.png", caption="📈 Gerçek vs Tahmin Fiyatları", use_container_width=True)

    # Geçmişe ekle
    history_entry = input_data.copy()
    history_entry["Tahmini Fiyat (₺)"] = f"{predicted_price:,.2f}"
    st.session_state.prediction_history.append(history_entry)

# Geçmişi göster
if st.session_state.prediction_history:
    st.subheader("📜 Tahmin Geçmişi")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)
