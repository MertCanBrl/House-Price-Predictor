import streamlit as st
import pandas as pd
import pickle
import os
from model import predict_house_price

# Sayfa başlığı
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.markdown("Aşağıdaki ev özelliklerini doldurun, tahmini fiyatı hesaplayalım.")

# Modeli ve scaler'ı yükle
@st.cache_resource
def load_model():
    model_path = "models/house_price_model.pkl"
    scaler_path = "models/feature_scaler.pkl"
    features_path = "models/feature_names.pkl"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        st.error("Model dosyaları eksik! Lütfen `models/` klasörünü kontrol edin.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Kullanıcıdan veri al
with st.form("input_form"):
    area = st.slider("Ev Alanı (sqft)", 500, 10000, 2000, step=100)
    bedrooms = st.slider("Yatak Odası Sayısı", 1, 10, 3)
    bathrooms = st.slider("Banyo Sayısı", 1, 5, 2)
    stories = st.selectbox("Kat Sayısı", [1, 2, 3])
    mainroad = st.selectbox("Ana yola yakın mı?", ["yes", "no"])
    guestroom = st.selectbox("Misafir odası var mı?", ["yes", "no"])
    basement = st.selectbox("Bodrum katı var mı?", ["yes", "no"])
    hotwaterheating = st.selectbox("Sıcak su sistemi var mı?", ["yes", "no"])
    airconditioning = st.selectbox("Klima var mı?", ["yes", "no"])
    parking = st.slider("Otopark alanı (araç)", 0, 4, 1)
    prefarea = st.selectbox("Tercih edilen bölgede mi?", ["yes", "no"])
    furnishingstatus = st.selectbox("Eşya durumu", ["furnished", "semi-furnished", "unfurnished"])

    submitted = st.form_submit_button("Tahmini Fiyatı Göster")

# Tahmin geçmişi oturumda saklanıyor mu, kontrol et
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Tahmin yapıldıysa tahmin geçmişine ekle
if submitted:
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    predicted_price = predict_house_price(model, scaler, input_data, feature_names)

    st.success(f"🏷️ Tahmini Ev Fiyatı: **${predicted_price:,.2f}**")

    # Geçmişe ekle
    history_entry = input_data.copy()
    history_entry["Predicted Price ($)"] = f"{predicted_price:,.2f}"
    st.session_state.prediction_history.append(history_entry)

# Girdi geçmişini göster
if st.session_state.prediction_history:
    st.subheader("📜 Tahmin Geçmişi")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df)
