import streamlit as st
import requests

st.set_page_config(page_title="Voyage Analytics", layout="centered")

# =========================
# HEADER
# =========================
st.title("✈️ Voyage Analytics")
st.caption("AI-powered Travel Recommendation & Prediction System 🚀")

# =========================
# API BASE URL
# =========================
API_URL = "https://voyage-backend-09jx.onrender.com" 

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "🌍 Recommendations",
    "💰 Price Prediction",
    "👤 Gender Prediction"
])

# =========================
# 🌍 RECOMMENDATIONS
# =========================
with tab1:
    st.subheader("Get Personalized Travel Recommendations")

    user_id = st.number_input("Enter User ID", min_value=0, step=1)

    if st.button("Get Recommendations", key="rec_btn"):
        with st.spinner("Fetching recommendations..."):
            try:
                res = requests.get(f"{API_URL}/recommend-trip?user_id={user_id}")
                data = res.json()

                if "error" in data:
                    st.error(data["error"])
                elif not data:
                    st.warning("No recommendations found")
                else:
                    st.success("Recommendations Loaded ✅")

                    for place, hotels in data.items():
                        st.markdown(f"### 📍 {place}")
                        for hotel in hotels:
                            st.write(f"🏨 {hotel['name']} (Popularity: {hotel['count']})")

            except Exception as e:
                st.error("Backend not reachable")
                st.text(str(e))

# =========================
# 💰 PRICE PREDICTION
# =========================
with tab2:
    st.subheader("Predict Flight Price")

    distance = st.slider("Distance", 100, 5000, 500)
    flightType = st.selectbox("Flight Type", ["Economy", "Business"])
    agency = st.selectbox("Agency", [0, 1, 2, 3])
    from_city = st.number_input("From (encoded)", min_value=0)
    to_city = st.number_input("To (encoded)", min_value=0)

    if st.button("Predict Price", key="price_btn"):
        with st.spinner("Predicting..."):
            try:
                payload = {
                    "distance": distance,
                    "flightType": 0 if flightType == "Economy" else 1,
                    "agency": agency,
                    "from": from_city,
                    "to": to_city
                }

                res = requests.post(f"{API_URL}/predict-price", json=payload)
                result = res.json()

                if "predicted_price" in result:
                    st.success(f"💸 Price: ₹ {round(result['predicted_price'], 2)}")
                else:
                    st.error(result.get("error", "Prediction failed"))

            except Exception as e:
                st.error(str(e))

# =========================
# 👤 GENDER PREDICTION
# =========================
with tab3:
    st.subheader("Predict User Gender")

    avg_price = st.number_input("Average Price", min_value=0.0)
    max_price = st.number_input("Max Price", min_value=0.0)
    avg_distance = st.number_input("Avg Distance", min_value=0.0)
    avg_time = st.number_input("Avg Time", min_value=0.0)
    total_trips = st.number_input("Total Trips", min_value=0)

    if st.button("Predict Gender", key="gender_btn"):
        with st.spinner("Predicting..."):
            try:
                payload = {
                    "avg_price": avg_price,
                    "max_price": max_price,
                    "avg_distance": avg_distance,
                    "avg_time": avg_time,
                    "total_trips": total_trips
                }

                res = requests.post(f"{API_URL}/predict-gender", json=payload)
                result = res.json()

                if "predicted_gender" in result:
                    gender = "Male" if result["predicted_gender"] == 0 else "Female"
                    st.success(f"👤 {gender}")
                else:
                    st.error(result.get("error", "Prediction failed"))

            except Exception as e:
                st.error(str(e))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with FastAPI + Streamlit + ML 🚀")
