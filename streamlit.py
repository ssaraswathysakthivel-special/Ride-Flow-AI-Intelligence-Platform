import streamlit as st
import joblib
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# PAGE CONFIG + HEADER
# =========================
st.set_page_config(page_title="RideFlow AI Intelligence Platform", layout="wide")

st.markdown("""
<div style='background: linear-gradient(to right, #2E86C1, #5DADE2); padding: 2px; border-radius: 5px; text-align: center; color: white;'>
<h1 style='margin-bottom: 2px;'>🚗 RideFlow AI Intelligence Platform</h1>
<p style='font-size:15px; margin-top: 0;'>📊 Real-Time Demand • Supply • Cancellation Insights • Feedback Sentiment • Ride Matching AI • Chatbot</p>
</div><br><hr style='border: 1px solid #ddd;'>
""", unsafe_allow_html=True)

# =========================
# LOAD ALL MODELS
# =========================
@st.cache_resource
def load_models():
    base_path = r"C:\Users\ADMIN\Documents\Guvi_Final_Project\RideFlow_Project\Notebook"
    
    system = joblib.load(base_path + r"\rideflow_system.pkl")
    demand_model, supply_model, cancel_model = system["demand_model"], system["supply_model"], system["cancel_model"]
    
    demand_features = joblib.load(base_path + r"\demand_features.pkl")
    supply_features = joblib.load(base_path + r"\features_supply.pkl")
    cancel_features = joblib.load(base_path + r"\features_cancel.pkl")
    le_pickup = joblib.load(base_path + r"\le_pickup.pkl")
    
    eta_model = joblib.load(base_path + r"\eta_model.pkl")
    cancel_model_ai = joblib.load(base_path + r"\cancel_model.pkl")
    eta_features = joblib.load(base_path + r"\eta_features.pkl")
    cancel_features_ai = joblib.load(base_path + r"\cancel_features.pkl")
    
    le_drop = joblib.load(base_path + r"\le_drop.pkl")
    le_traffic = joblib.load(base_path + r"\le_traffic.pkl")
    le_weather = joblib.load(base_path + r"\le_weather.pkl")

    feedback_model = AutoModelForSequenceClassification.from_pretrained(base_path + r"\feedback_model")
    tokenizer = AutoTokenizer.from_pretrained(base_path + r"\feedback_model")

    return (demand_model, supply_model, cancel_model, demand_features, supply_features, cancel_features, le_pickup,
            eta_model, cancel_model_ai, eta_features, cancel_features_ai, le_drop, le_traffic, le_weather,
            feedback_model, tokenizer)

models = load_models()
(demand_model, supply_model, cancel_model, demand_features, supply_features, cancel_features, le_pickup,
 eta_model, cancel_model_ai, eta_features, cancel_features_ai, le_drop, le_traffic, le_weather,
 feedback_model, tokenizer) = models

# ===============================
# Chat Function (Defined Globally)
# ===============================
def rideflow_chatbot(user_input, ride_info):
    text = user_input.lower()
    if any(word in text for word in ["driver", "where is my driver", "ड्राइवर", "டிரைவர்"]):
        return f"Your driver is {ride_info['driver_name']}."
    elif any(word in text for word in ["fare", "price", "किराया", "கட்டணம்"]):
        return f"The estimated fare is ₹{ride_info['fare']}."
    elif any(word in text for word in ["eta", "arrival", "समय", "நேரம்"]):
        return f"The driver will arrive in {ride_info['eta']} minutes."
    else:
        return "I'm sorry, I can only help with Driver info, Fare, or ETA. Please try again!"

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.selectbox("📌 Select Module", ["Demand System", "Intelligence System"])

if menu == "Demand System":
    st.header("📊 Demand, Supply & Cancellation System")
    
 # =========================================================
# MODULE 1 - DEMAND SYSTEM
# =========================================================

    # Use columns to save vertical space
    col1, col2 = st.columns(2)

    with col1:
        # ==============================
        # 🔹 DEMAND INPUT
        # ==============================
        st.subheader("🚖 Demand Prediction")
        
        d_zone = st.selectbox("Pickup Zone (Demand)", le_pickup.classes_)
        d_hour = st.slider("Hour (Demand)", 0, 23, 10, key="d_hour")
        d_weekday = st.slider("Weekday (Demand)", 0, 6, 2, key="d_weekday")
        d_is_weekend = st.selectbox("Is Weekend? (Demand)", [False, True], key="d_weekend")
        d_weather = st.selectbox("Weather (Demand)", le_weather.classes_, key="d_weather")
        d_traffic = st.selectbox("Traffic (Demand)", le_traffic.classes_, key="d_traffic")
        
        # Predictive Inputs (Lags)
        d_lag1 = st.number_input("Lag 1 (Demand)", value=10, key="d_lag1")
        d_lag2 = st.number_input("Lag 2 (Demand)", value=8, key="d_lag2")
        d_lag24 = st.number_input("Lag 24 (Demand)", value=12, key="d_lag24")
        d_roll3 = st.number_input("Rolling Mean 3 (Demand)", value=9, key="d_roll3")

        # Demand Calculation
        demand_input = pd.DataFrame([{
            "pickup_zone_enc": le_pickup.transform([d_zone])[0],
            "hour": d_hour,
            "weekday": d_weekday,
            "is_weekend_int": 1 if d_is_weekend else 0,
            "weather_enc": le_weather.transform([d_weather])[0],
            "traffic_level_enc": le_traffic.transform([d_traffic])[0],
            "lag_1": d_lag1,
            "lag_2": d_lag2,
            "lag_24": d_lag24,
            "rolling_mean_3": d_roll3
        }])

        # Ensure features match model training exactly
        demand_input = demand_input[demand_features]
        demand = demand_model.predict(demand_input)[0]
        st.success(f"**Predicted Demand:** {int(demand)} rides")

    with col2:
        # ==============================
        # 🔹 SUPPLY INPUT
        # ==============================
        st.subheader("👨‍✈️ Supply Prediction")
        
        s_zone = st.selectbox("Pickup Zone (Supply)", le_pickup.classes_, key="s_zone")
        s_hour = st.slider("Hour (Supply)", 0, 23, 10, key="s_hour")
        s_weekday = st.slider("Weekday (Supply)", 0, 6, 2, key="s_weekday")
        s_is_weekend = st.selectbox("Is Weekend? (Supply)", [False, True], key="s_weekend")
        s_weather = st.selectbox("Weather (Supply)", le_weather.classes_, key="s_weather")
        s_traffic = st.selectbox("Traffic (Supply)", le_traffic.classes_, key="s_traffic")
        
        s_lag1 = st.number_input("Lag 1 (Supply)", value=10, key="s_lag1")
        s_lag2 = st.number_input("Lag 2 (Supply)", value=8, key="s_lag2")
        s_lag24 = st.number_input("Lag 24 (Supply)", value=12, key="s_lag24")
        s_roll3 = st.number_input("Rolling Mean 3 (Supply)", value=9, key="s_roll3")

        supply_input = pd.DataFrame([{
            "pickup_zone_enc": le_pickup.transform([s_zone])[0],
            "hour": s_hour,
            "weekday": s_weekday,
            "is_weekend_int": 1 if s_is_weekend else 0,
            "weather_enc": le_weather.transform([s_weather])[0],
            "traffic_level_enc": le_traffic.transform([s_traffic])[0],
            "lag_1": s_lag1,
            "lag_2": s_lag2,
            "lag_24": s_lag24,
            "rolling_mean_3": s_roll3
        }])

        supply_input = supply_input[supply_features]
        supply = supply_model.predict(supply_input)[0]
        st.success(f"**Predicted Supply:** {int(supply)} drivers")

    st.markdown("---")

    # ==============================
    # 🔹 CANCELLATION INPUT
    # ==============================
    st.subheader("❌ Cancellation Prediction")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        c_eta_delay = st.number_input("ETA Delay (mins)", value=5, key="c_eta")
        c_weather = st.selectbox("Weather (Cancel)", le_weather.classes_, key="c_weather")
    with c2:
        c_traffic = st.selectbox("Traffic (Cancel)", le_traffic.classes_, key="c_traffic")
        c_distance = st.number_input("Ride Distance (km)", value=8.0, key="c_dist")
    with c3:
        c_surge = st.number_input("Surge Multiplier", value=1.0, key="c_surge")

    cancel_input = pd.DataFrame([{
        "eta_delay": c_eta_delay,
        "traffic_level_enc": le_traffic.transform([c_traffic])[0],
        "weather_enc": le_weather.transform([c_weather])[0],
        "ride_distance_km": c_distance,
        "surge_multiplier": c_surge
    }])

    cancel_input = cancel_input[cancel_features]
    cancel_pred = cancel_model.predict(cancel_input)[0]

    if cancel_pred == 1:
        st.error("🚨 Warning: High Cancellation Risk")
    else:
        st.success("✅ Confidence: Low Cancellation Risk")   

# module 2 - Intelligence System

elif menu == "Intelligence System":
    st.header("🧠 AI Intelligence System")
    sub = st.selectbox("Choose Feature", ["Feedback Sentiment", "Ride Matching AI", "Chatbot"])

    if sub == "Feedback Sentiment":
        st.subheader("💬 Feedback Sentiment Analysis")
        text = st.text_area("Enter feedback")
        if st.button("Analyze"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = feedback_model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            st.success(f"Sentiment: {label_map[pred]}")

    elif sub == "Ride Matching AI":
        st.subheader("🤖 AI Ride Matching Assistant")
        # ... (Your existing Ride Matching inputs) ...
        pickup = st.selectbox("Pickup Location", le_pickup.classes_)
        drop = st.selectbox("Drop Location", le_drop.classes_)
        traffic = st.selectbox("Traffic Level", le_traffic.classes_)
        driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.5)
        ride_distance_km = st.number_input("Ride Distance (km)", min_value=0.5, value=5.0)
        hour = st.slider("Hour", 0, 23, 10)
        surge_multiplier = st.number_input("Surge Multiplier", min_value=1.0, value=1.0)

        if st.button("Find Best Driver"):
            # Calculation logic
            pickup_enc = le_pickup.transform([pickup])[0]
            drop_enc = le_drop.transform([drop])[0]
            traffic_enc = le_traffic.transform([traffic])[0]
            
            eta_input = pd.DataFrame([{"pickup_zone_enc": pickup_enc, "drop_zone_enc": drop_enc, 
                                       "traffic_level_enc": traffic_enc, "ride_distance_km": ride_distance_km, "hour": hour}])
            eta_input = eta_input.reindex(columns=eta_features, fill_value=0)
            predicted_eta = eta_model.predict(eta_input)[0]
            
            cancel_input = pd.DataFrame([{"estimated_eta_min": predicted_eta, "driver_rating": driver_rating, 
                                          "surge_multiplier": surge_multiplier, "hour": hour}])
            cancel_input = cancel_input.reindex(columns=cancel_features_ai, fill_value=0)
            cancel_prob = cancel_model_ai.predict_proba(cancel_input)[0][1]
            
            score = ((1 / (1 + predicted_eta)) * 0.4 + (1 - cancel_prob) * 0.3 + (driver_rating / 5) * 0.3)
            recommended_driver_id = round(score * 5000)
            
            st.success(f"Recommended Driver ID: {recommended_driver_id}")
            st.info(f"ETA: {round(predicted_eta,2)} mins | Risk: {round(cancel_prob,2)}")

    elif sub == "Chatbot":
        st.subheader("💬 RideFlow AI Chatbot")
        ride_data = {"driver_name": "Arun", "eta": 5, "fare": 240}
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask about your ride (Driver, Fare, ETA):", key="chat_input")

        if st.button("Send"):
            if user_input.strip():
                reply = rideflow_chatbot(user_input, ride_data)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", reply))

        for speaker, message in st.session_state.chat_history:
            if speaker == "You":
                st.markdown(f"<div style='text-align:right; background:#DCF8C6; padding:8px; border-radius:10px; margin:5px;'><b>🧑 You:</b> {message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; background:#E8E8E8; padding:8px; border-radius:10px; margin:5px;'><b>🤖 Bot:</b> {message}</div>", unsafe_allow_html=True)