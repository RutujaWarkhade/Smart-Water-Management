import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ─── LOAD MODELS ─────────────────────────────────────────
model = joblib.load("D:/Water_Management/models/best_model.pkl")
scaler = joblib.load("D:/Water_Management/models/scaler.pkl")
kmeans = joblib.load("D:/Water_Management/models/kmeans.pkl")
cluster_scaler = joblib.load("D:/Water_Management/models/cluster_scaler.pkl")

# ─── LOAD DATASET ────────────────────────────────────────
df = pd.read_csv("D:/Water_Management/water_consumption.csv")
segmented_df = pd.read_csv("D:/Water_Management/outputs/segmented_consumers.csv")

st.set_page_config(page_title="Smart Water Dashboard", layout="wide")

# ─── SIDEBAR ─────────────────────────────────────────────
st.sidebar.title("⚙️ Input Parameters")

temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 30)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 10)
household = st.sidebar.slider("Household Size", 1, 10, 4)
hist_avg = st.sidebar.number_input("Last 3 Days Avg Usage", value=200.0)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
is_weekend = st.sidebar.selectbox("Weekend?", [0, 1])

# ─── TITLE ───────────────────────────────────────────────
st.title("💧 Smart Water Management Dashboard")
st.markdown("AI-powered system for water demand prediction & consumer segmentation")

# ─── PREDICTION ──────────────────────────────────────────
if st.sidebar.button("Predict"):

    # Prepare input
    input_data = np.array([[temperature, rainfall, household,
                            hist_avg, is_weekend, month]])

    # Predict usage
    prediction = model.predict(input_data)[0]

    # Clustering
    cluster_input = np.array([[prediction, household, hist_avg, month]])
    cluster_scaled = cluster_scaler.transform(cluster_input)
    cluster = kmeans.predict(cluster_scaled)[0]

    # Segment mapping
    if cluster == 0:
        segment = "Low Consumption"
    elif cluster == 1:
        segment = "Seasonal User"
    else:
        segment = "High Consumption"

    # ─── METRICS ─────────────────────────────────────────
    col1, col2 = st.columns(2)
    col1.metric("💧 Predicted Usage (Liters)", f"{prediction:.2f}")
    col2.metric("👤 Consumer Type", segment)

    # ─── BAR CHART ───────────────────────────────────────
    st.subheader("📊 Input Feature Overview")

    df_input = pd.DataFrame({
        "Feature": ["Temperature", "Rainfall", "Household", "Past Usage"],
        "Value": [temperature, rainfall, household, hist_avg]
    })

    fig1, ax1 = plt.subplots()
    ax1.bar(df_input["Feature"], df_input["Value"])
    ax1.set_title("Input Features")
    st.pyplot(fig1)

    # ─── REAL PIE CHART (SEGMENTS) ───────────────────────
    st.subheader("📈 Consumer Distribution (Real Data)")

    segment_counts = segmented_df["segment"].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(segment_counts.values,
            labels=segment_counts.index,
            autopct="%1.1f%%")

    st.pyplot(fig2)

    # ─── REAL TREND GRAPH ────────────────────────────────
    st.subheader("📉 Weekly Water Usage Trend")

    # Convert day numbers → names
    day_mapping = {
        0: "Mon", 1: "Tue", 2: "Wed",
        3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
    }

    df["day_name"] = df["day_of_week"].map(day_mapping)

    trend_data = df.groupby("day_name")["daily_usage_liters"].mean()

    # Correct order
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    trend_data = trend_data.reindex(order)

    fig3, ax3 = plt.subplots()
    ax3.plot(trend_data.index, trend_data.values, marker='o')

    ax3.set_title("Average Daily Water Usage")
    ax3.set_xlabel("Day of Week")
    ax3.set_ylabel("Liters")

    st.pyplot(fig3)

    # ─── AI INSIGHTS ─────────────────────────────────────
    st.subheader("🧠 AI Insights")

    if segment == "High Consumption":
        st.error("⚠️ High water usage detected. Suggest installing water-saving devices.")
    elif segment == "Seasonal User":
        st.warning("🌦 Usage varies seasonally. Monitor during summer months.")
    else:
        st.success("✅ Efficient usage. Keep maintaining this pattern!")

# ─── FOOTER ─────────────────────────────────────────────
st.markdown("---")
st.caption("🚀 Smart Water Management | AI/ML Project | Streamlit Dashboard")