import streamlit as st
import pandas as pd
import pickle


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="EV Energy Prediction",
    page_icon="⚡",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

try:

    with open("model.pkl", "rb") as file:

        model = pickle.load(file)

except FileNotFoundError:

    st.error(
        "Model file not found. "
        "Please run train_model.py first."
    )

    st.stop()


# ==========================================
# TITLE
# ==========================================

st.title(
    "⚡ EV Energy Consumption Prediction"
)

st.write(
    "Enter the EV charging station details "
    "to predict energy consumption."
)


# ==========================================
# INPUT SECTION
# ==========================================

st.subheader(
    "🔢 Enter Station Details"
)


# ==========================================
# CITY
# ==========================================

city = st.selectbox(

    "🏙️ City",

    [
        "Kochi",
        "Calicut",
        "Trivandrum",
        "Hyderabad",
        "Delhi",
        "Chennai",
        "Bengaluru",
        "Mumbai"
    ]
)


# ==========================================
# CHARGER TYPE
# ==========================================

charger_type = st.selectbox(

    "🔌 Charger Type",

    [
        "Fast",
        "DC Fast",
        "Level 2",
        "Slow"
    ]
)


# ==========================================
# VEHICLE TYPE
# ==========================================

vehicle_type = st.selectbox(

    "🚗 Vehicle Type",

    [
        "Car",
        "Bus",
        "Bike"
    ]
)


# ==========================================
# GRID LOAD
# ==========================================

grid_load = st.number_input(

    "⚡ Grid Load (kW)",

    min_value=0.0,

    max_value=1000.0,

    value=300.0,

    step=1.0
)


# ==========================================
# STATION STATUS
# ==========================================

station_status = st.selectbox(

    "🔋 Station Status",

    [
        "Active",
        "Inactive"
    ]
)


# ==========================================
# DATE
# ==========================================

date = st.date_input(

    "📅 Date"
)


# ==========================================
# PREDICT BUTTON
# ==========================================

st.divider()


if st.button(
    "🔮 Predict Energy Consumption",
    use_container_width=True
):

    # --------------------------------------
    # Convert date
    # --------------------------------------

    date = pd.to_datetime(date)

    year = date.year

    month = date.month

    day = date.day


    # --------------------------------------
    # Create input dataframe
    # --------------------------------------

    input_data = pd.DataFrame({

        "City": [
            city
        ],

        "Charger_Type": [
            charger_type
        ],

        "Vehicle_Type": [
            vehicle_type
        ],

        "Grid_Load_kW": [
            grid_load
        ],

        "Station_Status": [
            station_status
        ],

        "Year": [
            year
        ],

        "Month": [
            month
        ],

        "Day": [
            day
        ]
    })


    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------
    # Result
    # --------------------------------------

    st.success(
        "✅ Prediction Completed!"
    )


    st.subheader(
        "⚡ Predicted Energy Consumption"
    )


    st.metric(

        label="Energy Consumption",

        value=f"{prediction:.2f} kWh"
    )


    st.info(
        f"Based on the selected station details, "
        f"the predicted energy consumption is "
        f"**{prediction:.2f} kWh**."
    )