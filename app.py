import streamlit as st
import pandas as pd
import pickle


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EV Energy Prediction",
    page_icon="⚡",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()


# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("EV_Charging_Grid_Load_Dataset_5000.csv")


df = load_data()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("⚡ EV Energy System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Visualisation",
        "ℹ️ About"
    ]
)


# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.title("⚡ EV Energy Consumption Prediction")

    st.markdown("""
    ### Smart EV Charging Analytics

    This application uses **Machine Learning** to predict
    energy consumption in Electric Vehicle charging stations.

    Enter charging station information and get an
    estimated energy consumption value.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Prediction Model",
            "ML Regression"
        )

    with col3:
        st.metric(
            "Application",
            "EV Energy Analytics"
        )

    st.divider()

    st.subheader("🚀 Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔮 Energy Prediction**

        Predict expected energy consumption
        using charging station information.

        **📊 Data Visualisation**

        Explore EV charging data using
        interactive charts.
        """)

    with col2:
        st.markdown("""
        **⚡ Smart Analytics**

        Understand charging and grid-load patterns.

        **🌐 Web Application**

        Simple and user-friendly Streamlit interface.
        """)


# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "🔮 Prediction":

    st.title("🔮 Energy Consumption Prediction")

    st.write(
        "Enter the charging station details below "
        "to predict energy consumption."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        city = st.selectbox(
            "🏙️ City",
            sorted(df["City"].dropna().unique())
        )

        charger_type = st.selectbox(
            "🔌 Charger Type",
            sorted(df["Charger_Type"].dropna().unique())
        )

        vehicle_type = st.selectbox(
            "🚗 Vehicle Type",
            sorted(df["Vehicle_Type"].dropna().unique())
        )

    with col2:

        grid_load = st.number_input(
            "⚡ Grid Load (kW)",
            min_value=0.0,
            value=float(df["Grid_Load_kW"].median()),
            step=1.0
        )

        station_status = st.selectbox(
            "🔋 Station Status",
            sorted(df["Station_Status"].dropna().unique())
        )

        selected_date = st.date_input(
            "📅 Date"
        )

    st.divider()

    if st.button(
        "🔮 Predict Energy Consumption",
        width="stretch"
    ):

        date_value = pd.to_datetime(selected_date)

        input_data = pd.DataFrame({
            "City": [city],
            "Charger_Type": [charger_type],
            "Vehicle_Type": [vehicle_type],
            "Grid_Load_kW": [grid_load],
            "Station_Status": [station_status],
            "Year": [date_value.year],
            "Month": [date_value.month],
            "Day": [date_value.day],
            "DayOfYear": [date_value.dayofyear]
        })

        prediction = model.predict(input_data)[0]

        st.success(
            "Prediction completed successfully!"
        )

        st.metric(
            "⚡ Predicted Energy Consumption",
            f"{prediction:.2f} kWh"
        )


# ==================================================
# VISUALISATION PAGE
# ==================================================

elif page == "📊 Visualisation":

    st.title("📊 EV Charging Data Visualisation")

    st.write(
        "Explore the EV charging dataset and "
        "understand energy consumption patterns."
    )

    st.divider()

    # ----------------------------------------------
    # DATASET PREVIEW
    # ----------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.divider()

    # ----------------------------------------------
    # CITY-WISE ENERGY
    # ----------------------------------------------

    st.subheader(
        "🏙️ Average Energy Consumption by City"
    )

    city_energy = (
        df.groupby("City")["Energy_Consumed_kWh"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(city_energy)

    # ----------------------------------------------
    # VEHICLE TYPE
    # ----------------------------------------------

    st.subheader(
        "🚗 Energy Consumption by Vehicle Type"
    )

    vehicle_energy = (
        df.groupby("Vehicle_Type")[
            "Energy_Consumed_kWh"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(vehicle_energy)

    # ----------------------------------------------
    # CHARGER TYPE
    # ----------------------------------------------

    st.subheader(
        "🔌 Energy Consumption by Charger Type"
    )

    charger_energy = (
        df.groupby("Charger_Type")[
            "Energy_Consumed_kWh"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(charger_energy)

    # ----------------------------------------------
    # GRID LOAD VS ENERGY
    # ----------------------------------------------

    st.subheader(
        "⚡ Grid Load vs Energy Consumption"
    )

    chart_data = df[
        [
            "Grid_Load_kW",
            "Energy_Consumed_kWh"
        ]
    ]

    st.scatter_chart(
        chart_data,
        x="Grid_Load_kW",
        y="Energy_Consumed_kWh"
    )


# ==================================================
# ABOUT PAGE
# ==================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## ⚡ EV Energy Consumption Prediction

    This project is a **Machine Learning based web
    application** designed to predict energy consumption
    at Electric Vehicle charging stations.

    ### 🎯 Objective

    The main objective is to provide an easy-to-use
    system for estimating EV charging energy consumption.

    ### 🛠️ Technologies Used

    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Machine Learning
    - GitHub

    ### 🤖 Machine Learning

    The application uses a regression model to estimate:

    **Energy Consumed (kWh)**

    based on:

    - City
    - Charger Type
    - Vehicle Type
    - Grid Load
    - Station Status
    - Date

    ### 📊 Dataset

    The application uses an EV charging and
    grid-load dataset containing charging
    station records.

    ### 🌐 Web Application

    This project demonstrates how a Machine Learning
    model can be converted into an interactive
    web application using Streamlit.
    """)

    st.divider()

    st.info(
        "⚡ Built as a Data Science / Machine Learning project."
    )