import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Energy Prediction",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "EV_Charging_Grid_Load_Dataset_Improved.csv"
    )


df = load_data()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

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


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title(
        "⚡ EV Energy Consumption Prediction"
    )

    st.markdown("""
    ### Smart EV Charging Analytics

    This application uses **Machine Learning**
    to predict energy consumption at Electric
    Vehicle charging stations.

    Enter charging station information and get
    an estimated energy consumption value.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📊 Dataset Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "🤖 Prediction Model",
            "Random Forest"
        )

    with col3:

        st.metric(
            "📈 R² Score",
            "0.9792"
        )

    st.divider()

    st.subheader("🚀 Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 🔮 Energy Prediction

        Predict expected energy consumption
        using charging station information.

        **Input Features:**

        - City
        - Charger Type
        - Vehicle Type
        - Grid Load
        - Station Status
        - Date
        """)

    with col2:

        st.markdown("""
        ### 📊 Data Visualisation

        Explore EV charging data using
        interactive charts.

        **Available Analysis:**

        - City-wise energy consumption
        - Vehicle-wise energy consumption
        - Charger-wise energy consumption
        - Grid Load vs Energy
        - Station Status
        - Energy distribution
        """)

    st.divider()

    st.info(
        "⚡ This application uses a Random Forest "
        "Regression model for EV energy prediction."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Energy Consumption Prediction"
    )

    st.write(
        "Enter the charging station details below "
        "to predict energy consumption."
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        city = st.selectbox(
            "🏙️ City",
            sorted(
                df["City"]
                .dropna()
                .unique()
            )
        )

        charger_type = st.selectbox(
            "🔌 Charger Type",
            sorted(
                df["Charger_Type"]
                .dropna()
                .unique()
            )
        )

        vehicle_type = st.selectbox(
            "🚗 Vehicle Type",
            sorted(
                df["Vehicle_Type"]
                .dropna()
                .unique()
            )
        )

    with col2:

        grid_load = st.number_input(
            "⚡ Grid Load (kW)",
            min_value=0.0,
            max_value=500.0,
            value=float(
                df["Grid_Load_kW"].median()
            ),
            step=1.0
        )

        station_status = st.selectbox(
            "🔋 Station Status",
            sorted(
                df["Station_Status"]
                .dropna()
                .unique()
            )
        )

        selected_date = st.date_input(
            "📅 Date"
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Energy Consumption",
        width="stretch"
    ):

        date_value = pd.to_datetime(
            selected_date
        )

        # ====================================================
        # INPUT DATA
        #
        # IMPORTANT:
        # Current model expects:
        # Year, Month, Day, DayOfYear
        # ====================================================

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
                date_value.year
            ],

            "Month": [
                date_value.month
            ],

            "Day": [
                date_value.day
            ],

            "DayOfYear": [
                date_value.dayofyear
            ]
        })

        # ====================================================
        # PREDICTION
        # ====================================================

        try:

            prediction = model.predict(
                input_data
            )[0]

            # Prevent negative prediction
            prediction = max(
                0,
                prediction
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            st.success(
                "✅ Prediction completed successfully!"
            )

            st.divider()

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "⚡ Predicted Energy",
                    f"{prediction:.2f} kWh"
                )

            with col2:

                st.metric(
                    "🤖 Model",
                    "Random Forest"
                )

            with col3:

                st.metric(
                    "📈 R² Score",
                    "0.9792"
                )

            st.divider()

            # ------------------------------------------------
            # PREDICTION DETAILS
            # ------------------------------------------------

            st.subheader(
                "📋 Prediction Details"
            )

            result_df = pd.DataFrame({

                "Parameter": [
                    "City",
                    "Charger Type",
                    "Vehicle Type",
                    "Grid Load",
                    "Station Status",
                    "Date",
                    "Predicted Energy"
                ],

                "Value": [

                    city,

                    charger_type,

                    vehicle_type,

                    f"{grid_load:.2f} kW",

                    station_status,

                    str(selected_date),

                    f"{prediction:.2f} kWh"
                ]
            })

            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True
            )

            st.divider()

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "💡 Prediction Summary"
            )

            st.write(
                f"""
                Based on the selected charging station
                information, the estimated energy
                consumption is:

                ### ⚡ {prediction:.2f} kWh

                The prediction was generated using
                the **Random Forest Regression model**.
                """
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# VISUALISATION PAGE
# ============================================================

elif page == "📊 Visualisation":

    st.title(
        "📊 EV Charging Data Visualisation"
    )

    st.write(
        "Explore the EV charging dataset and "
        "understand energy consumption patterns."
    )

    st.divider()

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader(
        "📈 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📋 Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "🏙️ Cities",
            df["City"].nunique()
        )

    with col3:

        st.metric(
            "🔌 Charger Types",
            df["Charger_Type"].nunique()
        )

    with col4:

        st.metric(
            "🚗 Vehicle Types",
            df["Vehicle_Type"].nunique()
        )

    st.divider()

    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        width="stretch",
        hide_index=True
    )

    st.divider()

    # ========================================================
    # CITY-WISE ENERGY
    # ========================================================

    st.subheader(
        "🏙️ Average Energy Consumption by City"
    )

    city_energy = (
        df.groupby("City")[
            "Energy_Consumed_kWh"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        city_energy
    )

    st.divider()

    # ========================================================
    # VEHICLE TYPE
    # ========================================================

    st.subheader(
        "🚗 Energy Consumption by Vehicle Type"
    )

    vehicle_energy = (
        df.groupby("Vehicle_Type")[
            "Energy_Consumed_kWh"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        vehicle_energy
    )

    st.divider()

    # ========================================================
    # CHARGER TYPE
    # ========================================================

    st.subheader(
        "🔌 Energy Consumption by Charger Type"
    )

    charger_energy = (
        df.groupby("Charger_Type")[
            "Energy_Consumed_kWh"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        charger_energy
    )

    st.divider()

    # ========================================================
    # GRID LOAD VS ENERGY
    # ========================================================

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

    st.divider()

    # ========================================================
    # STATION STATUS
    # ========================================================

    st.subheader(
        "🔋 Energy Consumption by Station Status"
    )

    status_energy = (
        df.groupby("Station_Status")[
            "Energy_Consumed_kWh"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        status_energy
    )

    st.divider()

    # ========================================================
    # ENERGY DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Energy Consumption Distribution"
    )

    distribution_data = (
        df[
            "Energy_Consumed_kWh"
        ]
        .value_counts()
        .sort_index()
        .head(100)
    )

    st.bar_chart(
        distribution_data
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About This Project"
    )

    st.markdown("""
    ## ⚡ EV Energy Consumption Prediction

    This project is a **Machine Learning based
    web application** designed to predict energy
    consumption at Electric Vehicle charging
    stations.

    ---

    ### 🎯 Objective

    The main objective is to provide an
    easy-to-use system for estimating EV
    charging energy consumption.

    The system takes charging station information
    as input and predicts the expected energy
    consumption in **kWh**.

    ---

    ### 🛠️ Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Machine Learning
    - GitHub

    ---

    ### 🤖 Machine Learning Model

    The application uses a:

    **Random Forest Regression model**

    to predict:

    **Energy Consumed (kWh)**

    based on:

    - City
    - Charger Type
    - Vehicle Type
    - Grid Load
    - Station Status
    - Date

    ---

    ### 📈 Model Performance

    **R² Score: 0.9792**

    **MAE: 7.51 kWh**

    **RMSE: 9.85 kWh**

    ---

    ### 📊 Dataset

    The application uses an improved EV
    charging and grid-load dataset containing
    **5,000 records**.

    ---

    ### 🌐 Web Application

    This project demonstrates how a Machine
    Learning regression model can be converted
    into an interactive web application using
    Streamlit.
    """)

    st.divider()

    st.success(
        "⚡ Built as a Data Science / Machine Learning project."
    )