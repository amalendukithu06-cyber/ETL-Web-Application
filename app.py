import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from etl import run_etl_pipeline


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EV Charging ETL Dashboard",
    page_icon="⚡",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.title(
    "⚡ EV Charging Stations ETL Dashboard"
)

st.write(
    "Upload an EV Charging Stations CSV file "
    "and perform Extract, Transform and Load operations."
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ ETL Pipeline")

st.sidebar.markdown(
    """
    ### Pipeline

    📥 **Extract**

    🧹 **Transform**

    📊 **Analyze**

    💾 **Load**

    📥 **Download**
    """
)


# ==================================================
# FILE UPLOAD
# ==================================================

st.subheader("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ==================================================
# WHEN FILE IS UPLOADED
# ==================================================

if uploaded_file is not None:

    # ----------------------------------------------
    # Read CSV ONCE
    # ----------------------------------------------

    try:

        original_df = pd.read_csv(
            uploaded_file
        )

    except pd.errors.EmptyDataError:

        st.error(
            "❌ The uploaded CSV file is empty."
        )

        st.stop()

    except pd.errors.ParserError:

        st.error(
            "❌ Unable to read the CSV file. "
            "Please check the CSV format."
        )

        st.stop()

    except Exception as e:

        st.error(
            f"❌ Error reading CSV: {e}"
        )

        st.stop()


    # ----------------------------------------------
    # Check Dataset
    # ----------------------------------------------

    if original_df.empty:

        st.error(
            "❌ The uploaded dataset contains no rows."
        )

        st.stop()


    st.success(
        "✅ CSV file uploaded successfully!"
    )


    # ==================================================
    # ORIGINAL DATASET
    # ==================================================

    st.divider()

    st.subheader(
        "📋 Original Dataset"
    )


    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Rows",
            original_df.shape[0]
        )


    with col2:

        st.metric(
            "Total Columns",
            original_df.shape[1]
        )


    with col3:

        missing_values = int(
            original_df.isnull()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            missing_values
        )


    with col4:

        duplicate_values = int(
            original_df.duplicated()
            .sum()
        )

        st.metric(
            "Duplicate Rows",
            duplicate_values
        )


    # ----------------------------------------------
    # Preview
    # ----------------------------------------------

    st.write(
        "### Preview"
    )

    st.dataframe(
        original_df.head(10),
        use_container_width=True
    )


    # ==================================================
    # DATASET COLUMNS
    # ==================================================

    with st.expander(
        "🔎 View Dataset Columns"
    ):

        st.write(
            list(original_df.columns)
        )


    # ==================================================
    # RUN ETL
    # ==================================================

    st.divider()

    st.subheader(
        "⚙️ Run ETL Pipeline"
    )


    if st.button(
        "🚀 Run ETL Process",
        use_container_width=True
    ):

        with st.spinner(
            "Running ETL Pipeline..."
        ):

            try:

                # IMPORTANT:
                # Pass DataFrame instead of uploaded_file
                cleaned_df, summary, statistics = (
                    run_etl_pipeline(
                        original_df
                    )
                )

            except Exception as e:

                st.error(
                    f"❌ ETL processing failed: {e}"
                )

                st.stop()


        # ==================================================
        # SUCCESS
        # ==================================================

        st.success(
            "🎉 ETL Pipeline Completed Successfully!"
        )


        # ==================================================
        # ETL SUMMARY
        # ==================================================

        st.divider()

        st.subheader(
            "📊 ETL Summary"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Original Rows",
                statistics[
                    "original_rows"
                ]
            )


        with col2:

            st.metric(
                "Cleaned Rows",
                statistics[
                    "cleaned_rows"
                ]
            )


        with col3:

            st.metric(
                "Duplicates Removed",
                statistics[
                    "duplicates_removed"
                ]
            )


        with col4:

            st.metric(
                "Outliers Removed",
                statistics[
                    "outliers_removed"
                ]
            )


        # ==================================================
        # CLEANED DATA
        # ==================================================

        st.divider()

        st.subheader(
            "🧹 Cleaned Dataset"
        )


        st.write(
            f"Cleaned dataset contains "
            f"**{len(cleaned_df)} rows** "
            f"and "
            f"**{len(cleaned_df.columns)} columns**."
        )


        st.dataframe(
            cleaned_df,
            use_container_width=True
        )


        # ==================================================
        # SUMMARY STATISTICS
        # ==================================================

        st.divider()

        st.subheader(
            "📈 Summary Statistics"
        )


        st.dataframe(
            summary,
            use_container_width=True
        )


        # ==================================================
        # DATA TYPES
        # ==================================================

        st.divider()

        st.subheader(
            "🔍 Dataset Information"
        )


        info_col1, info_col2 = st.columns(2)


        # ----------------------------------------------
        # Data Types
        # ----------------------------------------------

        with info_col1:

            st.write(
                "### Data Types"
            )

            dtype_df = pd.DataFrame({

                "Column":
                    cleaned_df.columns,

                "Data Type":
                    cleaned_df.dtypes
                    .astype(str)
                    .values

            })


            st.dataframe(
                dtype_df,
                use_container_width=True
            )


        # ----------------------------------------------
        # Missing Values
        # ----------------------------------------------

        with info_col2:

            st.write(
                "### Missing Values"
            )


            missing_df = pd.DataFrame({

                "Column":
                    cleaned_df.columns,

                "Missing Values":
                    cleaned_df.isnull()
                    .sum()
                    .values

            })


            st.dataframe(
                missing_df,
                use_container_width=True
            )


        # ==================================================
        # VISUALIZATION
        # ==================================================

        st.divider()

        st.subheader(
            "📊 Data Visualization"
        )


        # ==================================================
        # CITY CHART
        # ==================================================

        if "City" in cleaned_df.columns:

            st.write(
                "### 🏙️ Charging Stations by City"
            )


            city_counts = (
                cleaned_df["City"]
                .value_counts()
                .head(10)
            )


            fig, ax = plt.subplots()


            city_counts.plot(
                kind="bar",
                ax=ax
            )


            ax.set_xlabel(
                "City"
            )

            ax.set_ylabel(
                "Number of Stations"
            )

            ax.set_title(
                "Top 10 Cities"
            )


            plt.xticks(
                rotation=45
            )

            plt.tight_layout()


            st.pyplot(fig)


        # ==================================================
        # CHARGER TYPE
        # ==================================================

        if "Charger_Type" in cleaned_df.columns:

            st.write(
                "### 🔌 Charger Type Distribution"
            )


            charger_counts = (
                cleaned_df[
                    "Charger_Type"
                ]
                .value_counts()
            )


            fig, ax = plt.subplots()


            charger_counts.plot(
                kind="bar",
                ax=ax
            )


            ax.set_xlabel(
                "Charger Type"
            )

            ax.set_ylabel(
                "Number of Stations"
            )

            ax.set_title(
                "Charger Type Distribution"
            )


            plt.xticks(
                rotation=45
            )

            plt.tight_layout()


            st.pyplot(fig)


        # ==================================================
        # VEHICLE TYPE
        # ==================================================

        if "Vehicle_Type" in cleaned_df.columns:

            st.write(
                "### 🚗 Vehicle Type Distribution"
            )


            vehicle_counts = (
                cleaned_df[
                    "Vehicle_Type"
                ]
                .value_counts()
            )


            fig, ax = plt.subplots()


            vehicle_counts.plot(
                kind="bar",
                ax=ax
            )


            ax.set_xlabel(
                "Vehicle Type"
            )

            ax.set_ylabel(
                "Count"
            )

            ax.set_title(
                "Vehicle Type Distribution"
            )


            plt.xticks(
                rotation=45
            )

            plt.tight_layout()


            st.pyplot(fig)


        # ==================================================
        # ENERGY CONSUMPTION
        # ==================================================

        if "Energy_Consumed_kWh" in cleaned_df.columns:

            st.write(
                "### ⚡ Energy Consumption Distribution"
            )


            fig, ax = plt.subplots()


            ax.hist(
                cleaned_df[
                    "Energy_Consumed_kWh"
                ],
                bins=30
            )


            ax.set_xlabel(
                "Energy Consumed (kWh)"
            )

            ax.set_ylabel(
                "Frequency"
            )

            ax.set_title(
                "Energy Consumption Distribution"
            )


            plt.tight_layout()


            st.pyplot(fig)


        # ==================================================
        # GRID LOAD
        # ==================================================

        if "Grid_Load_kW" in cleaned_df.columns:

            st.write(
                "### 🔋 Grid Load Distribution"
            )


            fig, ax = plt.subplots()


            ax.hist(
                cleaned_df[
                    "Grid_Load_kW"
                ],
                bins=30
            )


            ax.set_xlabel(
                "Grid Load (kW)"
            )

            ax.set_ylabel(
                "Frequency"
            )

            ax.set_title(
                "Grid Load Distribution"
            )


            plt.tight_layout()


            st.pyplot(fig)


        # ==================================================
        # DOWNLOAD
        # ==================================================

        st.divider()

        st.subheader(
            "💾 Download Cleaned Dataset"
        )


        csv_data = (
            cleaned_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(

            label=
                "📥 Download Cleaned CSV",

            data=
                csv_data,

            file_name=
                "EV_Charging_Stations_Cleaned.csv",

            mime=
                "text/csv",

            use_container_width=True
        )


# ==================================================
# NO FILE UPLOADED
# ==================================================

else:

    st.info(
        "👆 Please upload your "
        "EV Charging Stations CSV file "
        "to start the ETL process."
    )