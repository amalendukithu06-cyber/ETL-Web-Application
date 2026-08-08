import pandas as pd


def run_etl_pipeline(df):
    """
    ETL Pipeline

    Extract:
        Receive the uploaded DataFrame

    Transform:
        - Remove duplicate rows
        - Handle missing values
        - Clean City names
        - Remove Energy Consumed outliers

    Load:
        Return the cleaned DataFrame
    """

    # ==========================================
    # EXTRACT
    # ==========================================

    # Make a copy so the original data is not changed
    df = df.copy()

    original_rows = len(df)
    original_columns = len(df.columns)

    # ==========================================
    # TRANSFORM
    # ==========================================

    # ------------------------------------------
    # 1. Remove Duplicate Rows
    # ------------------------------------------

    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates()

    # ------------------------------------------
    # 2. Handle Missing Values
    # ------------------------------------------

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    # Fill numerical missing values with median
    for column in numeric_columns:

        if df[column].isnull().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

    # Fill categorical missing values with Unknown
    for column in categorical_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                "Unknown"
            )

    # ------------------------------------------
    # 3. Clean City Names
    # ------------------------------------------

    if "City" in df.columns:

        df["City"] = (
            df["City"]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # ------------------------------------------
    # 4. Remove Energy Consumption Outliers
    # ------------------------------------------

    outliers_removed = 0

    if "Energy_Consumed_kWh" in df.columns:

        # Make sure column is numeric
        df["Energy_Consumed_kWh"] = pd.to_numeric(
            df["Energy_Consumed_kWh"],
            errors="coerce"
        )

        # Fill any newly created missing values
        df["Energy_Consumed_kWh"] = (
            df["Energy_Consumed_kWh"]
            .fillna(
                df["Energy_Consumed_kWh"].median()
            )
        )

        Q1 = df["Energy_Consumed_kWh"].quantile(
            0.25
        )

        Q3 = df["Energy_Consumed_kWh"].quantile(
            0.75
        )

        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)

        upper_bound = Q3 + (1.5 * IQR)

        rows_before_outlier = len(df)

        df = df[
            (df["Energy_Consumed_kWh"] >= lower_bound)
            &
            (df["Energy_Consumed_kWh"] <= upper_bound)
        ]

        outliers_removed = (
            rows_before_outlier - len(df)
        )

    # ==========================================
    # LOAD
    # ==========================================

    cleaned_rows = len(df)

    # ------------------------------------------
    # Summary Statistics
    # ------------------------------------------

    summary = df.describe(
        include="all"
    ).transpose()

    # ------------------------------------------
    # ETL Statistics
    # ------------------------------------------

    statistics = {

        "original_rows":
            original_rows,

        "cleaned_rows":
            cleaned_rows,

        "columns":
            original_columns,

        "duplicates_removed":
            int(duplicate_count),

        "outliers_removed":
            int(outliers_removed),

        "missing_values":
            int(
                df.isnull().sum().sum()
            )
    }

    return (
        df,
        summary,
        statistics
    )