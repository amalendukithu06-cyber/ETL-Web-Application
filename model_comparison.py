import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "EV_Charging_Grid_Load_Dataset_Improved.csv"
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# DATE PROCESSING
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfYear"] = df["Date"].dt.dayofyear


# ============================================================
# FEATURES
# ============================================================

features = [
    "City",
    "Charger_Type",
    "Vehicle_Type",
    "Grid_Load_kW",
    "Station_Status",
    "Year",
    "Month",
    "Day",
    "DayOfYear"
]

target = "Energy_Consumed_kWh"


X = df[features]
y = df[target]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "City",
    "Charger_Type",
    "Vehicle_Type",
    "Station_Status"
]


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numeric_features = [
    "Grid_Load_kW",
    "Year",
    "Month",
    "Day",
    "DayOfYear"
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")


results = []


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    # Train
    pipeline.fit(
        X_train,
        y_train
    )


    # Predict
    predictions = pipeline.predict(
        X_test
    )


    # MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )


    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    # R2
    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# Sort according to R2
results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")


print(

    results_df.to_string(

        index=False,

        formatters={

            "MAE":
                "{:.2f}".format,

            "RMSE":
                "{:.2f}".format,

            "R2":
                "{:.4f}".format
        }
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]


print("\n====================================")
print("BEST MODEL")
print("====================================")


print(
    "Model:",
    best_model["Model"]
)

print(
    "R²:",
    round(
        best_model["R2"],
        4
    )
)

print(
    "MAE:",
    round(
        best_model["MAE"],
        2
    )
)

print(
    "RMSE:",
    round(
        best_model["RMSE"],
        2
    )
)


print("\n====================================")