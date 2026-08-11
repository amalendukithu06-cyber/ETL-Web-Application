import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# LOAD DATASET
# ==================================================

print("Loading dataset...")

df = pd.read_csv("EV_Charging_Grid_Load_Dataset_Improved.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==================================================
# DATE PROCESSING
# ==================================================

df["Date"] = pd.to_datetime(df["Date"])

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["day_of_year"] = df["Date"].dt.dayofyear


# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df[
    [
        "City",
        "Charger_Type",
        "Vehicle_Type",
        "Grid_Load_kW",
        "Station_Status",
        "year",
        "month",
        "day",
        "day_of_year"
    ]
]

y = df["Energy_Consumed_kWh"]


# ==================================================
# CATEGORICAL & NUMERICAL FEATURES
# ==================================================

categorical_features = [
    "City",
    "Charger_Type",
    "Vehicle_Type",
    "Station_Status"
]

numerical_features = [
    "Grid_Load_kW",
    "year",
    "month",
    "day",
    "day_of_year"
]


# ==================================================
# PREPROCESSING
# ==================================================

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
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==================================================
# RANDOM FOREST MODEL
# ==================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ==================================================
# PIPELINE
# ==================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==================================================
# TRAIN MODEL
# ==================================================

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)


# ==================================================
# PREDICTION
# ==================================================

y_pred = pipeline.predict(X_test)


# ==================================================
# MODEL PERFORMANCE
# ==================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n====================================")
print("RANDOM FOREST MODEL PERFORMANCE")
print("====================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("====================================")


# ==================================================
# SAVE MODEL
# ==================================================

with open(
    "model.pkl",
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )


print("\n====================================")
print("MODEL SAVED SUCCESSFULLY")
print("====================================")
print("Model: Random Forest")
print("File : model.pkl")
print("====================================")