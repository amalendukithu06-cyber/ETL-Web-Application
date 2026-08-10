import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("EV_Charging_Grid_Load_Dataset_5000.csv")


# Date processing
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfYear"] = df["Date"].dt.dayofyear


# Features
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


# Categorical columns
categorical_features = [
    "City",
    "Charger_Type",
    "Vehicle_Type",
    "Station_Status"
]

# Numerical columns
numeric_features = [
    "Grid_Load_kW",
    "Year",
    "Month",
    "Day",
    "DayOfYear"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# Model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Training
print("Training improved model...")

pipeline.fit(
    X_train,
    y_train
)


# Prediction
y_pred = pipeline.predict(X_test)


# Evaluation
mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("--------------------------------")
print("IMPROVED MODEL PERFORMANCE")
print("--------------------------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")
print("--------------------------------")


# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(pipeline, file)


print("Model saved successfully!")
print("File: model.pkl")