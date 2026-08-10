import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("EV_Charging_Grid_Load_Dataset_5000.csv")


# ==========================================
# 2. DATE PROCESSING
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

features = [
    "City",
    "Charger_Type",
    "Vehicle_Type",
    "Grid_Load_kW",
    "Station_Status",
    "Year",
    "Month",
    "Day"
]

target = "Energy_Consumed_kWh"


X = df[features]
y = df[target]


# ==========================================
# 4. COLUMNS
# ==========================================

categorical_features = [
    "City",
    "Charger_Type",
    "Vehicle_Type",
    "Station_Status"
]

numeric_features = [
    "Grid_Load_kW",
    "Year",
    "Month",
    "Day"
]


# ==========================================
# 5. PREPROCESSING
# ==========================================

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


# ==========================================
# 6. MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 7. PIPELINE
# ==========================================

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


# ==========================================
# 8. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("Training model...")

pipeline.fit(
    X_train,
    y_train
)


# ==========================================
# 10. EVALUATION
# ==========================================

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("--------------------------------")
print("MODEL PERFORMANCE")
print("--------------------------------")

print(
    f"Mean Absolute Error: {mae:.2f}"
)

print(
    f"R2 Score: {r2:.2f}"
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

with open(
    "model.pkl",
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )


print("--------------------------------")
print("Model saved successfully!")
print("File: model.pkl")
print("--------------------------------")