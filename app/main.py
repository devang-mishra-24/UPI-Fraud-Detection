from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="UPI Fraud Detection API",
    description="Real-time UPI transaction fraud risk detection system",
    version="1.0.0"
)


# ============================================================
# LOAD SAVED ML ARTIFACTS
# ============================================================

iso_forest = joblib.load(
    "models/isolation_forest.pkl"
)

model_features = joblib.load(
    "models/model_features.pkl"
)

risk_config = joblib.load(
    "models/risk_config.pkl"
)

isolation_scaler = joblib.load(
    "models/isolation_score_scaler.pkl"
)

iqr_config = joblib.load(
    "models/iqr_config.pkl"
)


# ============================================================
# LOAD HISTORICAL TRANSACTION DATA
# ============================================================

history_df = pd.read_csv(
    "upi_transactions_processed.csv"
)

history_df["Timestamp"] = pd.to_datetime(
    history_df["Timestamp"],
    errors="coerce"
)

history_df = history_df.sort_values(
    ["Sender ID", "Timestamp"]
).reset_index(drop=True)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class TransactionRequest(BaseModel):

    transaction_id: str = Field(
        ...,
        description="Unique transaction ID"
    )

    timestamp: str = Field(
        ...,
        description="Transaction timestamp"
    )

    sender_id: str

    receiver_id: str

    amount: float = Field(
        ...,
        gt=0
    )

    merchant_category: str

    transaction_type: str

    location: str

    device_type: str

    upi_channel: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "UPI Fraud Detection API is running!",
        "status": "active"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Isolation Forest",
        "version": "1.0.0"
    }


# ============================================================
# FEATURE ENGINEERING FOR NEW TRANSACTION
# ============================================================

def create_features(transaction):

    timestamp = pd.to_datetime(
        transaction.timestamp
    )

    sender = transaction.sender_id
    receiver = transaction.receiver_id
    amount = transaction.amount

    # --------------------------------------------------------
    # Get historical transactions of this sender
    # --------------------------------------------------------

    user_history = history_df[
        history_df["Sender ID"] == sender
    ].copy()

    # Only transactions BEFORE the new transaction
    user_history = user_history[
        user_history["Timestamp"] < timestamp
    ].copy()

    # --------------------------------------------------------
    # USER AMOUNT STATISTICS
    # --------------------------------------------------------

    if len(user_history) > 0:

        historical_amounts = (
            user_history["Amount"]
            .astype(float)
            .values
        )

        n = len(historical_amounts)

        # Match notebook's full-history behavioral idea
        combined_amounts = np.append(
            historical_amounts,
            amount
        )

        user_avg_amount = combined_amounts.mean()

        user_std_amount = combined_amounts.std(
            ddof=1
        ) if len(combined_amounts) > 1 else 0.0

    else:

        user_avg_amount = amount
        user_std_amount = 0.0

    # --------------------------------------------------------
    # AMOUNT DEVIATION
    # --------------------------------------------------------

    amount_deviation = (
        (amount - user_avg_amount)
        / (user_avg_amount + 1e-6)
    )

    # --------------------------------------------------------
    # AMOUNT Z-SCORE
    # --------------------------------------------------------

    amount_zscore = (
        (amount - user_avg_amount)
        / (user_std_amount + 1e-6)
    )

    # --------------------------------------------------------
    # TIME SINCE PREVIOUS TRANSACTION
    # --------------------------------------------------------

    if len(user_history) > 0:

        previous_timestamp = (
            user_history["Timestamp"].max()
        )

        time_since_last_transaction = (
            timestamp - previous_timestamp
        ).total_seconds()

    else:

        time_since_last_transaction = -1.0

    # --------------------------------------------------------
    # RECIPIENT TRANSACTION COUNT
    # --------------------------------------------------------

    recipient_count = len(
        history_df[
            (history_df["Sender ID"] == sender)
            &
            (history_df["Receiver ID"] == receiver)
            &
            (history_df["Timestamp"] < timestamp)
        ]
    )

    # Include the new transaction itself
    recipient_transaction_count = (
        recipient_count + 1
    )

    # --------------------------------------------------------
    # PREVIOUS DEVICE
    # --------------------------------------------------------

    if len(user_history) > 0:

        previous_device = (
            user_history
            .sort_values("Timestamp")
            .iloc[-1]["Device Type"]
        )

        device_changed = int(
            transaction.device_type != previous_device
        )

    else:

        previous_device = None
        device_changed = 0

    # --------------------------------------------------------
    # PREVIOUS LOCATION
    # --------------------------------------------------------

    if len(user_history) > 0:

        previous_location = (
            user_history
            .sort_values("Timestamp")
            .iloc[-1]["Location"]
        )

        location_changed = int(
            transaction.location != previous_location
        )

    else:

        previous_location = None
        location_changed = 0

    # --------------------------------------------------------
    # HOUR DEVIATION
    # --------------------------------------------------------

    current_hour = timestamp.hour

    if len(user_history) > 0:

        historical_hours = (
            user_history["Timestamp"]
            .dt.hour
            .astype(float)
            .values
        )

        combined_hours = np.append(
            historical_hours,
            current_hour
        )

        user_mean_hour = combined_hours.mean()

        hour_deviation = abs(
            current_hour - user_mean_hour
        )

    else:

        hour_deviation = 0.0

    # --------------------------------------------------------
    # TRANSACTIONS IN PREVIOUS 1 HOUR
    # --------------------------------------------------------

    one_hour_ago = timestamp - pd.Timedelta(
        hours=1
    )

    transactions_last_1h = len(
        history_df[
            (history_df["Sender ID"] == sender)
            &
            (history_df["Timestamp"] >= one_hour_ago)
            &
            (history_df["Timestamp"] < timestamp)
        ]
    )

    # --------------------------------------------------------
    # RETURN MODEL FEATURES
    # --------------------------------------------------------

    features = {

        "Amount": amount,

        "amount_deviation":
            amount_deviation,

        "amount_zscore":
            amount_zscore,

        "transactions_last_1h":
            transactions_last_1h,

        "time_since_last_transaction":
            time_since_last_transaction,

        "recipient_transaction_count":
            recipient_transaction_count,

        "device_changed":
            device_changed,

        "location_changed":
            location_changed,

        "hour_deviation":
            hour_deviation
    }

    return features


# ============================================================
# IQR SCORE
# ============================================================

def calculate_iqr_score(features):

    score = 0

    for column in [
        "Amount",
        "amount_deviation",
        "transactions_last_1h",
        "amount_zscore"
    ]:

        value = features[column]

        bounds = iqr_config[column]

        lower = bounds["lower"]
        upper = bounds["upper"]

        if value < lower or value > upper:

            score += 1

    return score


# ============================================================
# FRAUD REASON
# ============================================================

def generate_fraud_reasons(
    features,
    iqr_score,
    isolation_flag,
    time_series_flags
):

    reasons = []

    if iqr_score >= 2:

        reasons.append(
            "Unusual transaction amount"
        )

    if isolation_flag == 1:

        reasons.append(
            "Multivariate anomaly"
        )

    if time_series_flags["rapid_transaction_flag"] == 1:

        reasons.append(
            "Rapid transaction"
        )

    if time_series_flags["burst_flag"] == 1:

        reasons.append(
            "Transaction burst"
        )

    if time_series_flags["unusual_hour_flag"] == 1:

        reasons.append(
            "Unusual transaction time"
        )

    if features["device_changed"] == 1:

        reasons.append(
            "Device change"
        )

    if features["location_changed"] == 1:

        reasons.append(
            "Location change"
        )

    if len(reasons) == 0:

        reasons.append(
            "Normal behavior"
        )

    return reasons


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(transaction: TransactionRequest):

    # --------------------------------------------------------
    # 1. CREATE FEATURES
    # --------------------------------------------------------

    features = create_features(
        transaction
    )

    # --------------------------------------------------------
    # 2. CREATE MODEL INPUT
    # --------------------------------------------------------

    X = pd.DataFrame(
        [features]
    )

    X = X[model_features]

    # --------------------------------------------------------
    # 3. ISOLATION FOREST
    # --------------------------------------------------------

    isolation_prediction = (
        iso_forest.predict(X)[0]
    )

    isolation_flag = int(
        isolation_prediction == -1
    )

    raw_isolation_score = float(
        -iso_forest.score_samples(X)[0]
    )

    # Normalize using training scaler
    isolation_score_normalized = float(
        isolation_scaler.transform(
            [[raw_isolation_score]]
        )[0][0]
    )

    # Keep within 0-1
    isolation_score_normalized = float(
        np.clip(
            isolation_score_normalized,
            0,
            1
        )
    )

    # --------------------------------------------------------
    # 4. IQR
    # --------------------------------------------------------

    iqr_score = calculate_iqr_score(
        features
    )

    iqr_score_normalized = (
        iqr_score / 4
    )

    # --------------------------------------------------------
    # 5. TIME-SERIES SIGNALS
    # --------------------------------------------------------

    time_since_last = (
        features[
            "time_since_last_transaction"
        ]
    )

    rapid_transaction_flag = int(
        time_since_last >= 0
        and time_since_last <= 5 * 60
    )

    burst_flag = int(
        features[
            "transactions_last_1h"
        ] >= 2
    )

    unusual_hour_flag = int(
        features[
            "hour_deviation"
        ] >= 6
    )

    time_series_score = (
        rapid_transaction_flag
        + burst_flag
        + unusual_hour_flag
    )

    time_series_score_normalized = (
        time_series_score / 3
    )

    # --------------------------------------------------------
    # 6. FINAL FRAUD RISK SCORE
    # --------------------------------------------------------

    fraud_risk_score = (

        0.40 *
        isolation_score_normalized

        +

        0.35 *
        iqr_score_normalized

        +

        0.25 *
        time_series_score_normalized
    )

    # --------------------------------------------------------
    # 7. RISK LEVEL
    # --------------------------------------------------------

    if fraud_risk_score >= 0.75:

        risk_level = "CRITICAL"

    elif fraud_risk_score >= 0.50:

        risk_level = "HIGH"

    elif fraud_risk_score >= 0.25:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # --------------------------------------------------------
    # 8. FINAL FRAUD FLAG
    # --------------------------------------------------------

    fraud_flag = int(
        risk_level in [
            "HIGH",
            "CRITICAL"
        ]
    )

    # --------------------------------------------------------
    # 9. REASONS
    # --------------------------------------------------------

    time_series_flags = {

        "rapid_transaction_flag":
            rapid_transaction_flag,

        "burst_flag":
            burst_flag,

        "unusual_hour_flag":
            unusual_hour_flag
    }

    fraud_reasons = generate_fraud_reasons(
        features,
        iqr_score,
        isolation_flag,
        time_series_flags
    )

    # --------------------------------------------------------
    # 10. RESPONSE
    # --------------------------------------------------------

    return {

        "transaction_id":
            transaction.transaction_id,

        "fraud_risk_score":
            round(
                fraud_risk_score,
                4
            ),

        "risk_level":
            risk_level,

        "fraud_flag":
            fraud_flag,

        "fraud_reason":
            fraud_reasons,

        "model_details": {

            "iqr_score":
                iqr_score,

            "isolation_score":
                round(
                    raw_isolation_score,
                    4
                ),

            "isolation_flag":
                isolation_flag,

            "time_series_score":
                time_series_score,

            "transactions_last_1h":
                features[
                    "transactions_last_1h"
                ],

            "device_changed":
                features[
                    "device_changed"
                ],

            "location_changed":
                features[
                    "location_changed"
                ],

            "hour_deviation":
                round(
                    features[
                        "hour_deviation"
                    ],
                    2
                )
        }
    }