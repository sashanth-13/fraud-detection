# FraudGuard AI — ML Data Foundation

A full-stack machine learning data system for **Credit Card Fraud Detection** and **Customer Segmentation**.

## Tech Stack
- **Backend**: Python + FastAPI
- **Database**: SQLite (via SQLAlchemy ORM)
- **Frontend**: HTML + Tailwind CSS (multi-page SaaS dashboard)
- **ML**: XGBoost (Fraud) + K-Means (Segmentation)

## Project Structure
```
fraud_detection_system/
├── main.py                  # FastAPI backend + data generator
├── train_fraud.py           # Phase 1: XGBoost fraud detection training
├── train_segmentation.py    # Phase 2: K-Means customer segmentation
├── requirements.txt         # Python dependencies
└── static/
    └── index.html           # Multi-page dashboard (4 pages)
```

## Features

### Fraud Detection (13 ML Features)
| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `merchant_category` | Type of merchant |
| `transaction_method` | Chip / Tap / Online |
| `device_type` | Mobile / Desktop / Tablet |
| `transactions_last_24h` | Velocity feature |
| `amount_deviation_from_avg` | Historical deviation |
| `hour_of_day` | Time-based fraud signal (1-4 AM spike) |
| `is_weekend` | Weekend transaction flag |
| `is_international` | Foreign IP flag |
| `distance_from_home_km` | Geospatial anomaly |
| `time_since_last_txn_hrs` | Rapid-sequence attack detector |
| `is_round_amount` | Card-testing flag |
| `failed_attempts_24h` | Brute-force signal |

### Customer Segmentation (10 RFM+ Features)
- Recency, Frequency, Monetary (classic RFM)
- `avg_transaction_amount`, `online_purchase_ratio`
- `night_owl_ratio`, `unique_merchants_count`
- `churn_risk_score`, `preferred_payment_method`

### 4 Customer Segments
- **Champions** — High-Value, Frequent, Recent
- **Loyal Mid-Tier** — Steady, Moderate Spend
- **At-Risk** — Declining Engagement
- **Dormant** — Inactive / Low-Value

## Setup & Run

```bash
# Install dependencies
python -m pip install -r fraud_detection_system/requirements.txt

# Start the server
cd fraud_detection_system
python main.py
```

Open **http://127.0.0.1:8000** in your browser.

1. Go to **Data Init** → Click "Generate Synthetic Dataset" (5,000 users, 50,000 transactions)
2. Go to **Fraud Detection** → Simulate a transaction with all 13 features
3. Train models:
   ```bash
   python train_fraud.py       # Phase 1: XGBoost
   python train_segmentation.py # Phase 2: K-Means
   ```
