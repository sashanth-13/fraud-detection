import random
import math
import os
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from faker import Faker
import uvicorn

# ─── Configuration ────────────────────────────────────────
DATABASE_URL = "sqlite:///./data/fraud_detection.db"
os.makedirs("data", exist_ok=True)

engine     = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()
fake         = Faker()

# ─── ORM Models ───────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    user_id                 = Column(Integer, primary_key=True, index=True)
    # Original RFM features
    avg_monthly_spend       = Column(Float)
    recency_days            = Column(Integer)
    frequency_per_month     = Column(Integer)
    total_monetary_value    = Column(Float)
    top_category_preference = Column(String)
    # NEW segmentation features
    avg_transaction_amount  = Column(Float)      # avg spend per single transaction
    online_purchase_ratio   = Column(Float)      # 0-1: proportion of online txns
    night_owl_ratio         = Column(Float)      # 0-1: proportion of txns 10pm-6am
    unique_merchants_count  = Column(Integer)    # number of distinct merchant types used
    preferred_payment_method= Column(String)     # Chip / Tap / Online
    churn_risk_score        = Column(Float)      # 0-1 composite churn risk
    # Home location (used for distance feature in transactions)
    home_lat                = Column(Float)
    home_lon                = Column(Float)

    transactions = relationship("Transaction", back_populates="user")


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id           = Column(Integer, primary_key=True, index=True)
    user_id                  = Column(Integer, ForeignKey("users.user_id"))
    amount                   = Column(Float)
    merchant_category        = Column(String)
    timestamp                = Column(DateTime, default=datetime.utcnow)
    transaction_method       = Column(String)   # Chip, Tap, Online
    device_type              = Column(String)   # Mobile, Desktop, Tablet
    ip_address               = Column(String)
    location_lat             = Column(Float)
    location_lon             = Column(Float)
    # Original ML features
    transactions_last_24h        = Column(Integer)
    amount_deviation_from_avg    = Column(Float)
    is_fraud                     = Column(Boolean, default=False)
    # NEW fraud-detection features
    hour_of_day                  = Column(Integer)   # 0-23 (fraud spikes 1-4 AM)
    is_weekend                   = Column(Boolean)   # Sat/Sun flag
    is_international              = Column(Boolean)   # IP country differs from home
    distance_from_home_km        = Column(Float)     # km from user's home coords
    time_since_last_txn_hrs      = Column(Float)     # hours since previous transaction
    is_round_amount              = Column(Boolean)   # amount is suspiciously round
    failed_attempts_24h          = Column(Integer)   # failed auth attempts in 24h

    user = relationship("User", back_populates="transactions")


def recreate_tables():
    """Drop and recreate all tables to pick up schema changes."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

Base.metadata.create_all(bind=engine)   # create if not exist on startup

# ─── Pydantic Models ──────────────────────────────────────
class TransactionInput(BaseModel):
    user_id: int
    amount: float
    merchant_category: str
    transaction_method: str
    device_type: str

class GenerateResponse(BaseModel):
    message: str
    users_count: int
    transactions_count: int

# ─── FastAPI App ──────────────────────────────────────────
app = FastAPI(title="Fraud Detection Data System")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# ─── Helper Functions ─────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def is_round(amount: float) -> bool:
    """True if amount is a suspicious round number (e.g. 500.00, 1000.00)."""
    return amount % 50 == 0 and amount >= 100

# ─── Data Generation ──────────────────────────────────────
def generate_bulk_data(db: Session, num_users=5000, num_transactions=50000):
    CATEGORIES   = ['Groceries', 'Electronics', 'Travel', 'Dining', 'Entertainment', 'Fashion', 'Utility']
    METHODS      = ['Chip', 'Tap', 'Online']
    DEVICES      = ['Mobile', 'Desktop', 'Tablet']

    # ── 1. Generate Users ────────────────────────────────
    print(f"Generating {num_users} users...")
    users = []
    for _ in range(num_users):
        freq   = random.randint(5, 50)
        spend  = round(random.uniform(500, 5000), 2)
        total  = round(spend * random.uniform(12, 36), 2)   # 1-3 year lifetime
        rec    = random.randint(0, 30)
        avg_tx = round(spend / freq, 2)

        online_ratio  = round(random.uniform(0.05, 0.75), 2)
        night_ratio   = round(random.uniform(0.02, 0.35), 2)
        unique_merch  = random.randint(2, len(CATEGORIES))
        pref_method   = random.choice(METHODS)

        # Churn risk: high recency + low frequency + low spend → higher churn risk
        churn = round(
            (rec / 30) * 0.4 +
            (1 - min(freq, 30) / 30) * 0.35 +
            (1 - min(spend, 5000) / 5000) * 0.25,
            4
        )

        user = User(
            avg_monthly_spend       = spend,
            recency_days            = rec,
            frequency_per_month     = freq,
            total_monetary_value    = total,
            top_category_preference = random.choice(CATEGORIES),
            avg_transaction_amount  = avg_tx,
            online_purchase_ratio   = online_ratio,
            night_owl_ratio         = night_ratio,
            unique_merchants_count  = unique_merch,
            preferred_payment_method= pref_method,
            churn_risk_score        = churn,
            home_lat                = float(fake.latitude()),
            home_lon                = float(fake.longitude()),
        )
        users.append(user)

    db.add_all(users)
    db.commit()

    user_list    = db.query(User).all()
    user_avg_map = {u.user_id: u.avg_transaction_amount for u in user_list}

    # ── 2. Generate Transactions ─────────────────────────
    print(f"Generating {num_transactions} transactions...")
    transactions = []
    # Track last transaction time per user for time_since_last_txn
    last_txn_time: dict = {}

    for _ in range(num_transactions):
        user     = random.choice(user_list)
        is_fraud = random.random() < 0.01   # 1% base fraud rate

        # Fraud rewires several features to realistic fraud patterns
        if is_fraud:
            amount_mult        = random.uniform(4.0, 12.0)
            tx_last_24h        = random.randint(8, 50)   # velocity spike
            failed_attempts    = random.randint(1, 8)
            is_intl            = random.random() < 0.70  # usually international
            time_since_hrs     = round(random.uniform(0.01, 1.5), 2)  # rapid follow-up
            # Fraud transactions cluster in late-night hours (1–4 AM)
            hour               = random.choices(range(24),
                                    weights=[1,3,4,4,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2])[0]
            dist_km            = random.uniform(500, 15000)   # far from home
        else:
            amount_mult        = random.uniform(0.4, 1.8)
            tx_last_24h        = random.randint(0, 4)
            failed_attempts    = random.choices([0, 1, 2], weights=[85, 12, 3])[0]
            is_intl            = random.random() < 0.05
            time_since_hrs     = round(random.uniform(2, 72), 2)
            hour               = random.choices(range(24),
                                    weights=[1,1,1,1,1,2,3,5,6,6,5,5,6,6,5,5,5,5,5,4,4,3,2,1])[0]
            dist_km            = random.uniform(0, 50)

        base_avg = user_avg_map.get(user.user_id, 50.0) or 50.0
        amount   = round(base_avg * amount_mult, 2)
        deviation= round(amount / base_avg, 4) if base_avg > 0 else 1.0

        # Timestamp: work backward from now
        ts      = fake.date_time_between(start_date="-1y", end_date="now")
        ts      = ts.replace(hour=hour)
        weekend = ts.weekday() >= 5   # Sat=5, Sun=6

        # Location may be far from home if international/fraudulent
        if is_intl or dist_km > 200:
            tx_lat = float(fake.latitude())
            tx_lon = float(fake.longitude())
        else:
            # Jitter ≈ 50 km from home
            tx_lat = user.home_lat + random.uniform(-0.45, 0.45)
            tx_lon = user.home_lon + random.uniform(-0.45, 0.45)

        real_dist = haversine_km(user.home_lat, user.home_lon, tx_lat, tx_lon)

        tx = Transaction(
            user_id                  = user.user_id,
            amount                   = amount,
            merchant_category        = random.choice(CATEGORIES),
            timestamp                = ts,
            transaction_method       = random.choice(['Online'] if is_fraud and is_intl else METHODS),
            device_type              = random.choice(DEVICES),
            ip_address               = fake.ipv4_public() if is_intl else fake.ipv4_private(),
            location_lat             = round(tx_lat, 6),
            location_lon             = round(tx_lon, 6),
            # Original features
            transactions_last_24h    = tx_last_24h,
            amount_deviation_from_avg= deviation,
            is_fraud                 = is_fraud,
            # New features
            hour_of_day              = hour,
            is_weekend               = weekend,
            is_international         = is_intl,
            distance_from_home_km    = round(real_dist, 2),
            time_since_last_txn_hrs  = time_since_hrs,
            is_round_amount          = is_round(amount),
            failed_attempts_24h      = failed_attempts,
        )
        transactions.append(tx)

        if len(transactions) >= 1000:
            db.add_all(transactions)
            db.commit()
            transactions = []

    if transactions:
        db.add_all(transactions)
        db.commit()


# ─── API Endpoints ────────────────────────────────────────
@app.get("/stats")
async def get_stats():
    db = SessionLocal()
    try:
        return {
            "users": db.query(User).count(),
            "transactions": db.query(Transaction).count(),
            "fraud_count": db.query(Transaction).filter(Transaction.is_fraud == True).count(),
        }
    finally:
        db.close()

@app.post("/generate-data", response_model=GenerateResponse)
async def generate_data_endpoint():
    db = SessionLocal()
    try:
        # Full schema recreation to pick up new columns
        recreate_tables()

        generate_bulk_data(db)

        u_count = db.query(User).count()
        t_count = db.query(Transaction).count()

        return GenerateResponse(
            message="Synthetic data generation completed successfully with enhanced ML features.",
            users_count=u_count,
            transactions_count=t_count,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/simulate-transaction")
async def simulate_transaction(input_data: TransactionInput):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == input_data.user_id).first()
        if not user:
            user = User(
                user_id=input_data.user_id,
                avg_monthly_spend=1000.0, recency_days=5, frequency_per_month=10,
                total_monetary_value=12000.0, top_category_preference="Unknown",
                avg_transaction_amount=100.0, online_purchase_ratio=0.3,
                night_owl_ratio=0.1, unique_merchants_count=4,
                preferred_payment_method="Chip", churn_risk_score=0.3,
                home_lat=float(fake.latitude()), home_lon=float(fake.longitude()),
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        avg_tx   = user.avg_transaction_amount or 100.0
        deviation = round(input_data.amount / avg_tx, 4)
        now      = datetime.utcnow()
        hour     = now.hour

        # Simulate features
        velocity        = random.randint(0, 4)
        failed_attempts = 0
        is_intl         = input_data.transaction_method == "Online" and random.random() < 0.1
        dist_km         = random.uniform(0, 30) if not is_intl else random.uniform(1000, 8000)
        time_since      = round(random.uniform(2, 24), 2)

        # Multi-rule fraud heuristic (richer than before)
        fraud_score = 0.0
        if deviation > 5.0:          fraud_score += 0.5
        if deviation > 10.0:         fraud_score += 0.3
        if velocity > 8:             fraud_score += 0.3
        if hour in range(1, 5):      fraud_score += 0.2   # 1–4 AM
        if dist_km > 500:            fraud_score += 0.2
        if is_intl:                  fraud_score += 0.2
        if is_round(input_data.amount): fraud_score += 0.1

        is_fraud = fraud_score >= 0.5

        tx_lat = user.home_lat + random.uniform(-0.45, 0.45)
        tx_lon = user.home_lon + random.uniform(-0.45, 0.45)

        new_tx = Transaction(
            user_id=user.user_id, amount=input_data.amount,
            merchant_category=input_data.merchant_category,
            timestamp=now, transaction_method=input_data.transaction_method,
            device_type=input_data.device_type,
            ip_address=fake.ipv4_public() if is_intl else fake.ipv4_private(),
            location_lat=round(tx_lat, 6), location_lon=round(tx_lon, 6),
            transactions_last_24h=velocity, amount_deviation_from_avg=deviation,
            is_fraud=is_fraud, hour_of_day=hour,
            is_weekend=now.weekday() >= 5, is_international=is_intl,
            distance_from_home_km=round(dist_km, 2),
            time_since_last_txn_hrs=time_since,
            is_round_amount=is_round(input_data.amount),
            failed_attempts_24h=failed_attempts,
        )
        db.add(new_tx)
        db.commit()
        db.refresh(new_tx)

        return {
            "transaction_id": new_tx.transaction_id,
            "is_fraud": new_tx.is_fraud,
            "fraud_score": round(fraud_score, 2),
            "amount_deviation_from_avg": deviation,
            "distance_from_home_km": round(dist_km, 2),
            "hour_of_day": hour,
            "is_international": is_intl,
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
