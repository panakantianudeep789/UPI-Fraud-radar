"""
Generates a synthetic UPI-style transaction dataset with realistic
legitimate behavior + injected fraud patterns:
  - Account takeover (new device + odd hour + high amount)
  - Money mule bursts (many rapid small txns to new payees)
  - Geographic impossibility (location jump in short time)
  - Round-amount / testing fraud (small probing txns before a big one)
"""
import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

N_USERS = 800
N_TXNS = 60000
FRAUD_RATE = 0.012  # ~1.2% fraud, realistic for payment fraud

cities = ["Hyderabad", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Pune",
          "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
devices_pool = [fake.uuid4() for _ in range(N_USERS * 2)]

# Each user has a home city, 1-2 usual devices, and a typical spend range
users = []
for i in range(N_USERS):
    home_city = random.choice(cities)
    avg_amount = np.random.gamma(2, 400) + 50
    usual_devices = random.sample(devices_pool, k=random.choice([1, 1, 1, 2]))
    users.append({
        "user_id": f"user_{i:04d}",
        "vpa": f"{fake.user_name()}@{random.choice(['okhdfcbank','ybl','oksbi','paytm','ibl'])}",
        "home_city": home_city,
        "avg_amount": avg_amount,
        "usual_devices": usual_devices,
    })
users_df = pd.DataFrame(users)

start_time = datetime(2026, 1, 1)
rows = []
last_txn_time = {u["user_id"]: start_time for u in users}
last_txn_city = {u["user_id"]: u["home_city"] for u in users}

n_fraud_target = int(N_TXNS * FRAUD_RATE)
fraud_indices = set(random.sample(range(N_TXNS), n_fraud_target))

for i in range(N_TXNS):
    user = users[random.randint(0, N_USERS - 1)]
    uid = user["user_id"]
    is_fraud = i in fraud_indices

    # timestamp: mostly business hours, some spread
    prev_time = last_txn_time[uid]
    gap_minutes = np.random.exponential(scale=600)  # avg 10hr between txns normally
    txn_time = prev_time + timedelta(minutes=gap_minutes)
    hour = txn_time.hour

    receiver_vpa = f"{fake.user_name()}@{random.choice(['okhdfcbank','ybl','oksbi','paytm','ibl'])}"

    if not is_fraud:
        amount = max(10, np.random.normal(user["avg_amount"], user["avg_amount"] * 0.3))
        device = random.choice(user["usual_devices"])
        city = user["home_city"] if random.random() > 0.05 else random.choice(cities)
        hour = int(np.clip(np.random.normal(14, 5), 0, 23))
        fraud_type = "none"
    else:
        pattern = random.choice(["takeover", "mule_burst", "geo_jump", "probing"])
        if pattern == "takeover":
            amount = user["avg_amount"] * random.uniform(4, 12)
            device = fake.uuid4()  # brand new device
            city = random.choice([c for c in cities if c != user["home_city"]])
            hour = random.choice([1, 2, 3, 4, 23])
        elif pattern == "mule_burst":
            amount = np.random.uniform(500, 2000)
            device = random.choice(user["usual_devices"] + [fake.uuid4()])
            city = user["home_city"]
            hour = int(np.clip(np.random.normal(14, 5), 0, 23))
            gap_minutes = np.random.uniform(0.2, 3)  # very rapid
            txn_time = prev_time + timedelta(minutes=gap_minutes)
        elif pattern == "geo_jump":
            amount = user["avg_amount"] * random.uniform(1, 3)
            device = fake.uuid4()
            city = random.choice([c for c in cities if c != last_txn_city[uid]])
            gap_minutes = np.random.uniform(5, 40)  # too fast to travel
            txn_time = prev_time + timedelta(minutes=gap_minutes)
            hour = txn_time.hour
        else:  # probing
            amount = np.random.uniform(1, 10)
            device = fake.uuid4()
            city = user["home_city"]
            hour = int(np.clip(np.random.normal(14, 5), 0, 23))
        fraud_type = pattern

    new_device = device not in user["usual_devices"]
    new_city = city != last_txn_city[uid]
    time_since_last_min = max((txn_time - prev_time).total_seconds() / 60, 0.01)

    rows.append({
        "txn_id": f"T{i:07d}",
        "timestamp": txn_time,
        "user_id": uid,
        "sender_vpa": user["vpa"],
        "receiver_vpa": receiver_vpa,
        "amount": round(amount, 2),
        "hour": hour,
        "device_id": device,
        "city": city,
        "avg_user_amount": round(user["avg_amount"], 2),
        "is_new_device": int(new_device),
        "is_new_city": int(new_city),
        "minutes_since_last_txn": round(time_since_last_min, 2),
        "fraud_type": fraud_type,
        "is_fraud": int(is_fraud),
    })

    last_txn_time[uid] = txn_time
    last_txn_city[uid] = city

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.to_csv("upi_transactions.csv", index=False)
print(f"Generated {len(df)} transactions, {df['is_fraud'].sum()} fraudulent ({df['is_fraud'].mean()*100:.2f}%)")
print(df["fraud_type"].value_counts())
