"""Generate a reproducible, synthetic telecom churn dataset for RetainIQ demos."""
from pathlib import Path
import numpy as np
import pandas as pd



def generate_customers(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    customer_id = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    gender = rng.choice(["Female", "Male"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.integers(1, 73, n)
    contract = np.array([rng.choice(["Month-to-month", "One year", "Two year"], p=[0.58, 0.25, 0.17] if t < 12 else [0.35, 0.35, 0.30]) for t in tenure])
    phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.48, 0.38, 0.14])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n, p=[0.38, 0.25, 0.22, 0.15])
    monthly = np.clip(rng.normal(72, 24, n) + (internet == "Fiber optic") * 12 - (internet == "No") * 28, 18, 145).round(2)
    total = (monthly * tenure * rng.uniform(0.88, 1.12, n)).round(2)
    tech = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n, p=[0.55, 0.45]))
    security = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n, p=[0.48, 0.52]))
    streaming = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n, p=[0.50, 0.50]))
    tickets = np.clip(rng.poisson(1.1 + (tenure < 12) * 0.5 + (monthly > 95) * 0.25, n), 0, 9)
    usage_drop = np.clip(rng.normal(12, 18, n) + tickets * 5 + (tenure < 6) * 10, 0, 100).round(1)
    satisfaction = np.clip(rng.normal(7.0, 1.4, n) - tickets * 0.35 - usage_drop * 0.018, 1, 10).round(1)

    # Synthetic churn tendency: patterns are intentionally interpretable for teaching.
    logit = (-2.1 + (contract == "Month-to-month") * 1.20 + (contract == "One year") * 0.20
              - (contract == "Two year") * 0.75 + (tenure < 6) * 0.90 + (tenure < 12) * 0.40
              + (monthly > 95) * 0.45 + (payment == "Electronic check") * 0.35
              + tickets * 0.18 + usage_drop * 0.018 - (satisfaction - 5) * 0.22
              + (tech == "No") * 0.25 + senior * 0.12)
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability, n)

    df = pd.DataFrame({
        "customer_id": customer_id, "gender": gender, "senior_citizen": senior,
        "partner": partner, "dependents": dependents, "tenure_months": tenure,
        "phone_service": phone, "internet_service": internet, "contract": contract,
        "payment_method": payment, "monthly_charges": monthly, "total_charges": total,
        "tech_support": tech, "online_security": security, "streaming_tv": streaming,
        "support_tickets_3m": tickets, "usage_drop_pct": usage_drop,
        "satisfaction_score": satisfaction, "churn": churn
    })
    return df



if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "telco_customers_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_customers().to_csv(out, index=False)
    print(f"Wrote {out}")
