"""
STEP 5 - Two regression models that answer the questions you raised.

QUESTION 1: "Current balance is not enough. Based on how they TRANSACT,
             can they build a down payment?"
  -> Model A predicts SAVINGS VELOCITY: how much this customer actually
     accumulates per month, learned from their spending behaviour.
     Then we work out how many months until they can fund a down payment.

QUESTION 2: "What if they keep their money at another bank and only
             transact with us?"
  -> Model B estimates MONEY HELD ELSEWHERE from the outflow trail:
     self-transfers, SIP debits to other AMCs, credit card payments to
     other banks, insurance premiums to other insurers.
     We cannot see their other balance. We can see the money walking there.

Both are REGRESSION problems (predict a number), unlike step 3 which was
CLASSIFICATION (predict yes/no). Good to be able to explain that difference.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

RS = 42
df = pd.read_csv("customer_data.csv")


def run_regression(X, y, name, unit=""):
    """Train 4 regressors, compare, return the best fitted pipeline."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RS),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.08, random_state=RS),
    }

    rows, fitted = [], {}
    for cname, model in candidates.items():
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="r2", n_jobs=-1)
        rows.append({
            "Model": cname,
            "R2 (test)": round(r2_score(yte, pred), 3),
            "R2 (CV)": round(cv.mean(), 3),
            f"MAE{unit}": round(mean_absolute_error(yte, pred), 0),
        })
        fitted[cname] = pipe

    table = pd.DataFrame(rows).sort_values("R2 (test)", ascending=False)
    print("=" * 74)
    print(name)
    print("=" * 74)
    print(table.to_string(index=False))

    best = table.iloc[0]["Model"]
    print(f"\nBest: {best}")
    print("R2 = share of the variation the model explains. 1.0 is perfect, 0 is useless.")
    print("MAE = average error in rupees. Easier to explain to a business person.\n")
    return fitted[best], best, table, (Xte, yte, fitted[best].predict(Xte))


# ======================================================================
# MODEL A - SAVINGS VELOCITY (can they build a down payment?)
# ======================================================================
# Note what is NOT in this list: avg_savings_balance, liquid_assets, fd,
# mutual funds. We deliberately exclude what they ALREADY have, so the model
# is forced to learn from BEHAVIOUR. That is exactly your point - a customer
# with a small balance today may still be a strong saver.
SAVINGS_FEATURES = [
    "monthly_income", "income_stability", "salary_credits_6m", "annual_bonus",
    "rent_amount", "shopping_spend", "dining_spend", "travel_spend",
    "fuel_spend", "medical_spend", "education_fees", "utility_bill_count",
    "upi_txn_count", "avg_upi_ticket", "existing_emi", "emi_count",
    "cc_utilisation", "cc_outstanding", "dependents", "age", "city_tier",
    "discretionary_ratio", "foir", "months_ending_positive",
    "salary_day_balance_ratio", "balance_volatility", "sip_amount",
]

print("\nMODEL A trains on SPENDING BEHAVIOUR only - no current balance.\n")
savings_model, savings_best, savings_table, savings_eval = run_regression(
    df[SAVINGS_FEATURES], df["savings_velocity"],
    "MODEL A: MONTHLY SAVINGS CAPACITY (rupees per month)", " (Rs)")

df["predicted_monthly_saving"] = savings_model.predict(df[SAVINGS_FEATURES]).round(0)

# ======================================================================
# MODEL B - HIDDEN MONEY ELSEWHERE (share of wallet)
# ======================================================================
WALLET_FEATURES = [
    "monthly_income", "self_transfer_out", "self_transfer_count",
    "external_sip_debit", "external_cc_payment", "external_insurance_debit",
    "wallet_leak_ratio", "unexplained_outflow", "months_with_bank",
    "products_held", "age", "city_tier", "upi_txn_count",
    "salary_credits_6m", "credit_score", "cc_limit",
]

print("\nMODEL B learns the hidden pot from the OUTFLOW TRAIL.\n")
wallet_model, wallet_best, wallet_table, wallet_eval = run_regression(
    df[WALLET_FEATURES], df["true_external_assets"],
    "MODEL B: ESTIMATED MONEY HELD AT OTHER BANKS (rupees)", " (Rs)")

df["estimated_external_assets"] = wallet_model.predict(df[WALLET_FEATURES]).clip(0).round(-3)

print("=" * 74)
print("SHARE OF WALLET: HOW MUCH OF THE CUSTOMER DO WE ACTUALLY HOLD?")
print("=" * 74)
df["total_estimated_wealth"] = df["liquid_assets"] + df["estimated_external_assets"]
df["share_of_wallet"] = (df["liquid_assets"] /
                         df["total_estimated_wealth"].replace(0, np.nan)).fillna(1).round(3)

sow = pd.cut(df["share_of_wallet"], [-0.01, 0.25, 0.5, 0.75, 1.01],
             labels=["We hold <25%", "25-50%", "50-75%", "We hold >75%"])
t = df.groupby(sow, observed=True).agg(
    customers=("customer_id", "count"),
    avg_with_us=("liquid_assets", "mean"),
    avg_elsewhere=("estimated_external_assets", "mean"),
    home_loan_rate=("took_home_loan", "mean"))
t["avg_with_us"] = (t["avg_with_us"] / 1e5).round(1)
t["avg_elsewhere"] = (t["avg_elsewhere"] / 1e5).round(1)
t["home_loan_rate"] = (t["home_loan_rate"] * 100).round(1)
t.columns = ["customers", "with us (Rs L)", "elsewhere (Rs L)", "conversion %"]
print(t.to_string())

leaking = df[(df["share_of_wallet"] < 0.4) & (df["estimated_external_assets"] > 5_00_000)]
print(f"\n{len(leaking):,} customers keep most of their money elsewhere but transact with us.")
print(f"Estimated money sitting outside: Rs {leaking['estimated_external_assets'].sum()/1e7:,.0f} Cr")
print("These are prime targets: they trust us enough to transact, not enough to save.")

# ======================================================================
# PUT THEM TOGETHER: WHEN CAN THIS CUSTOMER AFFORD A DOWN PAYMENT?
# ======================================================================
print("\n" + "=" * 74)
print("DOWN-PAYMENT FORECAST (the answer to your question)")
print("=" * 74)

# Funding power = what they hold with us + what we think they hold elsewhere
# (discounted, since we cannot verify it) + what they will save in 12 months
df["funding_power_today"] = (df["liquid_assets"]
                             + 0.5 * df["estimated_external_assets"]).round(-3)
df["funding_power_12m"] = (df["funding_power_today"]
                           + 12 * df["predicted_monthly_saving"].clip(lower=0)).round(-3)

df["dp_gap_today"] = (df["downpayment_needed"] - df["funding_power_today"]).round(-3)
df["months_to_downpayment"] = np.where(
    df["dp_gap_today"] <= 0, 0,
    np.where(df["predicted_monthly_saving"] > 1000,
             (df["dp_gap_today"] / df["predicted_monthly_saving"]).round(0), 999))

def readiness_band(m):
    if m == 0:
        return "Ready now"
    if m <= 6:
        return "Ready in 6 months"
    if m <= 12:
        return "Ready in 12 months"
    if m <= 24:
        return "Ready in 24 months"
    return "Not on track"

df["dp_readiness_band"] = df["months_to_downpayment"].apply(readiness_band)

order = ["Ready now", "Ready in 6 months", "Ready in 12 months",
         "Ready in 24 months", "Not on track"]
band = df.groupby("dp_readiness_band").agg(
    customers=("customer_id", "count"),
    avg_saving_pm=("predicted_monthly_saving", "mean"),
    avg_gap=("dp_gap_today", "mean"),
    converted=("took_home_loan", "mean")).reindex(order)
band["avg_saving_pm"] = band["avg_saving_pm"].round(-2)
band["avg_gap"] = (band["avg_gap"] / 1e5).round(1)
band["converted"] = (band["converted"] * 100).round(1)
band.columns = ["customers", "saves Rs/month", "gap (Rs L)", "conversion %"]
print(band.to_string())

print("\nThis is the pipeline view a sales head actually wants:")
print("  'Ready now'        -> call this week")
print("  'Ready in 6-12m'   -> start an RD/SIP now, loan later, keep them with us")
print("  'Not on track'     -> do not waste calls; nurture instead")

# ---------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

Xte_s, yte_s, pred_s = savings_eval
axes[0].scatter(yte_s, pred_s, s=6, alpha=0.3, color="#2E5A88")
lim = [min(yte_s.min(), pred_s.min()), max(yte_s.max(), pred_s.max())]
axes[0].plot(lim, lim, "r--", lw=0.8)
axes[0].set_xlabel("Actual monthly saving"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Model A: savings capacity ({savings_best})")

Xte_w, yte_w, pred_w = wallet_eval
axes[1].scatter(yte_w / 1e5, pred_w / 1e5, s=6, alpha=0.3, color="#4C8C5A")
lim = [0, max(yte_w.max(), pred_w.max()) / 1e5]
axes[1].plot(lim, lim, "r--", lw=0.8)
axes[1].set_xlabel("Actual money elsewhere (Rs L)"); axes[1].set_ylabel("Estimated (Rs L)")
axes[1].set_title(f"Model B: hidden wealth ({wallet_best})")

band["customers"].plot(kind="bar", ax=axes[2], color="#C1703C")
axes[2].set_title("Down-payment pipeline")
axes[2].set_ylabel("Customers"); axes[2].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("savings_and_wallet.png", dpi=120)

joblib.dump({"savings": savings_model, "savings_features": SAVINGS_FEATURES,
             "wallet": wallet_model, "wallet_features": WALLET_FEATURES},
            "regression_models.pkl")

keep = ["customer_id", "predicted_monthly_saving", "estimated_external_assets",
        "total_estimated_wealth", "share_of_wallet", "funding_power_today",
        "funding_power_12m", "dp_gap_today", "months_to_downpayment",
        "dp_readiness_band"]
df[keep].to_csv("savings_forecast.csv", index=False)

print("\nSaved -> savings_and_wallet.png, savings_forecast.csv, regression_models.pkl")
