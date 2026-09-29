"""
STEP 2 - Explore the data before modelling.

Run this and read the output top to bottom. Every number here is something
you should be able to explain out loud in an interview.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

df = pd.read_csv("customer_data.csv")

print("=" * 72)
print("1. SHAPE, TYPES, MISSING VALUES")
print("=" * 72)
print(f"{df.shape[0]} rows, {df.shape[1]} columns")
miss = df.isna().sum()
miss = miss[miss > 0]
print("\nMissing:")
for c, n in miss.items():
    print(f"  {c:20s} {n:5d}  ({n/len(df):.1%})")
print("\nAll under 10% -> impute with the median rather than dropping rows.")

print("\n" + "=" * 72)
print("2. TARGET BALANCE")
print("=" * 72)
print(df["took_home_loan"].value_counts().to_string())
print(f"Positive rate: {df['took_home_loan'].mean():.1%}")
print("Imbalanced -> accuracy is a misleading metric here. Use ROC-AUC and recall.")

print("\n" + "=" * 72)
print("3. WHO CONVERTS vs WHO DOESN'T")
print("=" * 72)
cols = ["monthly_income", "rent_to_income", "foir", "cc_utilisation",
        "liquid_assets", "liquidity_months", "downpayment_readiness",
        "sip_amount", "products_held", "credit_score", "builder_txn_6m",
        "property_search_hits", "discretionary_ratio"]
cmp = df.groupby("took_home_loan")[cols].mean().T
cmp.columns = ["No loan", "Took loan"]
cmp["gap %"] = ((cmp["Took loan"] - cmp["No loan"]) / cmp["No loan"].abs() * 100).round(1)
print(cmp.round(2).to_string())

print("\n" + "=" * 72)
print("4. THE LIQUID-MONEY STORY (your core idea)")
print("=" * 72)
df["dp_bucket"] = pd.cut(df["downpayment_readiness"],
                         bins=[-0.01, 0.25, 0.5, 1.0, 3.01],
                         labels=["<25% ready", "25-50%", "50-100%", "Fully funded"])
t = df.groupby("dp_bucket", observed=True).agg(
    customers=("customer_id", "count"),
    conversion=("took_home_loan", "mean"),
    avg_liquid=("liquid_assets", "mean"),
)
t["conversion"] = (t["conversion"] * 100).round(1)
t["avg_liquid"] = t["avg_liquid"].round(-3)
print(t.to_string())
print("\nCustomers whose savings already cover the down payment convert far more.")
print("That is the single most actionable signal for the sales team.")

print("\n" + "=" * 72)
print("5. CONVERSION BY RENT BURDEN (renters only)")
print("=" * 72)
r = df[df["rent_amount"] > 0].copy()
r["rent_bucket"] = pd.cut(r["rent_to_income"],
                          bins=[0, 0.15, 0.20, 0.25, 0.30, 1.0],
                          labels=["<15%", "15-20%", "20-25%", "25-30%", ">30%"])
rt = r.groupby("rent_bucket", observed=True)["took_home_loan"].agg(["count", "mean"])
rt.columns = ["customers", "conversion"]
rt["conversion"] = (rt["conversion"] * 100).round(1)
print(rt.to_string())

print("\n" + "=" * 72)
print("6. RELATIONSHIP DEPTH: DOES HOLDING MORE PRODUCTS HELP?")
print("=" * 72)
p = df.groupby("products_held")["took_home_loan"].agg(["count", "mean"])
p.columns = ["customers", "conversion"]
p["conversion"] = (p["conversion"] * 100).round(1)
print(p.to_string())

print("\n" + "=" * 72)
print("7. STRONGEST CORRELATIONS WITH THE TARGET")
print("=" * 72)
corr = df.select_dtypes("number").corr()["took_home_loan"].drop("took_home_loan")
print("Top positive:")
print(corr.sort_values(ascending=False).head(8).round(3).to_string())
print("\nTop negative:")
print(corr.sort_values().head(5).round(3).to_string())

# --------------------------- charts ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

t["conversion"].plot(kind="bar", ax=axes[0, 0], color="#2E5A88")
axes[0, 0].set_title("Conversion by down-payment readiness")
axes[0, 0].set_ylabel("% who took a home loan")
axes[0, 0].tick_params(axis="x", rotation=0)

rt["conversion"].plot(kind="bar", ax=axes[0, 1], color="#4C8C5A")
axes[0, 1].set_title("Conversion by rent-to-income")
axes[0, 1].set_ylabel("%")
axes[0, 1].tick_params(axis="x", rotation=0)

corr.sort_values().tail(10).plot(kind="barh", ax=axes[1, 0], color="#C1703C")
axes[1, 0].set_title("Strongest positive signals")

axes[1, 1].scatter(df["liquidity_months"].clip(0, 30), df["foir"].clip(0, 0.6),
                   c=df["took_home_loan"], cmap="coolwarm", s=6, alpha=0.4)
axes[1, 1].set_xlabel("Liquid assets (months of income)")
axes[1, 1].set_ylabel("FOIR (existing obligations)")
axes[1, 1].set_title("Red = took a loan: high liquidity, low obligations")

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=120)
print("\nCharts saved -> eda_charts.png")
