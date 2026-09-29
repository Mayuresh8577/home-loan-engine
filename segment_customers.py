"""
STEP 6 - Understand the whole spending cycle: customer segmentation.

Everything so far was SUPERVISED learning (we had an answer to learn from).
This is UNSUPERVISED learning: nobody labels customers as "Big Spender" or
"Quiet Saver". KMeans finds natural groups in the data by itself, and we read
the groups afterwards and name them.

Why this matters for your pitch: the same home loan message should not go to
a 26-year-old spending everything they earn and a 40-year-old quietly building
a corpus. Segments let you change the MESSAGE, not just the target list.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RS = 42
df = pd.read_csv("customer_data.csv")

# Behaviour only - not income size. We want to group by HOW people handle
# money, not by how much they earn.
BEHAVIOUR = [
    "discretionary_ratio",      # how much of income goes on lifestyle
    "savings_ratio",            # how much is left over
    "rent_to_income",           # housing burden
    "foir",                     # debt burden
    "cc_utilisation",           # credit dependence
    "investment_ratio",         # invested vs idle
    "liquidity_months",         # cushion in months of income
    "upi_txn_count",            # transaction activity
    "savings_consistency",      # how often they end the month positive
    "balance_volatility",       # how wildly the balance swings
    "wallet_leak_ratio",        # money moving to other banks
]

X = df[BEHAVIOUR]

# ------------------------------------------------------------------
# How many clusters? Try a few and look at the silhouette score.
# ------------------------------------------------------------------
print("Choosing the number of segments:\n")
print(f"{'k':>3} {'silhouette':>12}   (higher = better separated)")
scores = {}
for k in range(3, 8):
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("km", KMeans(n_clusters=k, n_init=10, random_state=RS))])
    labels = pipe.fit_predict(X)
    s = silhouette_score(pipe[:-1].transform(X), labels)
    scores[k] = s
    print(f"{k:>3} {s:>12.3f}")

K = 5   # readable for a business audience, and scores well
print(f"\nUsing k={K}: enough detail for a campaign, few enough to explain.\n")

pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler()),
                 ("km", KMeans(n_clusters=K, n_init=20, random_state=RS))])
df["segment_id"] = pipe.fit_predict(X)

# ------------------------------------------------------------------
# Read the segments and give them business names
# ------------------------------------------------------------------
profile = df.groupby("segment_id")[BEHAVIOUR + ["monthly_income", "age",
                                                "liquid_assets", "took_home_loan"]].mean()

# Compare clusters to EACH OTHER (z-scores), not to fixed thresholds.
# Then match each cluster to the archetype it fits best, one name each.
z = ((profile - profile.mean()) / profile.std().replace(0, 1))

ARCHETYPES = {
    "Stretched Borrower":  {"foir": 2.0, "cc_utilisation": 1.0, "savings_ratio": -0.5},
    "Wealth Builder":      {"liquidity_months": 1.5, "investment_ratio": 1.5,
                            "liquid_assets": 1.0},
    "Banking Elsewhere":   {"wallet_leak_ratio": 2.0, "liquidity_months": -0.8},
    "Lifestyle Spender":   {"discretionary_ratio": 1.5, "savings_ratio": -1.5,
                            "upi_txn_count": 0.5},
    "Steady Saver":        {"savings_ratio": 1.5, "savings_consistency": 1.0,
                            "foir": -0.5},
}

# score every cluster against every archetype, then assign greedily
pairs = []
for sid in profile.index:
    for nm, weights in ARCHETYPES.items():
        score = sum(w * z.loc[sid, col] for col, w in weights.items()
                    if col in z.columns)
        pairs.append((score, sid, nm))
pairs.sort(reverse=True)

names, taken_sid, taken_nm = {}, set(), set()
for score, sid, nm in pairs:
    if sid in taken_sid or nm in taken_nm:
        continue
    names[sid] = nm
    taken_sid.add(sid)
    taken_nm.add(nm)

df["segment"] = df["segment_id"].map(names)

print("=" * 78)
print("THE SEGMENTS KMEANS FOUND")
print("=" * 78)
summary = df.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_income=("monthly_income", "mean"),
    discretionary=("discretionary_ratio", "mean"),
    savings_rate=("savings_ratio", "mean"),
    foir=("foir", "mean"),
    liquid_Rs_L=("liquid_assets", "mean"),
    leak_ratio=("wallet_leak_ratio", "mean"),
    conversion=("took_home_loan", "mean"),
).sort_values("customers", ascending=False)
summary["avg_income"] = summary["avg_income"].round(-2)
summary["liquid_Rs_L"] = (summary["liquid_Rs_L"] / 1e5).round(1)
summary["conversion"] = (summary["conversion"] * 100).round(1)
summary[["discretionary", "savings_rate", "foir", "leak_ratio"]] = \
    summary[["discretionary", "savings_rate", "foir", "leak_ratio"]].round(3)
print(summary.to_string())

# ------------------------------------------------------------------
# What to say to each segment
# ------------------------------------------------------------------
PLAYBOOK = {
    "Steady Saver":
        "Reliable surplus every month. Strongest home loan audience. "
        "Message: you are already saving an EMI's worth - own instead of rent.",
    "Wealth Builder":
        "Money is invested, not idle. Do not ask them to break investments. "
        "Message: keep the portfolio, fund the down payment with a loan against MF or FD.",
    "Lifestyle Spender":
        "High discretionary spend, thin savings. Not loan-ready today. "
        "Message: start a small RD or SIP; show them the down payment is reachable.",
    "Stretched Borrower":
        "High FOIR or card utilisation. Do not push more debt. "
        "Message: consolidate the card outstanding first, revisit in two quarters.",
    "Banking Elsewhere":
        "Transacts with us but saves elsewhere. Biggest untapped pool. "
        "Message: bring the balance here, get a preferential home loan rate.",
}

print("\n" + "=" * 78)
print("WHAT TO SAY TO EACH SEGMENT")
print("=" * 78)
for seg in summary.index:
    base = seg
    n = summary.loc[seg, "customers"]
    print(f"\n{seg}  ({n:,} customers, {summary.loc[seg,'conversion']}% convert)")
    print(f"  {PLAYBOOK.get(base, 'Nurture and re-score next quarter.')}")

# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------
coords = PCA(n_components=2, random_state=RS).fit_transform(pipe[:-1].transform(X))
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

for seg in df["segment"].unique():
    m = df["segment"] == seg
    axes[0].scatter(coords[m, 0], coords[m, 1], s=5, alpha=0.4, label=seg)
axes[0].set_title("Segments (PCA view)")
axes[0].legend(fontsize=7, markerscale=2)
axes[0].set_xlabel("Component 1"); axes[0].set_ylabel("Component 2")

summary["customers"].plot(kind="barh", ax=axes[1], color="#2E5A88")
axes[1].set_title("Segment sizes")

summary["conversion"].plot(kind="barh", ax=axes[2], color="#4C8C5A")
axes[2].set_title("Home loan conversion by segment (%)")

plt.tight_layout()
plt.savefig("segments.png", dpi=120)

df[["customer_id", "segment", "segment_id"]].to_csv("customer_segments.csv", index=False)
print("\n\nSaved -> segments.png, customer_segments.csv")
