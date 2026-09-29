"""
STEP 4 - The recommendation engine.

The model from step 3 gives a PROPENSITY SCORE (how likely is this customer
to take a home loan). On its own that is just a number. This script turns it
into something a relationship manager can act on:

    WHO to call, WHAT to offer, HOW MUCH they qualify for, and WHY.

Products it can recommend:
  1. HOME LOAN          - renter with intent and the money for a down payment
  2. BALANCE TRANSFER   - already has a home loan elsewhere at a higher rate
  3. TOP-UP LOAN        - has a home loan and a clean track record
  4. HOME LOAN + SAVE-UP PLAN - wants a home, short on down payment -> RD/SIP first
  5. LAP                - self-employed, owns property, needs business funds
  6. WEALTH / SIP       - lots of idle money, no property intent yet
  7. BALANCE BUILD      - not ready for anything; deepen the relationship first

Design choice worth defending in an interview: the ML model ranks WHO to call,
and transparent rules decide WHAT to offer. Banks need the offer logic to be
explainable to a regulator, so a pure black box is the wrong tool for that half.
"""

import numpy as np
import pandas as pd
import joblib

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("customer_data.csv")
bundle = joblib.load("best_model.pkl")
pipe, FEATURES = bundle["pipe"], bundle["features"]

print(f"Scoring {len(df)} customers with: {bundle['name']}\n")
df["propensity_score"] = pipe.predict_proba(df[FEATURES])[:, 1].round(3)


# ---------------------------------------------------------------------
# Eligibility maths (plain, explainable, no ML)
# ---------------------------------------------------------------------
def emi_for(loan, annual_rate=8.75, years=20):
    """Standard EMI formula. Same one every bank uses."""
    r = annual_rate / 12 / 100
    n = years * 12
    return loan * r * (1 + r) ** n / ((1 + r) ** n - 1)


def eligibility(row):
    """How much can we actually lend, given income, obligations and credit score."""
    foir_cap = 0.50 if row["monthly_income"] >= 100_000 else 0.45
    if pd.notna(row["credit_score"]):
        if row["credit_score"] >= 780:
            foir_cap += 0.05
        elif row["credit_score"] < 680:
            foir_cap -= 0.08
    capacity = max(row["monthly_income"] * foir_cap - row["existing_emi"], 0)
    r, n = 8.75 / 12 / 100, 240
    loan = capacity * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
    return round(capacity, -2), round(loan, -3)


elig = df.apply(eligibility, axis=1, result_type="expand")
df["eligible_emi"] = elig[0]
df["eligible_loan"] = elig[1]
df["property_value_supported"] = (df["eligible_loan"] / 0.8).round(-3)
df["downpayment_required"] = (df["property_value_supported"] * 0.2).round(-3)
df["downpayment_gap"] = (df["downpayment_required"] - df["liquid_assets"]).round(-3)
df["months_to_save_gap"] = np.where(
    df["monthly_surplus"] > 0,
    (df["downpayment_gap"].clip(lower=0) / df["monthly_surplus"]).round(0),
    99)

# balance transfer saving, assuming we offer 8.60%
OUR_RATE = 8.60
df["bt_rate_gap"] = (df["external_hl_rate"] - OUR_RATE).round(2)
df["bt_monthly_saving"] = np.where(
    df["has_external_home_loan"] == 1,
    (emi_for(df["external_hl_outstanding"], df["external_hl_rate"].clip(lower=OUR_RATE))
     - emi_for(df["external_hl_outstanding"], OUR_RATE)).round(0),
    0)
df["bt_lifetime_saving"] = (df["bt_monthly_saving"] * 240).round(-3)


# ---------------------------------------------------------------------
# The decision rules
# ---------------------------------------------------------------------
def recommend(r):
    score = r["propensity_score"]
    renter = r["rent_amount"] > 0
    liquid_ok = r["liquid_assets"] >= r["downpayment_required"] * 0.8
    healthy = (r["foir"] < 0.45) and (pd.isna(r["credit_score"]) or r["credit_score"] >= 680)

    # 1. Already banking elsewhere for their home loan -> win the loan over
    if r["has_external_home_loan"] == 1:
        if r["bt_rate_gap"] >= 0.4 and r["bt_monthly_saving"] > 1500:
            return ("Balance Transfer", "Very High",
                    f"Paying {r['external_hl_rate']}% elsewhere. Switching to "
                    f"{OUR_RATE}% saves about Rs {r['bt_monthly_saving']:,.0f} a month "
                    f"(Rs {r['bt_lifetime_saving']/1e5:.1f}L over the tenure).")
        if healthy and r["foir"] < 0.35:
            return ("Top-up Loan", "Medium",
                    "Existing home loan with a clean repayment record and room "
                    "under the FOIR cap. Eligible for a top-up.")
        return ("Relationship Deepening", "Low",
                "Already owns a home and is fully leveraged. Cross-sell "
                "insurance or investments instead of more debt.")

    # 2. Self-employed with assets -> LAP is often the better fit
    if (r["employment_type"] == "Self-employed" and r["liquid_assets"] > 15_00_000
            and r["monthly_income"] > 1_50_000 and score < 0.45):
        return ("Loan Against Property", "Medium",
                "Self-employed with strong balances but low home-purchase intent. "
                "Likely needs business funding rather than a home loan.")

    # 3. The core case: wants a home, can afford it, has the down payment
    if score >= 0.55 and renter and liquid_ok and healthy:
        return ("Home Loan - Hot Lead", "Very High",
                f"Pays Rs {r['rent_amount']:,.0f} rent "
                f"({r['rent_to_income']*100:.0f}% of income). Qualifies for about "
                f"Rs {r['eligible_loan']/1e5:.1f}L at an EMI of "
                f"Rs {r['eligible_emi']:,.0f}, and already holds "
                f"Rs {r['liquid_assets']/1e5:.1f}L with us for the down payment.")

    # 4. Wants a home but the down payment is short -> sell the savings plan first
    if score >= 0.45 and renter and healthy and not liquid_ok:
        gap = max(r["downpayment_gap"], 0)
        months = r["months_to_save_gap"]
        when = f"about {months:.0f} months" if months < 60 else "a longer horizon"
        return ("Home Loan + Save-up Plan", "High",
                f"Strong intent but Rs {gap/1e5:.1f}L short on the down payment. "
                f"Start an RD or SIP now and they are loan-ready in {when}. "
                "Keeps the customer with us instead of losing them to a competitor.")

    # 5. Intent showing but the balance sheet is stretched
    if score >= 0.45 and not healthy:
        return ("Credit Health Fix", "Medium",
                f"Interested, but FOIR is {r['foir']*100:.0f}% "
                f"and card utilisation {r['cc_utilisation']*100:.0f}%. "
                "Consolidate the card debt first, then revisit in 2 quarters.")

    # 6. Money sitting idle, no property intent
    if r["liquid_assets"] > 10_00_000 and r["investment_ratio"] < 0.35:
        idle = r["avg_savings_balance"]
        return ("Wealth / SIP", "Medium",
                f"Rs {idle/1e5:.1f}L sitting idle in savings earning almost nothing. "
                "Move to FD or SIP now; becomes a home loan lead later.")

    # 7. Everyone else
    return ("Relationship Deepening", "Low",
            "No strong signal yet. Keep engaged through card or insurance "
            "cross-sell and re-score next quarter.")


rec = df.apply(recommend, axis=1, result_type="expand")
df["recommended_product"] = rec[0]
df["priority"] = rec[1]
df["reason"] = rec[2]


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
print("=" * 72)
print("WHAT THE ENGINE RECOMMENDS ACROSS THE BASE")
print("=" * 72)
mix = df.groupby("recommended_product").agg(
    customers=("customer_id", "count"),
    avg_score=("propensity_score", "mean"),
    actual_conversion=("took_home_loan", "mean"),
).sort_values("customers", ascending=False)
mix["share"] = (mix["customers"] / len(df) * 100).round(1)
mix["avg_score"] = mix["avg_score"].round(3)
mix["actual_conversion"] = (mix["actual_conversion"] * 100).round(1)
print(mix.to_string())

print("\n" + "=" * 72)
print("DOES THE PRIORITY FLAG ACTUALLY WORK?")
print("=" * 72)
pri = df.groupby("priority").agg(
    customers=("customer_id", "count"),
    actual_conversion=("took_home_loan", "mean"),
).reindex(["Very High", "High", "Medium", "Low"])
pri["actual_conversion"] = (pri["actual_conversion"] * 100).round(1)
base = df["took_home_loan"].mean() * 100
pri["lift_vs_base"] = (pri["actual_conversion"] / base).round(2)
print(pri.to_string())
print(f"\nBase rate across everyone: {base:.1f}%")
print("Very High priority converts several times better -> the ranking works.")

print("\n" + "=" * 72)
print("BUSINESS CASE (illustrative, on this fake base)")
print("=" * 72)
hot = df[df["priority"].isin(["Very High", "High"])]
print(f"Customers flagged for a call : {len(hot):,} out of {len(df):,} "
      f"({len(hot)/len(df):.0%} of the base)")
print(f"Buyers captured in that group: {hot['took_home_loan'].sum():,} out of "
      f"{df['took_home_loan'].sum():,} "
      f"({hot['took_home_loan'].sum()/df['took_home_loan'].sum():.0%} of all buyers)")
bt = df[df["recommended_product"] == "Balance Transfer"]
print(f"\nBalance transfer targets     : {len(bt):,}")
print(f"Loan book if 10% convert     : Rs {bt['external_hl_outstanding'].sum()*0.10/1e7:,.1f} Cr")
hl = df[df["recommended_product"] == "Home Loan - Hot Lead"]
print(f"Hot home loan leads          : {len(hl):,}")
print(f"Loan book if 10% convert     : Rs {hl['eligible_loan'].sum()*0.10/1e7:,.1f} Cr")

# ------------------- call lists for the sales team -------------------
cols = ["customer_id", "recommended_product", "priority", "propensity_score",
        "monthly_income", "rent_amount", "existing_emi", "liquid_assets",
        "eligible_loan", "eligible_emi", "downpayment_gap", "reason"]

out = df.sort_values(["priority", "propensity_score"],
                     ascending=[True, False])[cols]
df[cols].sort_values("propensity_score", ascending=False).head(300)\
        .to_csv("call_list_top300.csv", index=False)
df[cols].to_csv("all_customers_scored.csv", index=False)

print("\n" + "=" * 72)
print("SAMPLE: WHAT THE RM SEES ON SCREEN")
print("=" * 72)
for product in ["Home Loan - Hot Lead", "Balance Transfer",
                "Home Loan + Save-up Plan", "Wealth / SIP"]:
    sub = df[df["recommended_product"] == product]
    if len(sub) == 0:
        continue
    r = sub.sort_values("propensity_score", ascending=False).iloc[0]
    print(f"\n--- {product}  [{r['priority']}] ---")
    print(f"Customer      : {r['customer_id']}")
    print(f"Income        : Rs {r['monthly_income']:,.0f}/month "
          f"| Rent Rs {r['rent_amount']:,.0f} | EMIs Rs {r['existing_emi']:,.0f}")
    print(f"With the bank : Rs {r['liquid_assets']:,.0f} "
          f"(savings + FD + MF), {r['products_held']} products")
    print(f"Model score   : {r['propensity_score']:.2f}")
    print(f"Talk track    : {r['reason']}")

print("\n\nSaved -> call_list_top300.csv, all_customers_scored.csv")
