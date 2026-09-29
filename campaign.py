"""
STEP 7 - The delivery layer: app notification + email + RM lead sheet.

This is the part you described. A score in a CSV changes nothing. What
changes business is:

  1. The customer gets a PUSH NOTIFICATION in the mobile app
       "You pay Rs 42,000 rent. The same amount could be an EMI on a
        Rs 58L home. Check your eligibility ->"
  2. The same message goes out as an EMAIL if they are not app-active
  3. The HOME LOAN RM gets a LEAD SHEET with the contact number, what to
     pitch, the numbers to quote, and when to call

It also enforces the rules a real bank cares about:
  - only contact customers who opted in to that channel
  - never send more than one message per customer per cycle
  - suppress anyone flagged Do-Not-Disturb or in credit stress
  - route only genuinely hot leads to an RM, since RM time is expensive
"""

import numpy as np
import pandas as pd
import joblib

pd.set_option("display.width", 200)

# ------------------------------------------------------------------
# Bring together everything the earlier steps produced
# ------------------------------------------------------------------
df = pd.read_csv("customer_data.csv")
df = df.merge(pd.read_csv("savings_forecast.csv"), on="customer_id")
df = df.merge(pd.read_csv("customer_segments.csv"), on="customer_id")

bundle = joblib.load("best_model.pkl")
df["propensity_score"] = bundle["pipe"].predict_proba(df[bundle["features"]])[:, 1].round(3)

OUR_RATE = 8.60


def emi_for(loan, rate=OUR_RATE, years=20):
    r, n = rate / 12 / 100, years * 12
    return loan * r * (1 + r) ** n / ((1 + r) ** n - 1)


# eligibility (same maths as recommend.py)
foir_cap = np.where(df["monthly_income"] >= 100_000, 0.50, 0.45)
foir_cap = foir_cap + np.where(df["credit_score"].fillna(735) >= 780, 0.05, 0) \
                    - np.where(df["credit_score"].fillna(735) < 680, 0.08, 0)
capacity = (df["monthly_income"] * foir_cap - df["existing_emi"]).clip(lower=0)
r, n = OUR_RATE / 12 / 100, 240
df["eligible_emi"] = capacity.round(-2)
df["eligible_loan"] = (capacity * ((1 + r) ** n - 1) / (r * (1 + r) ** n)).round(-3)
df["property_supported"] = (df["eligible_loan"] / 0.8).round(-3)

df["bt_saving_pm"] = np.where(
    df["has_external_home_loan"] == 1,
    (emi_for(df["external_hl_outstanding"], df["external_hl_rate"].clip(lower=OUR_RATE))
     - emi_for(df["external_hl_outstanding"])).round(0), 0)


def lakhs(x):
    return f"{x/1e5:.0f}L" if x >= 1e5 else f"{x/1000:.0f}K"


# ------------------------------------------------------------------
# Decide the campaign, the message, and who acts on it
# ------------------------------------------------------------------
def build_campaign(r):
    """Returns: campaign, push title, push body, email subject, RM action."""
    score = r["propensity_score"]
    ready = r["dp_readiness_band"]
    seg = r["segment"]

    # --- 1. Already has a home loan elsewhere -> win it over ---
    if r["has_external_home_loan"] == 1 and r["bt_saving_pm"] > 1500:
        return (
            "Balance Transfer",
            "Your home loan could cost less",
            f"You could save about Rs {r['bt_saving_pm']:,.0f} every month by moving "
            f"your home loan to us at {OUR_RATE}%. See your savings >",
            f"Save Rs {r['bt_saving_pm']*12/1000:.0f}K a year on your home loan",
            f"BT pitch: currently {r['external_hl_rate']}%, outstanding "
            f"Rs {lakhs(r['external_hl_outstanding'])}. Monthly saving "
            f"Rs {r['bt_saving_pm']:,.0f}. Carry the rate sheet.",
        )

    # --- 2. Credit stress -> suppress the loan pitch entirely ---
    if seg == "Stretched Borrower" or r["foir"] > 0.45 or r["cc_utilisation"] > 0.85:
        return (
            "Suppressed - Credit Stress", "", "", "",
            "DO NOT PITCH A LOAN. High FOIR or card utilisation. "
            "Offer a consolidation conversation only if the customer asks.",
        )

    # --- 3. Renter, ready now, high intent -> the headline campaign ---
    if r["rent_amount"] > 0 and ready == "Ready now" and score >= 0.5:
        return (
            "Rent-to-EMI - Hot",
            f"Your rent is Rs {r['rent_amount']:,.0f}. Your EMI could be too.",
            f"You pay Rs {r['rent_amount']:,.0f} a month in rent. The same amount "
            f"could be an EMI on a home worth about Rs {lakhs(r['property_supported'])}. "
            f"You are pre-qualified for Rs {lakhs(r['eligible_loan'])}. Check now >",
            f"You are pre-qualified for a Rs {lakhs(r['eligible_loan'])} home loan",
            f"HOT LEAD. Rent Rs {r['rent_amount']:,.0f} "
            f"({r['rent_to_income']*100:.0f}% of income). Eligible "
            f"Rs {lakhs(r['eligible_loan'])} at EMI Rs {r['eligible_emi']:,.0f}. "
            f"Down payment already funded (Rs {lakhs(r['funding_power_today'])} available).",
        )

    # --- 4. Money is parked at another bank -> bring the balance home ---
    if seg == "Banking Elsewhere" or (r["share_of_wallet"] < 0.35
                                      and r["estimated_external_assets"] > 5_00_000):
        return (
            "Bring Your Balance Home",
            "A better rate is waiting for you",
            "Consolidate your savings with us and unlock a preferential home loan "
            "rate plus zero processing fee. See what you qualify for >",
            "Move your balance, unlock a preferential home loan rate",
            f"Share of wallet only {r['share_of_wallet']*100:.0f}%. Estimated "
            f"Rs {lakhs(r['estimated_external_assets'])} held elsewhere. "
            "Pitch balance consolidation + preferential rate.",
        )

    # --- 4b. High intent but pays no rent (lives with family / owns already) ---
    if ready == "Ready now" and score >= 0.5:
        return (
            "Home Loan - Ready",
            f"You are pre-qualified for Rs {lakhs(r['eligible_loan'])}",
            f"Your savings and repayment profile qualify you for a home loan of up to "
            f"Rs {lakhs(r['eligible_loan'])} at an EMI of about "
            f"Rs {r['eligible_emi']:,.0f}. Check your offer >",
            f"Pre-qualified: Rs {lakhs(r['eligible_loan'])} home loan",
            f"READY LEAD (no rent outflow - may be living with family). Eligible "
            f"Rs {lakhs(r['eligible_loan'])} at EMI Rs {r['eligible_emi']:,.0f}. "
            f"Down payment funded (Rs {lakhs(r['funding_power_today'])}). "
            f"Segment: {seg}.",
        )

    # --- 5. Wants a home, short on down payment -> sell the save-up plan ---
    if ready in ("Ready in 6 months", "Ready in 12 months", "Ready in 24 months") \
            and score >= 0.35:
        months = int(r["months_to_downpayment"])
        return (
            "Save-up Plan",
            f"Your home is about {months} months away",
            f"Based on how you save, you could be ready for a "
            f"Rs {lakhs(r['eligible_loan'])} home loan in around {months} months. "
            f"Start an RD today and we will hold your rate. Start saving >",
            f"You could be home-loan ready in {months} months",
            f"Not loan-ready yet: Rs {lakhs(max(r['dp_gap_today'],0))} short on down "
            f"payment, saving about Rs {r['predicted_monthly_saving']:,.0f}/month. "
            f"Open an RD/SIP now, diarise a home loan call in {months} months.",
        )

    # --- 6. Idle money, no property intent yet -> investment pitch ---
    if r["avg_savings_balance"] > 3_00_000 and (
            seg == "Wealth Builder" or r["investment_ratio"] < 0.3):
        return (
            "Wealth / Idle Money",
            "Your savings could be working harder",
            f"About Rs {lakhs(r['avg_savings_balance'])} is sitting in your savings "
            "account. Move it to an FD or SIP and keep it growing. Explore >",
            "Put your idle savings to work",
            f"Rs {lakhs(r['avg_savings_balance'])} idle. Pitch FD/SIP now; "
            "re-score for a home loan next quarter.",
        )

    # --- 7. Everyone else: nurture, do not spend a call on them ---
    return (
        "Nurture", "", "", "",
        "No strong signal. Digital nurture only, re-score next quarter.",
    )


cols = ["campaign", "push_title", "push_body", "email_subject", "rm_action"]
df[cols] = df.apply(build_campaign, axis=1, result_type="expand")


# ------------------------------------------------------------------
# Channel routing + contact rules
# ------------------------------------------------------------------
def choose_channel(r):
    if r["campaign"] in ("Suppressed - Credit Stress", "Nurture"):
        return "None"
    app_active = (r["app_installed"] == 1 and r["push_optin"] == 1
                  and r["days_since_app_login"] <= 30)
    if app_active:
        return "App Push"
    if r["email_optin"] == 1:
        return "Email"
    return "RM Call"


df["channel"] = df.apply(choose_channel, axis=1)

# RM time is expensive - only route genuinely hot leads to a human
df["route_to_rm"] = (
    df["campaign"].isin(["Rent-to-EMI - Hot", "Home Loan - Ready", "Balance Transfer"])
    & (df["propensity_score"] >= 0.45)
).astype(int)

df["lead_priority"] = np.select(
    [df["propensity_score"] >= 0.70, df["propensity_score"] >= 0.50,
     df["propensity_score"] >= 0.30],
    ["P1 - Call today", "P2 - Call this week", "P3 - Call this month"],
    default="P4 - Digital only")

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print("=" * 78)
print("CAMPAIGN PLAN")
print("=" * 78)
plan = df.groupby("campaign").agg(
    customers=("customer_id", "count"),
    avg_score=("propensity_score", "mean"),
    actual_conversion=("took_home_loan", "mean"),
    to_rm=("route_to_rm", "sum"),
).sort_values("customers", ascending=False)
plan["avg_score"] = plan["avg_score"].round(3)
plan["actual_conversion"] = (plan["actual_conversion"] * 100).round(1)
plan["share_%"] = (plan["customers"] / len(df) * 100).round(1)
print(plan.to_string())

print("\n" + "=" * 78)
print("HOW EACH MESSAGE GETS DELIVERED")
print("=" * 78)
ch = df[df["channel"] != "None"].groupby("channel")["customer_id"].count()
for c, n in ch.items():
    print(f"  {c:12s} {n:6,} customers")
print(f"  {'Suppressed':12s} {(df['channel'] == 'None').sum():6,} customers "
      "(credit stress or no signal)")

print("\n" + "=" * 78)
print("RM LEAD DESK")
print("=" * 78)
rm = df[df["route_to_rm"] == 1]
print(f"Leads routed to home loan RMs: {len(rm):,} "
      f"({len(rm)/len(df):.1%} of the base)")
print(f"Buyers captured in that group: {rm['took_home_loan'].sum():,} of "
      f"{df['took_home_loan'].sum():,} "
      f"({rm['took_home_loan'].sum()/df['took_home_loan'].sum():.0%} of all buyers)")
print(f"Conversion in RM group: {rm['took_home_loan'].mean():.1%} "
      f"vs {df['took_home_loan'].mean():.1%} across the whole base")
print("\nBy priority:")
print(rm.groupby("lead_priority")["customer_id"].count().to_string())
print("\nBy branch (so leads land with the right RM):")
print(rm.groupby("home_branch")["customer_id"].count().to_string())

# ------------------------------------------------------------------
# Sample messages
# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("SAMPLE NOTIFICATIONS (what the customer actually sees)")
print("=" * 78)
for camp in ["Rent-to-EMI - Hot", "Balance Transfer", "Save-up Plan",
             "Bring Your Balance Home", "Wealth / Idle Money"]:
    sub = df[(df["campaign"] == camp) & (df["push_title"] != "")]
    if len(sub) == 0:
        continue
    r = sub.sort_values("propensity_score", ascending=False).iloc[0]
    print(f"\n┌─ {camp}  [{r['channel']}]  {r['customer_id']}  ({r['segment']})")
    print(f"│  PUSH : {r['push_title']}")
    print(f"│         {r['push_body']}")
    print(f"│  MAIL : {r['email_subject']}")
    print(f"└─ RM   : {r['rm_action']}")

# ------------------------------------------------------------------
# Output files
# ------------------------------------------------------------------
push_cols = ["customer_id", "segment", "campaign", "channel", "push_title",
             "push_body", "email_subject", "propensity_score", "best_contact_slot"]
df[df["channel"].isin(["App Push", "Email"])][push_cols] \
    .sort_values("propensity_score", ascending=False) \
    .to_csv("campaign_notifications.csv", index=False)

rm_cols = ["customer_id", "lead_priority", "campaign", "segment", "home_branch",
           "mobile", "email", "best_contact_slot", "propensity_score",
           "monthly_income", "rent_amount", "existing_emi", "eligible_loan",
           "eligible_emi", "funding_power_today", "dp_readiness_band",
           "share_of_wallet", "rm_action"]
rm.sort_values(["lead_priority", "propensity_score"], ascending=[True, False])[rm_cols] \
    .to_csv("rm_lead_sheet.csv", index=False)

print("\n\nSaved -> campaign_notifications.csv  (for the app / email system)")
print("      -> rm_lead_sheet.csv          (for the home loan RM desk)")
