"""
STEP 1 - Build a FAKE customer dataset.

Nothing here touches real bank data. Every number is invented by this script.
Each row = one customer, summarised the way a bank would summarise the last
6 months of their relationship:

  A) Who they are           - age, city, job type, dependents
  B) Money coming in        - salary, bonus, stability of credits
  C) Money going out        - rent, shopping, travel, fuel, school fees, medical
  D) What they owe          - EMIs, credit card usage, loans at other banks
  E) What they hold WITH US - savings balance, FD, mutual funds, SIP, insurance
  F) Intent signals         - builder payments, furniture spend, rent hike

(E) is the important one for your idea: liquid money sitting with the bank is
what decides whether a customer can actually fund a down payment.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 8000


def build(n):
    # ---------------------------------------------------------------
    # A) WHO THEY ARE
    # ---------------------------------------------------------------
    age = rng.integers(23, 58, n).astype(float)
    city_tier = rng.choice([1, 2, 3], n, p=[0.55, 0.30, 0.15])
    employment_type = rng.choice(["Salaried", "Self-employed"], n, p=[0.72, 0.28])
    dependents = rng.choice([0, 1, 2, 3, 4], n, p=[0.25, 0.25, 0.28, 0.15, 0.07])
    months_with_bank = rng.integers(3, 144, n)

    # ---------------------------------------------------------------
    # B) MONEY COMING IN
    # ---------------------------------------------------------------
    monthly_income = np.round(rng.lognormal(11.15, 0.5, n), -2).clip(25_000, 8_00_000)
    income_stability = np.where(
        employment_type == "Salaried",
        rng.uniform(0.80, 1.00, n),
        rng.uniform(0.35, 0.85, n),
    ).round(2)
    salary_credits_6m = np.where(employment_type == "Salaried",
                                 rng.integers(5, 7, n), rng.integers(2, 7, n))
    annual_bonus = np.where(
        (employment_type == "Salaried") & (rng.random(n) < 0.55),
        np.round(monthly_income * rng.uniform(0.5, 3.0, n), -2), 0)

    # ---------------------------------------------------------------
    # C) MONEY GOING OUT (transaction behaviour)
    # ---------------------------------------------------------------
    pays_rent = rng.random(n) < 0.58
    rent_amount = np.where(pays_rent,
                           np.round(monthly_income * rng.uniform(0.12, 0.42, n), -2), 0)
    rent_hike_flag = np.where(pays_rent, rng.random(n) < 0.22, False).astype(int)

    shopping_spend = np.round(monthly_income * rng.uniform(0.03, 0.22, n), -2)
    dining_spend = np.round(monthly_income * rng.uniform(0.01, 0.10, n), -2)
    travel_spend = np.round(monthly_income * rng.uniform(0.00, 0.14, n), -2)
    fuel_spend = np.round(monthly_income * rng.uniform(0.00, 0.06, n), -2)
    medical_spend = np.round(monthly_income * rng.uniform(0.00, 0.08, n), -2)
    education_fees = np.where(dependents > 0,
                              np.round(monthly_income * rng.uniform(0.02, 0.18, n), -2), 0)
    utility_bill_count = rng.integers(0, 7, n)
    upi_txn_count = rng.integers(10, 260, n)
    avg_upi_ticket = np.round(rng.uniform(150, 4000, n), 0)

    # ---------------------------------------------------------------
    # D) WHAT THEY OWE
    # ---------------------------------------------------------------
    emi_count = rng.choice([0, 1, 2, 3], n, p=[0.42, 0.33, 0.18, 0.07])
    existing_emi = np.where(emi_count > 0,
                            np.round(monthly_income * rng.uniform(0.04, 0.11, n)
                                     * emi_count, -2), 0)
    cc_limit = np.where(rng.random(n) < 0.65,
                        np.round(monthly_income * rng.uniform(1.0, 5.0, n), -3), 0)
    cc_utilisation = np.where(cc_limit > 0, rng.uniform(0.02, 0.95, n), 0).round(2)
    cc_outstanding = np.round(cc_limit * cc_utilisation, -2)

    # Loans held at OTHER banks - only visible with customer consent through the
    # RBI Account Aggregator framework. Kept as a separate flag on purpose.
    has_external_home_loan = (rng.random(n) < 0.18).astype(int)
    external_hl_rate = np.where(has_external_home_loan == 1,
                                rng.uniform(8.4, 10.9, n).round(2), 0)
    external_hl_outstanding = np.where(
        has_external_home_loan == 1,
        np.round(monthly_income * rng.uniform(15, 60, n), -3), 0)

    credit_score = np.clip(rng.normal(735, 68, n), 300, 900).astype(float)

    # ---------------------------------------------------------------
    # E) WHAT THEY HOLD WITH US (liquid money + investments)
    # ---------------------------------------------------------------
    avg_savings_balance = np.round(monthly_income * rng.uniform(0.2, 5.0, n), -2)
    fd_balance = np.where(rng.random(n) < 0.35,
                          np.round(monthly_income * rng.uniform(1, 30, n), -3), 0)
    mf_portfolio_value = np.where(rng.random(n) < 0.30,
                                  np.round(monthly_income * rng.uniform(1, 40, n), -3), 0)
    sip_amount = np.where(rng.random(n) < 0.38,
                          np.round(monthly_income * rng.uniform(0.02, 0.16, n), -2), 0)
    has_demat = ((mf_portfolio_value > 0) & (rng.random(n) < 0.7)).astype(int)
    insurance_premium_annual = np.where(
        rng.random(n) < 0.45,
        np.round(monthly_income * rng.uniform(0.3, 2.0, n), -2), 0)

    products_held = (
        1
        + (fd_balance > 0).astype(int)
        + (mf_portfolio_value > 0).astype(int)
        + (cc_limit > 0).astype(int)
        + (insurance_premium_annual > 0).astype(int)
        + has_demat
    )

    # ---------------------------------------------------------------
    # F) INTENT SIGNALS (hints of a home purchase)
    # ---------------------------------------------------------------
    builder_txn_6m = rng.poisson(0.22, n)
    home_improve_spend = np.where(rng.random(n) < 0.20,
                                  np.round(monthly_income * rng.uniform(0.05, 0.9, n), -2), 0)
    recent_marriage_flag = (rng.random(n) < 0.10).astype(int)
    property_search_hits = rng.poisson(0.6, n)

    # ---------------------------------------------------------------
    # G) MONEY HELD SOMEWHERE ELSE  (share of wallet)
    #
    # We cannot see another bank's balance. But money LEAVING our account
    # towards the customer's own other accounts leaves a trail we CAN see:
    #   - regular self-transfers (NEFT/IMPS to accounts in the same name)
    #   - SIP debits to AMCs where we are not the distributor
    #   - credit card bill payments to other banks
    #   - insurance premium debits to other insurers
    # A regression model learns to estimate the hidden pot from that trail.
    # ---------------------------------------------------------------
    true_external_assets = np.where(
        rng.random(n) < 0.62,
        np.round(monthly_income * rng.uniform(0.5, 45, n), -3), 0)

    leak = true_external_assets / np.maximum(monthly_income, 1)
    self_transfer_out = np.round(
        monthly_income * np.clip(leak * rng.uniform(0.004, 0.012, n), 0, 0.45), -2)
    self_transfer_count = np.where(self_transfer_out > 0,
                                   rng.integers(1, 12, n), 0)
    external_sip_debit = np.where(
        (true_external_assets > 0) & (rng.random(n) < 0.45),
        np.round(monthly_income * rng.uniform(0.01, 0.10, n), -2), 0)
    external_cc_payment = np.where(rng.random(n) < 0.40,
                                   np.round(monthly_income * rng.uniform(0.02, 0.30, n), -2), 0)
    external_insurance_debit = np.where(rng.random(n) < 0.30,
                                        np.round(monthly_income * rng.uniform(0.01, 0.08, n), -2), 0)

    # ---------------------------------------------------------------
    # H) SAVINGS BEHAVIOUR OVER TIME
    # Balance trend tells you whether they are actually accumulating,
    # which is what decides if they can fund a down payment later.
    # ---------------------------------------------------------------
    balance_trend_6m = np.round(rng.normal(0.02, 0.09, n), 3)          # % change per month
    min_balance_6m = np.round(avg_savings_balance * rng.uniform(0.15, 0.9, n), -2)
    max_balance_6m = np.round(avg_savings_balance * rng.uniform(1.1, 2.6, n), -2)
    salary_day_balance_ratio = np.round(rng.uniform(0.8, 3.5, n), 2)
    months_ending_positive = rng.integers(0, 7, n)

    # ---------------------------------------------------------------
    # I) DIGITAL / CONTACT (needed for the notification layer)
    # ---------------------------------------------------------------
    app_installed = (rng.random(n) < 0.78).astype(int)
    app_logins_30d = np.where(app_installed == 1, rng.poisson(9, n), 0)
    days_since_app_login = np.where(app_installed == 1, rng.integers(0, 90, n), 999)
    push_optin = np.where(app_installed == 1, (rng.random(n) < 0.72).astype(int), 0)
    email_optin = (rng.random(n) < 0.80).astype(int)
    preferred_channel = rng.choice(["App Push", "Email", "SMS", "RM Call"],
                                   n, p=[0.40, 0.25, 0.15, 0.20])
    best_contact_slot = rng.choice(["10am-1pm", "1pm-4pm", "4pm-7pm", "7pm-9pm"],
                                   n, p=[0.30, 0.22, 0.28, 0.20])
    mobile = ["+91-9" + "".join(rng.choice(list("0123456789"), 9)) for _ in range(n)]
    email = [f"customer{100000+i}@example.com" for i in range(n)]
    home_branch = rng.choice(["Koramangala", "Indiranagar", "Whitefield",
                              "Jayanagar", "MG Road", "Electronic City"], n)

    return pd.DataFrame({
        "customer_id": [f"CUST{100000+i}" for i in range(n)],
        "age": age,
        "city_tier": city_tier,
        "employment_type": employment_type,
        "dependents": dependents,
        "months_with_bank": months_with_bank,
        "monthly_income": monthly_income,
        "income_stability": income_stability,
        "salary_credits_6m": salary_credits_6m,
        "annual_bonus": annual_bonus,
        "rent_amount": rent_amount,
        "rent_hike_flag": rent_hike_flag,
        "shopping_spend": shopping_spend,
        "dining_spend": dining_spend,
        "travel_spend": travel_spend,
        "fuel_spend": fuel_spend,
        "medical_spend": medical_spend,
        "education_fees": education_fees,
        "utility_bill_count": utility_bill_count,
        "upi_txn_count": upi_txn_count,
        "avg_upi_ticket": avg_upi_ticket,
        "emi_count": emi_count,
        "existing_emi": existing_emi,
        "cc_limit": cc_limit,
        "cc_utilisation": cc_utilisation,
        "cc_outstanding": cc_outstanding,
        "has_external_home_loan": has_external_home_loan,
        "external_hl_rate": external_hl_rate,
        "external_hl_outstanding": external_hl_outstanding,
        "credit_score": credit_score,
        "avg_savings_balance": avg_savings_balance,
        "fd_balance": fd_balance,
        "mf_portfolio_value": mf_portfolio_value,
        "sip_amount": sip_amount,
        "has_demat": has_demat,
        "insurance_premium_annual": insurance_premium_annual,
        "products_held": products_held,
        "builder_txn_6m": builder_txn_6m,
        "home_improve_spend": home_improve_spend,
        "recent_marriage_flag": recent_marriage_flag,
        "property_search_hits": property_search_hits,
        # G) money elsewhere
        "self_transfer_out": self_transfer_out,
        "self_transfer_count": self_transfer_count,
        "external_sip_debit": external_sip_debit,
        "external_cc_payment": external_cc_payment,
        "external_insurance_debit": external_insurance_debit,
        "true_external_assets": true_external_assets,   # label for the model only
        # H) savings behaviour
        "balance_trend_6m": balance_trend_6m,
        "min_balance_6m": min_balance_6m,
        "max_balance_6m": max_balance_6m,
        "salary_day_balance_ratio": salary_day_balance_ratio,
        "months_ending_positive": months_ending_positive,
        # I) digital / contact
        "app_installed": app_installed,
        "app_logins_30d": app_logins_30d,
        "days_since_app_login": days_since_app_login,
        "push_optin": push_optin,
        "email_optin": email_optin,
        "preferred_channel": preferred_channel,
        "best_contact_slot": best_contact_slot,
        "mobile": mobile,
        "email": email,
        "home_branch": home_branch,
    })


def derive(df):
    """Ratios a credit manager actually looks at. This is feature engineering."""
    inc = df["monthly_income"]

    df["rent_to_income"] = (df["rent_amount"] / inc).round(3)
    df["foir"] = ((df["existing_emi"] + df["cc_outstanding"] * 0.05) / inc).round(3)

    df["total_discretionary"] = (df["shopping_spend"] + df["dining_spend"]
                                 + df["travel_spend"] + df["fuel_spend"])
    df["discretionary_ratio"] = (df["total_discretionary"] / inc).round(3)

    df["total_outflow"] = (df["rent_amount"] + df["existing_emi"] + df["education_fees"]
                           + df["medical_spend"] + df["total_discretionary"])
    df["savings_ratio"] = ((inc - df["total_outflow"]) / inc).round(3)
    df["monthly_surplus"] = (inc - df["total_outflow"]).round(0)

    # LIQUID MONEY WITH US - the heart of the idea
    df["liquid_assets"] = (df["avg_savings_balance"].fillna(0) + df["fd_balance"]
                           + df["mf_portfolio_value"])
    df["liquidity_months"] = (df["liquid_assets"] / inc).round(2)
    df["investment_ratio"] = ((df["fd_balance"] + df["mf_portfolio_value"]) /
                              df["liquid_assets"].replace(0, np.nan)).fillna(0).round(3)

    # Rough eligibility: how much loan can this EMI capacity support?
    # EMI capacity = 50% of income, minus EMIs already running.
    df["emi_capacity"] = (inc * 0.50 - df["existing_emi"]).clip(lower=0).round(0)
    # 20-year loan at about 8.75% -> roughly 114x the monthly EMI
    df["indicative_loan_amount"] = (df["emi_capacity"] * 114).round(-3)
    # Down payment = 20% of property value; property value ~ loan / 0.8
    df["downpayment_needed"] = (df["indicative_loan_amount"] / 0.8 * 0.20).round(-3)
    df["downpayment_readiness"] = (df["liquid_assets"] /
                                   df["downpayment_needed"].replace(0, np.nan)
                                   ).fillna(0).clip(0, 3).round(2)

    df["rent_vs_emi_gap"] = (df["rent_amount"] - df["emi_capacity"]).round(0)

    # --- SAVINGS VELOCITY: are they actually accumulating month on month? ---
    # Surplus on paper is not the same as money that stays in the account.
    df["balance_growth_amount"] = (df["avg_savings_balance"].fillna(0)
                                   * df["balance_trend_6m"]).round(0)
    df["savings_velocity"] = (
        0.55 * df["monthly_surplus"] + 0.45 * df["balance_growth_amount"]
    ).round(0)
    df["savings_consistency"] = (df["months_ending_positive"] / 6).round(2)
    df["balance_volatility"] = ((df["max_balance_6m"] - df["min_balance_6m"]) /
                                df["avg_savings_balance"].replace(0, np.nan)
                                ).fillna(0).round(2)

    # --- WALLET LEAK: money flowing out to the customer's other accounts ---
    df["wallet_leak_total"] = (df["self_transfer_out"] + df["external_sip_debit"]
                               + df["external_cc_payment"]
                               + df["external_insurance_debit"])
    df["wallet_leak_ratio"] = (df["wallet_leak_total"] / inc).round(3)
    df["unexplained_outflow"] = (
        inc - df["total_outflow"] - df["wallet_leak_total"]
        - df["balance_growth_amount"]).round(0)
    return df


def make_target(df):
    """
    Who actually took a home loan in the next 6 months?

    Built as score -> probability -> weighted coin flip. The randomness is
    deliberate: real customers are not perfectly predictable, and a model that
    scores 100% is a model that has leaked its answer.
    """
    z = (
        -3.6
        + 1.6 * df["rent_to_income"]
        + 0.8 * df["rent_hike_flag"]
        + 0.9 * df["builder_txn_6m"]
        + 0.35 * df["property_search_hits"]
        + 0.6 * df["recent_marriage_flag"]
        + 1.6 * df["downpayment_readiness"].clip(0, 2) / 2
        + 0.5 * df["liquidity_months"].clip(0, 12) / 12
        + 0.4 * (df["sip_amount"] > 0)
        + 0.25 * df["products_held"] / 6
        + 0.8 * (df["credit_score"].fillna(735) - 735) / 100
        + 0.8 * df["income_stability"].fillna(0.8)
        - 2.2 * df["foir"]
        - 0.9 * df["cc_utilisation"]
        - 0.9 * df["has_external_home_loan"]
        - 0.025 * (df["age"].fillna(35) - 30).clip(0, None)
        + 0.3 * (df["months_with_bank"] > 24)
    )
    prob = 1 / (1 + np.exp(-z))
    df["took_home_loan"] = (rng.random(len(df)) < prob).astype(int)
    return df


def punch_holes(df):
    """Real data always has gaps. Add some so the project is realistic."""
    for col, frac in [("credit_score", 0.07), ("age", 0.02),
                      ("income_stability", 0.03)]:
        idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


if __name__ == "__main__":
    df = build(N)
    df = derive(df)
    df = make_target(df)
    df = punch_holes(df)
    df.to_csv("customer_data.csv", index=False)

    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    print(f"Took home loan: {df['took_home_loan'].mean():.1%}")
    print(f"Hold a home loan elsewhere (BT targets): {df['has_external_home_loan'].mean():.1%}")
    print(f"Have money invested with us: {(df['fd_balance'] + df['mf_portfolio_value'] > 0).mean():.1%}")
    print("\nMissing values:")
    print(df.isna().sum()[lambda s: s > 0].to_string())
    print("\nSaved -> customer_data.csv")
