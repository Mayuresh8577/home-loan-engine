"""
STEP 3 - Train several ML models and pick the best one.

Models compared:
  1. Logistic Regression  - simple, explainable, what banks usually prefer
  2. Decision Tree        - easy to show to a business audience
  3. Random Forest        - many trees voting, stronger
  4. Gradient Boosting    - trees that fix each other's mistakes, usually best

Everything sits inside a Pipeline so missing-value filling and scaling are
learned from the TRAINING data only. That is how data leakage is avoided.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

RANDOM_STATE = 42

df = pd.read_csv("customer_data.csv")

# ------------------------------------------------------------------
# 1. Pick features. Grouped so you can explain each block in an interview.
# ------------------------------------------------------------------
PROFILE = ["age", "city_tier", "dependents", "months_with_bank"]
INCOME = ["monthly_income", "income_stability", "salary_credits_6m", "annual_bonus"]
SPEND = ["rent_amount", "rent_hike_flag", "shopping_spend", "dining_spend",
         "travel_spend", "fuel_spend", "medical_spend", "education_fees",
         "utility_bill_count", "upi_txn_count", "avg_upi_ticket",
         "home_improve_spend"]
LIABILITY = ["emi_count", "existing_emi", "cc_limit", "cc_utilisation",
             "cc_outstanding", "has_external_home_loan", "credit_score"]
ASSETS = ["avg_savings_balance", "fd_balance", "mf_portfolio_value", "sip_amount",
          "has_demat", "insurance_premium_annual", "products_held"]
INTENT = ["builder_txn_6m", "recent_marriage_flag", "property_search_hits"]
DERIVED = ["rent_to_income", "foir", "discretionary_ratio", "savings_ratio",
           "monthly_surplus", "liquid_assets", "liquidity_months",
           "investment_ratio", "emi_capacity", "downpayment_readiness",
           "rent_vs_emi_gap"]

NUMERIC = PROFILE + INCOME + SPEND + LIABILITY + ASSETS + INTENT + DERIVED
CATEGORICAL = ["employment_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "took_home_loan"

print(f"Using {len(FEATURES)} features across 7 groups\n")

X, y = df[FEATURES], df[TARGET]

# ------------------------------------------------------------------
# 2. Split first, clean later. Stratify keeps the positive rate equal.
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"Train {len(X_train)} rows | Test {len(X_test)} rows")
print(f"Positive rate -> train {y_train.mean():.1%}, test {y_test.mean():.1%}\n")

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=40, class_weight="balanced",
        random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=15,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.08,
        random_state=RANDOM_STATE),
}

results, summary = {}, []

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    auc = roc_auc_score(y_test, prob)
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)

    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    summary.append({
        "Model": name,
        "Test AUC": round(auc, 3),
        "CV AUC": round(cv.mean(), 3),
        "CV std": round(cv.std(), 3),
        "Precision": round(rep["1"]["precision"], 3),
        "Recall": round(rep["1"]["recall"], 3),
        "F1": round(rep["1"]["f1-score"], 3),
    })
    results[name] = {"pipe": pipe, "prob": prob, "pred": pred, "auc": auc}
    print(f"  trained: {name:22s} AUC {auc:.3f}")

print("\n" + "=" * 72)
print("MODEL COMPARISON")
print("=" * 72)
comp = pd.DataFrame(summary).sort_values("Test AUC", ascending=False)
print(comp.to_string(index=False))

best_name = comp.iloc[0]["Model"]
best = results[best_name]
print(f"\nBest model: {best_name}")
print("CV std is small, so the score is stable and not a lucky split.")

print("\n" + "=" * 72)
print(f"DETAIL FOR {best_name}")
print("=" * 72)
cm = confusion_matrix(y_test, best["pred"])
print("Confusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(cm,
                   index=["Actual: no loan", "Actual: took loan"],
                   columns=["Pred: no", "Pred: yes"]).to_string())
print("\n" + classification_report(y_test, best["pred"],
                                   target_names=["No loan", "Took loan"],
                                   zero_division=0))

# ------------------------------------------------------------------
# 3. What drives the prediction?
# ------------------------------------------------------------------
feat_names = NUMERIC + list(
    results[best_name]["pipe"].named_steps["prep"]
    .named_transformers_["cat"].get_feature_names_out(CATEGORICAL))

print("=" * 72)
print("WHAT THE MODEL LOOKS AT")
print("=" * 72)
model_obj = best["pipe"].named_steps["model"]
if hasattr(model_obj, "feature_importances_"):
    imp = pd.Series(model_obj.feature_importances_, index=feat_names)
else:
    imp = pd.Series(np.abs(model_obj.coef_[0]), index=feat_names)
imp = imp.sort_values(ascending=False)
print(imp.head(12).round(4).to_string())

lr = results["Logistic Regression"]["pipe"].named_steps["model"]
coefs = pd.Series(lr.coef_[0], index=feat_names).sort_values()
print("\nDirection of effect (Logistic Regression coefficients):")
print("  Pushes AWAY from a home loan:")
print(coefs.head(5).round(3).to_string())
print("  Pushes TOWARDS a home loan:")
print(coefs.tail(6).round(3).to_string())

# ------------------------------------------------------------------
# 4. The business view: targeting beats mass calling
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("BUSINESS IMPACT: CALL THE TOP-SCORED CUSTOMERS ONLY")
print("=" * 72)
scored = (pd.DataFrame({"prob": best["prob"], "actual": y_test.values})
          .sort_values("prob", ascending=False).reset_index(drop=True))
base = scored["actual"].mean()
print(f"Calling everyone converts at {base:.1%}\n")
print(f"{'Top %':>6} {'Called':>8} {'Buyers found':>14} {'Hit rate':>10} {'Lift':>7}")
for pct in [5, 10, 20, 30, 50]:
    n = int(len(scored) * pct / 100)
    top = scored.head(n)
    hit = top["actual"].mean()
    print(f"{pct:>5}% {n:>8} {int(top['actual'].sum()):>14} {hit:>9.1%} {hit/base:>6.2f}x")

n10 = int(len(scored) * 0.10)
found10 = scored.head(n10)["actual"].sum()
total = scored["actual"].sum()
print(f"\nOne-line pitch: calling the top 10% of customers reaches "
      f"{found10/total:.0%} of all buyers while making 90% fewer calls.")

# ------------------------------------------------------------------
# 5. Charts + save model
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    axes[0].plot(fpr, tpr, label=f"{name} ({r['auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Random guessing")
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve: all four models"); axes[0].legend(fontsize=8)

imp.head(12).sort_values().plot(kind="barh", ax=axes[1], color="#2E5A88")
axes[1].set_title(f"Top signals ({best_name})")

lift = [scored.head(int(len(scored)*p/100))["actual"].mean()/base
        for p in range(10, 101, 10)]
axes[2].bar(range(10, 101, 10), lift, width=7, color="#4C8C5A")
axes[2].axhline(1, color="red", ls="--", lw=0.8)
axes[2].set_xlabel("Top % of customers contacted")
axes[2].set_ylabel("Lift over random calling")
axes[2].set_title("Why targeting beats mass calling")

plt.tight_layout()
plt.savefig("model_results.png", dpi=120)

joblib.dump({"pipe": best["pipe"], "features": FEATURES, "name": best_name},
            "best_model.pkl")
comp.to_csv("model_comparison.csv", index=False)

print("\nSaved -> model_results.png, model_comparison.csv, best_model.pkl")
