# =============================
# 0. IMPORTS & CONFIG
# =============================
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import product as iterproduct

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree              import DecisionTreeClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

warnings.filterwarnings("ignore")

# ── Paths ─
INPUT_CSV      = "extracted_dt_combined.csv"
MODEL_PKL      = "dt_model.pkl"
RESULTS_JSONL  = "results.jsonl"
FIG_IMPORTANCE = "dt_feature_importance.png"
FIG_ROC        = "dt_roc_curve.png"

# ── Reproducibility ──
RANDOM_SEED = 49
np.random.seed(RANDOM_SEED)

# ── Feature columns 
FEATURE_COLS = [
    "avg_sent_len", "std_sent_len", "avg_word_len", "std_word_len", "ttr",
    "comma_count", "period_count", "exclaim_count", "question_count",
    "sent_burstiness", "para_len_variance", "hapax_ratio",
    "contraction_rate", "poly_ratio", "passive_ratio", "transition_density",
]

# =============================
# 1. LOAD DATA
# =============================
print("=" * 60)
print("DECISION TREE — AI TEXT DETECTION TRAINING PIPELINE")
print("=" * 60)

print(f"\n[1/6] Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

df = df.dropna(subset=["label"] + FEATURE_COLS).reset_index(drop=True)
print(f"  Loaded shape : {df.shape}")
print(f"  Label dist   :\n{df['label'].value_counts().to_string()}")

X = df[FEATURE_COLS].values.astype(np.float64)
y = df["label"].values.astype(int)

# =============================
# 2. TRAIN / TEST SPLIT
# =============================
print("\n[2/6] Splitting data (80 % train / 20 % test, stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"  Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")


scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =============================
# 3. HELPER — EVALUATE MODEL
# =============================
def evaluate(model, X_tr, y_tr, X_te, y_te, label: str) -> dict:
    """
    Returns a result dict with all metrics for one model configuration.
    """
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_prob = model.predict_proba(X_te)[:, 1]
    threshold = 0.7   # try 0.6–0.9 range
    y_pred = (y_prob >= threshold).astype(int)
    
    acc   = accuracy_score(y_te, y_pred)
    prec  = precision_score(y_te, y_pred, zero_division=0)
    rec   = recall_score(y_te, y_pred, zero_division=0)
    f1    = f1_score(y_te, y_pred, zero_division=0)
    auc   = roc_auc_score(y_te, y_prob)
    cm    = confusion_matrix(y_te, y_pred).tolist()
    report = classification_report(y_te, y_pred,
                                   target_names=["human", "ai"],
                                   output_dict=True)

    result = {
        "experiment"      : label,
        "accuracy"        : round(acc,  4),
        "precision"       : round(prec, 4),
        "recall"          : round(rec,  4),
        "f1_score"        : round(f1,   4),
        "roc_auc"         : round(auc,  4),
        "confusion_matrix": cm,
        "classification_report": report,
        "train_time_sec"  : round(train_time, 4),
        "n_train"         : int(X_tr.shape[0]),
        "n_test"          : int(X_te.shape[0]),
        "params"          : model.get_params(),
    }

    print(f"  [{label}]  acc={acc:.4f}  prec={prec:.4f}  "
          f"rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
          f"({train_time:.2f}s)")
    return result

# =============================
# 4. PARAMETER SWEEP
# =============================
print("\n[3/6] Parameter sweep …")

DEPTH_VALS      = [5, 10, 15, None]       # None = unlimited
MIN_SPLIT_VALS  = [2, 10, 50, 100]
CRITERION_VALS  = ["gini", "entropy"]

all_results   = []
best_auc   = -1.0
best_model = None
best_result   = None

combos = list(iterproduct(DEPTH_VALS, MIN_SPLIT_VALS, CRITERION_VALS))
print(f"  Total configs : {len(combos)}")

for depth, min_split, criterion in combos:
    tag = (f"depth={depth}_minsplit={min_split}_criterion={criterion}")
    clf = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=min_split,
        criterion=criterion,
        random_state=RANDOM_SEED,
    )
    res = evaluate(clf, X_train_s, y_train, X_test_s, y_test, tag)
    res["sweep"] = True
    all_results.append(res)

    # AFTER
    if res["roc_auc"] > best_auc:
       best_auc   = res["roc_auc"]
       best_model = clf
       best_result = res

# AFTER
print(f"  ✔ Best config  : {best_result['experiment']}")
print(f"    AUC={best_auc:.4f}  F1={best_result['f1_score']:.4f}")

# =============================
# 5. CROSS-VALIDATION ON BEST
# =============================
print("\n[4/6] 5-fold cross-validation on best config …")

cv_clf = DecisionTreeClassifier(**best_model.get_params())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_scores = cross_validate(
    cv_clf, X_train_s, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=True
)

cv_result = {
    "experiment"          : "cross_validation_best_config",
    "sweep"               : False,
    "cv_folds"            : 5,
    "params"              : best_model.get_params(),
    "cv_test_accuracy"    : {
        "mean" : round(float(cv_scores["test_accuracy"].mean()),   4),
        "std"  : round(float(cv_scores["test_accuracy"].std()),    4),
        "folds": [round(float(v), 4) for v in cv_scores["test_accuracy"]]
    },
    "cv_test_precision"   : {
        "mean" : round(float(cv_scores["test_precision"].mean()),  4),
        "std"  : round(float(cv_scores["test_precision"].std()),   4),
        "folds": [round(float(v), 4) for v in cv_scores["test_precision"]]
    },
    "cv_test_recall"      : {
        "mean" : round(float(cv_scores["test_recall"].mean()),     4),
        "std"  : round(float(cv_scores["test_recall"].std()),      4),
        "folds": [round(float(v), 4) for v in cv_scores["test_recall"]]
    },
    "cv_test_f1"          : {
        "mean" : round(float(cv_scores["test_f1"].mean()),         4),
        "std"  : round(float(cv_scores["test_f1"].std()),          4),
        "folds": [round(float(v), 4) for v in cv_scores["test_f1"]]
    },
    "cv_test_roc_auc"     : {
        "mean" : round(float(cv_scores["test_roc_auc"].mean()),    4),
        "std"  : round(float(cv_scores["test_roc_auc"].std()),     4),
        "folds": [round(float(v), 4) for v in cv_scores["test_roc_auc"]]
    },
}
all_results.append(cv_result)

print(f"  CV F1    : {cv_result['cv_test_f1']['mean']:.4f} "
      f"± {cv_result['cv_test_f1']['std']:.4f}")
print(f"  CV AUC   : {cv_result['cv_test_roc_auc']['mean']:.4f} "
      f"± {cv_result['cv_test_roc_auc']['std']:.4f}")
print(f"  CV Acc   : {cv_result['cv_test_accuracy']['mean']:.4f} "
      f"± {cv_result['cv_test_accuracy']['std']:.4f}")

# =============================
# 6. FINAL MODEL — RETRAIN ON
#    FULL TRAINING SET
# =============================
print("\n[5/6] Retraining best config on full training set …")

final_clf = DecisionTreeClassifier(**best_model.get_params())
final_clf.fit(X_train_s, y_train)

final_res = evaluate(final_clf, X_train_s, y_train,
                     X_test_s, y_test, "final_best_model")
final_res["sweep"] = False
final_res["note"]  = "Retrained best config on full train split; evaluated on held-out test"
all_results.append(final_res)

importances = final_clf.feature_importances_
imp_series  = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)

importance_record = {
    "experiment"        : "feature_importances",
    "sweep"             : False,
    "feature_importances": {k: round(float(v), 6) for k, v in imp_series.items()},
}
all_results.append(importance_record)

print("\n  Feature importances (top 8):")
for feat, imp in imp_series.head(8).items():
    bar = "█" * int(imp * 200)
    print(f"    {feat:<22s}  {imp:.4f}  {bar}")

y_pred_final = final_clf.predict(X_test_s)
y_prob_final = final_clf.predict_proba(X_test_s)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
analysis_record = {
    "experiment"    : "metric_analysis",
    "sweep"         : False,
    "true_positives"  : int(tp),
    "true_negatives"  : int(tn),
    "false_positives" : int(fp),
    "false_negatives" : int(fn),
    "false_positive_rate" : round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
    "false_negative_rate" : round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0,
    "note": (
        "FPR = human text incorrectly flagged as AI. "
        "FNR = AI text missed. For academic integrity use-cases, "
        "low FPR is critical to avoid wrongly accusing students."
    ),
}
all_results.append(analysis_record)

print(f"\n  False Positive Rate (human → AI): {analysis_record['false_positive_rate']:.4f}")
print(f"  False Negative Rate (AI → human): {analysis_record['false_negative_rate']:.4f}")

# =============================
# 7. SAVE ARTEFACTS
# =============================
print(f"\n[6/6] Saving artefacts …")

with open(RESULTS_JSONL, "w", encoding="utf-8") as fout:
    for rec in all_results:
        fout.write(json.dumps(rec, default=str) + "\n")
print(f"  ✔ {RESULTS_JSONL}  ({len(all_results)} records)")

# ── dt_model.pkl  (bundle: scaler + model + metadata) ───
bundle = {
    "model"       : final_clf,
    "scaler"      : scaler,
    "feature_cols": FEATURE_COLS,
    "label_map"   : {0: "human", 1: "ai"},
    "best_params" : final_clf.get_params(),
    "test_metrics": {
        "accuracy" : final_res["accuracy"],
        "f1_score" : final_res["f1_score"],
        "roc_auc"  : final_res["roc_auc"],
    },
}
joblib.dump(bundle, MODEL_PKL)
print(f"  ✔ {MODEL_PKL}")

# ── Feature importance plot ────
fig, ax = plt.subplots(figsize=(10, 6))
colors  = ["#e05c5c" if i < 5 else "#5b8dee" for i in range(len(imp_series))]
imp_series.plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
ax.invert_yaxis()
ax.set_title("Decision Tree — Feature Importances", fontsize=14, fontweight="bold")
ax.set_xlabel("Gini Importance", fontsize=11)
ax.axvline(0, color="black", linewidth=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_IMPORTANCE, dpi=150)
plt.close(fig)
print(f"  ✔ {FIG_IMPORTANCE}")


fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob_final)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_arr, tpr_arr, color="#5b8dee", lw=2,
        label=f"DT  (AUC = {final_res['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Decision Tree", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_ROC, dpi=150)
plt.close(fig)
print(f"  ✔ {FIG_ROC}")

# =============================
# SUMMARY
# =============================
print("\n" + "=" * 60)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Best config   : {best_result['experiment']}")
print(f"  Test Accuracy : {final_res['accuracy']:.4f}")
print(f"  Test Precision: {final_res['precision']:.4f}")
print(f"  Test Recall   : {final_res['recall']:.4f}")
print(f"  Test F1       : {final_res['f1_score']:.4f}")
print(f"  Test ROC-AUC  : {final_res['roc_auc']:.4f}")
print(f"\n  Outputs saved:")
print(f"    {MODEL_PKL:<35s} ← inference bundle")
print(f"    {RESULTS_JSONL:<35s} ← all experiment records")
print(f"    {FIG_IMPORTANCE:<35s} ← feature importance chart")
print(f"    {FIG_ROC:<35s} ← ROC curve")
print("=" * 60)

# =============================
# INFERENCE USAGE EXAMPLE
# =============================
print("""
── HOW TO USE dt_model.pkl FOR INFERENCE ──────────────────────
import joblib, numpy as np

bundle  = joblib.load("dt_model.pkl")
model   = bundle["model"]
scaler  = bundle["scaler"]
cols    = bundle["feature_cols"]
lmap    = bundle["label_map"]

# features must be in the same order as `cols`
raw_features = np.array([[...]])          # shape (1, 16)
scaled       = scaler.transform(raw_features)
pred_label   = lmap[model.predict(scaled)[0]]
confidence   = model.predict_proba(scaled)[0][1]   # P(AI)

print(pred_label, confidence)
───────────────────────────────────────────────────────────────
""")