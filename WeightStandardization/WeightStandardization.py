"""
ELI5 + PM-FRIENDLY GUIDE (IN THE CODE ITSELF)
---------------------------------------------
This script is a "toy churn model" that teaches the MOST IMPORTANT parts
of real ML projects, using simple fake data so you can focus on concepts.

You are a PM working with Data Scientists, so this script tries to answer:
- What is each artifact / chart supposed to show?
- Why does it matter for real business decisions?
- What questions should a PM ask when seeing these outputs?

IMPORTANT CONCEPT:
A deployed ML model is NOT just "the algorithm".
It is a package of:
  (1) preprocessing rules (like scaling)
  (2) the model itself
  (3) a threshold decision for turning probabilities into actions
  (4) evaluation and monitoring logic

This script includes all of that, but with toy data.

WHAT THE TOY DATA MEANS
-----------------------
We create 2 features per customer:
1) big_feature: large magnitude (think: monthly spend, usage minutes, data volume)
2) rand_feature: small magnitude (mostly noise)

We create churn labels mostly based on big_feature,
so the model SHOULD learn:
- big_feature matters
- rand_feature matters little

FILES CREATED
-------------
models/toy_churn_pipeline.joblib  -> saved pipeline (preprocessing + model)
models/threshold.json             -> saved threshold decision

PLOTS YOU WILL SEE
------------------
1) ROC Curve
   - Shows ranking quality (can we separate churners from non-churners?)
   - Good for "overall model ability" conversations

2) Precision-Recall Curve
   - Shows churn-capture ability when churn is rare
   - Often more relevant than ROC in churn use cases

3) Confusion Matrix (at chosen threshold)
   - Shows actual business errors:
     False alarms vs missed churners

4) Calibration Plot
   - Shows whether probabilities can be trusted
   - Example: If model says 70% churn, do ~70% actually churn?
   - Critical for interventions, budgeting, and stakeholder trust

5) Threshold Tradeoffs Charts
   - Shows how changing threshold changes:
     precision, recall, false positive rate, and workload (# flagged)
   - This is literally the PM/business decision layer

6) Weights Plot (coefficients)
   - Shows which features the model is leaning on
   - Helps interpretability and "does this make sense?" checks
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve
import joblib


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "toy_churn_pipeline.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"


def make_toy_churn_data(seed: int = 42):
    """
    ELI5: Make fake customer records.

    big_feature:
      - Large numbers (600–2400)
      - Pretend this is the REAL driver of churn

    rand_feature:
      - Tiny numbers (1–3)
      - Pretend this is mostly noise

    churn label:
      - We generate churn so it mostly follows big_feature
      - This allows us to validate: does the model "learn the right thing"?

    PM WHY THIS MATTERS:
    In real life, you want features that reflect meaningful customer behavior.
    If a model learns from nonsense inputs, it won't generalize to the real world.
    """
    rng = np.random.default_rng(seed)
    n = 220

    big_feature = rng.integers(600, 2401, size=n)
    rand_feature = rng.integers(1, 4, size=n)

    noise = rng.normal(0, 0.35, size=n)

    # This makes churn probability rise as big_feature rises (for demonstration).
    centered = (big_feature - 1500) / 300
    logits = 1.2 * centered + 0.05 * (rand_feature - 2) + noise
    prob = 1 / (1 + np.exp(-logits))

    churn = (prob > 0.55).astype(int)

    X = np.column_stack([big_feature.astype(float), rand_feature.astype(float)])
    y = churn.astype(int)
    return X, y


def pick_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    ELI5: A model gives probabilities (like 0.73).
    But the business usually needs a Yes/No decision.

    Threshold = the cutoff:
      - If prob >= threshold -> predict churn (1)
      - else -> predict stay (0)

    We select the threshold that produces the best F1 score.

    PM WHY THIS MATTERS:
    The threshold is NOT a "math detail".
    It defines:
    - How many customers you'll contact
    - How many false alarms you'll trigger
    - How many churners you'll miss
    In other words: COST, WORKLOAD, and RETENTION IMPACT.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    ROC CURVE: WHAT IT SHOWS
    - X-axis: False Positive Rate (how often we false-alarm)
    - Y-axis: True Positive Rate (how many churners we catch)

    ROC AUC: a single number summarizing this.
    - 0.5 = random guessing
    - 1.0 = perfect ranking

    PM WHY IT MATTERS:
    ROC is a good "overall quality" ranking metric.
    It answers: "Can this model generally separate churners from non-churners?"
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve (Ranking quality)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    PRECISION-RECALL CURVE: WHAT IT SHOWS
    - Precision: when we say churn, how often are we correct?
    - Recall: how many churners did we catch?

    PM WHY IT MATTERS:
    In churn, positives (churners) are often a minority.
    PR curves usually reflect business reality better than ROC.

    If retention outreach is expensive,
    you care a lot about precision (avoid wasting money).
    If churn is very painful,
    you care a lot about recall (catch as many as possible).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve (Churn capture tradeoff)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()


def plot_confusion_matrix(cm: np.ndarray, threshold: float) -> None:
    """
    CONFUSION MATRIX: WHAT IT SHOWS (AT ONE THRESHOLD)

    It is the most "business-readable" evaluation artifact:

    TN: stayed, predicted stayed (good)
    FP: stayed, predicted churn (false alarm -> wasted outreach)
    FN: churned, predicted stayed (missed churner -> lost customer)
    TP: churned, predicted churn (good catch)

    PM WHY IT MATTERS:
    FP and FN are where money/loss lives.
    Every churn project ends up being:
    - Are we ok with more false alarms to catch more churners?
    - Or do we want fewer false alarms even if we miss some churners?
    """
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Stay (0)", "Churn (1)"])
    plt.yticks([0, 1], ["Stay (0)", "Churn (1)"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> None:
    """
    CALIBRATION PLOT: WHAT IT SHOWS
    We bucket predictions by probability.
    Then we compare:
      - average predicted probability in the bucket
      - actual churn rate in the bucket

    Perfect calibration is the diagonal line.

    PM WHY IT MATTERS (THIS IS HUGE):
    Many business decisions rely on probability meaning something real:
      - budget: "we expect 30% of these to churn"
      - intervention intensity: "high risk gets a stronger offer"
      - KPI forecasting: "how many churners next month?"

    If calibration is bad, probabilities are misleading.
    The model might still rank well (good ROC),
    but its probability values won't be trustworthy.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model calibration")
    plt.title("Calibration Plot (Can we trust probabilities?)")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Actual fraction of churners")
    plt.legend()
    plt.tight_layout()


def threshold_tradeoffs(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None) -> None:
    """
    THRESHOLD TRADEOFFS: WHAT THIS SHOWS
    This is your PM dashboard.

    We try different thresholds and show:
    - Precision: "how accurate are our churn flags?"
    - Recall: "how many churners did we catch?"
    - False Positive Rate: "how often do we false-alarm?"
    - # flagged: "how many customers do we target (workload/cost)?"

    PM WHY IT MATTERS:
    Choosing the threshold is a business decision:
    - Limited team capacity? Raise threshold to flag fewer people.
    - High cost of churn? Lower threshold to catch more churners.
    - Expensive retention offers? Keep precision high (raise threshold).
    """
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        flagged = (tp + fp)

        results.append((t, precision, recall, fpr, flagged))

    print("\n=== Threshold tradeoffs (business slider) ===")
    print("threshold | precision | recall | false_pos_rate | # flagged")
    print("-" * 62)
    for t, p, r, fpr, flagged in results:
        print(f"{t:8.2f} | {p:9.3f} | {r:6.3f} | {fpr:13.3f} | {flagged:9d}")

    # Visual: how metrics change as you move the threshold
    ts = np.array([row[0] for row in results])
    precisions = np.array([row[1] for row in results])
    recalls = np.array([row[2] for row in results])
    fprs = np.array([row[3] for row in results])
    flaggeds = np.array([row[4] for row in results])

    plt.figure()
    plt.plot(ts, precisions, marker="o", label="Precision")
    plt.plot(ts, recalls, marker="o", label="Recall")
    plt.plot(ts, fprs, marker="o", label="False Positive Rate")
    plt.title("Threshold Tradeoffs (Business decision chart)")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(ts, flaggeds, marker="o")
    plt.title("Workload vs Threshold (# Customers Flagged)")
    plt.xlabel("Threshold")
    plt.ylabel("# Flagged (TP + FP)")
    plt.tight_layout()


def plot_weights(weights: np.ndarray, bias: float) -> None:
    """
    WEIGHTS/Bias: WHAT THIS SHOWS
    Logistic Regression learns:
      score = (w1 * feature1) + (w2 * feature2) + bias

    Because we are scaling features, weights are more comparable.

    PM WHY IT MATTERS:
    This is a simple interpretability check:
    - Does the model lean on the feature we EXPECT to matter?
    - Are there features with near-zero weight that can be removed?
    - Do weights "make sense" or do we suspect data leakage / bad features?
    """
    labels = ["Big feature weight", "Rand feature weight"]

    plt.figure()
    plt.bar(labels, weights)
    plt.title(f"Model Weights (Bias={bias:.3f})")
    plt.ylabel("Weight value")
    plt.tight_layout()


def main():
    # 1) Make toy data
    X, y = make_toy_churn_data(seed=42)

    # PM note: stratify keeps churn % similar across train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2) Build a pipeline (best practice)
    # ELI5:
    # Pipeline ensures we ALWAYS apply the same preprocessing during training and prediction.
    # This is how you avoid production mistakes like "forgot to scale new data."
    pipe = Pipeline(steps=[
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    # 3) Train
    pipe.fit(X_train, y_train)

    # 4) Predict probabilities on test data
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Model-quality metrics (ranking + churn-focused)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Choose a threshold (business decision)
    best_t = pick_threshold_for_f1(y_test, y_prob)
    y_pred = (y_prob >= best_t).astype(int)

    print("\n=== Evaluation (Toy Churn) ===")
    print(f"ROC AUC: {roc_auc:.4f}   (ranking quality)")
    print(f"PR  AUC: {pr_auc:.4f}   (churn-focused quality)")
    print(f"Chosen threshold: {best_t:.2f} (turn probabilities into actions)\n")

    print("Classification report (at chosen threshold):")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # 5) Plots (each shows something different + important)
    plot_roc(y_test, y_prob)
    plot_pr(y_test, y_prob)
    plot_confusion_matrix(cm, best_t)

    # Calibration: can we trust probabilities?
    plot_calibration(y_test, y_prob, bins=10)

    # Threshold tradeoffs: the PM decision layer
    threshold_tradeoffs(y_test, y_prob, thresholds=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])

    # Weights + bias: interpretability sanity check
    model = pipe.named_steps["model"]
    weights = model.coef_[0]
    bias = float(model.intercept_[0])
    plot_weights(weights, bias)

    # 6) Save model + threshold (what deployment would use)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best_t}, f, indent=2)

    print("\nSaved artifacts for 'deployment':")
    print(f"- Pipeline model: {MODEL_PATH}")
    print(f"- Threshold:      {THRESHOLD_PATH}")

    # 7) Load and predict new customers (prove we can reproduce behavior)
    loaded_pipe = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        loaded_threshold = float(json.load(f)["threshold"])

    new_customers = np.array([
        [1700, 2],
        [1800, 1],
        [900,  3],
        [2350, 1],
    ], dtype=float)

    new_prob = loaded_pipe.predict_proba(new_customers)[:, 1]
    new_pred = (new_prob >= loaded_threshold).astype(int)

    print("\nNew-customer predictions (loaded model):")
    for i in range(len(new_customers)):
        print(
            f"Customer {i+1} features={new_customers[i].tolist()} "
            f"-> churn_prob={new_prob[i]:.3f} churn_pred={int(new_pred[i])}"
        )

    # PyCharm: show the plot windows
    plt.show()


if __name__ == "__main__":
    main()
