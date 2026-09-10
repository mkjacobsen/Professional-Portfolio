"""
examples/credit_risk_demo.py

Full ML governance workflow for credit risk scoring models.

Steps:
1. Generate synthetic credit risk dataset (~2000 samples)
2. Train two models: LogisticRegression and GradientBoostingClassifier
3. Evaluate both, compute SHAP importances, compute fairness slices
4. Register both models in the local registry
5. Compute risk scores for both
6. Generate self-conatined HTML reports
7. Compare the two models side by side
8. Promote the better model to "production", archive the other
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Make sure the ml_governance/ layout is importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_governance"))

from ml_governance import (
    ModelCard,
    ModelEvaluator,
    ModelRegistry,
    PerformanceMetrics,
    ReportGenerator,
    RiskScorer,
    SHAPExplainer,
)

warnings.filterwarnings("ignore")

# Configuration

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
REGISTRY_DIR = Path(__file__).resolve().parent.parent / "output" / "registry"
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)

# Step 1: Generate synthetic credit risk dataset
def generate_credit_dataset(n: int = 2000) -> pd. DataFrame:
    """
    Synthetic credit risk dataset.

    Features

    age                   : 18-70
    income                : 25k-120k USD
    debt_ratio            : 0.05-0.95
    credit_history_length : 0-20 years
    num_late_payments     : 0-15
    employment_status     : "employed", "self_employed", "unemployed"
    loan_amount           : 5k-50k USD

    Target

    default: binary, ~20% positive rate
    """

    print("=" * 60)
    print("Step 1: Generating synthetic credit risk dataset")
    print("=" * 60)

    age = RNG.integers(18, 70, n)
    income = RNG.uniform(25_000, 120_000, n)
    debt_ratio = RNG.uniform(0.05, 0.95, n)
    credit_history_length = RNG.uniform(0, 20, n)
    num_late_payments = RNG.integers(0, 15, n)
    employment_status = RNG.choice(
        ["employed", "self_employed", "unemployed"],
        n,
        p = [0.65, 0.20, 0.15],
    )
    loan_amount = RNG.uniform(5_000, 50_000, n)

    # Logistic default probability: driven by debt ratio, late payments, employment
    log_odds = (
        -2.0
        + 3.5 * debt_ratio
        + 0.15 * num_late_payments
        - 0.008 * (income / 1_000)
        - 0.06 * credit_history_length
        + 0.3 * (employment_status == "unemployed").astype(float)
        + 0.15 * (employment_status == "self_employed").astype(float)
        + 0.005 * (loan_amount / 1_000)
        + RNG.standard_normal(n) * 0.5 #noise
    )
    prob_default = 1.0 / (1.0 + np.exp(-log_odds))
    default = RNG.binomial(1, prob_default, n)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "debt_ratio": debt_ratio,
            "credit_history_length": credit_history_length,
            "num_late_payments": num_late_payments,
            "employment_status": employment_status,
            "loan_amount": loan_amount,
            "default": default,
        }
    )

    # Age groups for fairness slicing
    df["age_groups"] = pd.cut(
        df["age"],
        bins = [17,25,35,45,55,70],
        labels=["18-25","26-35","35-45","46-55","56-70"],
    ).astype(str)

    print(f"   Dataset shape  : {df.shape}")
    print(f"   Default rate   : {df['default'].mean():.1%}")
    print(f"   Age range      : {df['age'].min()}-{df['age'].max()}")
    print()
    return df

# Step 2: Prepare features and train/test split

def prepare_features(df: pd.DataFrame):
    """Encode categoricals, split into train/val sets."""
    le = LabelEncoder()
    df = df.copy()
    df["employment_status_enc"] = le.fit_transform(df["employment_status"])

    feature_cols = [
        "age",
        "income",
        "debt_ratio",
        "credit_history_length",
        "num_late_payments",
        "employment_status_enc",
        "loan_amount",
    ]

    X = df[feature_cols]
    Y = df["default"].values
    X_meta = df[["age_group", "employment_status"]]

    X_train, X_val, Y_train, Y_val, meta_train, meta_val = train_test_split(
        X, Y, X_meta, test_size = 0.25, random_state = RANDOM_STATE, stratify = Y
    )

    return X_train, X_val, Y_train, Y_val, meta_train, meta_val, feature_cols

# Step 3: Train models

def train_models(X_train, Y_train):
    print("=" * 60)
    print("Training models")
    print("=" * 60)

    lr = LogisticRegression(max_iter = 500, C = 0.1, random_state = RANDOM_STATE)
    lr.fit(X_train, Y_train)
    print("   [1/2] Logistic Regression            - trained")

    gb = GradientBoostingClassifier(
        n_estimators = 150, max_depth = 4, learning_rate = 0.05, random_state = RANDOM_STATE
    )
    gb.fit(X_train, Y_train)
    print(".   [2/2] GradientBoostingClassifier    - trained")
    print()
    return lr, gb

# Step 4: Evaluate SHAP, fairness slices

def evaluate_model(
        model, 
        model_name: str,
        model_type_str: str,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        X_meta_val: pd.DataFrame,
        feature_names: list[str],
        version: str,
) -> tuple[ModelCard, object]:
    """Evaluate one model -> return (ModelCard, shap_figure)"""
    print(f"   Evaluating {model_name}...")

    evaluator = ModelEvaluator()

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]

    # Performance
    perf = evaluator.compute_performance(Y_val, y_pred, y_prob)

    # SHAP importances (use 100-row background for speed)
    bg = X_train.sample(min(100, len(X_train)), random_state = RANDOM_STATE)
    explainer = SHAPExplainer(model, bg)
    importances = explainer.computer_importances(X_val)
    shap_fig = explainer.plot_summary(X_val, max_features=7)

    # Fairness slices on age_group
    X_val_with_meta = X_val.copy()
    X_val_with_meta["age_group"] = X_meta_val["age_group"].values
    slices = evaluator.compute_fairness_slices(
        X_val_with_meta, Y_val, y_pred, y_prob, ["age_group"]
    )

    card = ModelCard(
        model_id = "", # assigned by registry
        model_name = model_name,
        version = version,
        description = (
            f"{model_type_str} train on synthetic consumer credit data. "
            "Predicts probability of load default (binary classification). "
        ),
        model_type = model_type_str,
        training_date = "2024-06-01",
        training_dataset_size=len(X_train),
        feature_names = feature_names,
        performance = perf,
        fairness_slices = slices,
        feature_importances = importances,
        intended_use = (
            "FICTIONAL - Approve or decline personal load applications for consumers "
            "with annual income between $25k-$120k."
        ),
        limitations = (
            "Not validated for commercial or small-business lending. "
            "Requires re-validation if applicant population shifts by >10%. "
        ),
    )

    return card, shap_fig

# Main Workflow

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data
    df = generate_credit_dataset(n=2000)
    X_train, X_val, y_train, y_val, meta_train, meta_val, feature_cols = (
        prepare_features(df)
    )

    # 2. Train
    lr_model, gb_model = train_models(X_train, y_train)

    # 3. Evaluate
    print("=" * 60)
    print("Evaluating models + SHAP + fairness slices")
    print("=" * 60)

    lr_card, lr_shap_fig = evaluate_model(
        lr_model, "CreditScorer-LR", "LogisticRegression",
        X_train, X_val, y_train, y_val, meta_val, feature_cols, "v1.0",
    )
    gb_card, gb_shap_fig = evaluate_model(
        gb_model, "CreditScorer-GB", "GradientBoostingClassifier",
        X_train, X_val, y_train, y_val, meta_val, feature_cols, "v1.0",
    )
    print()

    # 4. Register
    print("=" * 60)
    print("Registering Models")
    print("=" * 60)

    registry = ModelRegistry (REGISTRY_DIR)
    lr_id = registry.register(
        lr_model, "CreditScorer-IR", "v1.0", lr_card,
        tags = {"algorithm": "logistie_regression", "team": "model_risk"},
    )
    gb_id = registry.register(
        gb_model, "CreditScorer-GB", "v1.0", gb_card,
        tags = {"algorithm": "gradient_boosting","team": "model_risk"},
    )
    print(f" LR model ID: {lr_id}")
    print(f" GB model ID: {gb_id}")
    print()

    # 5. Risk scoring
    print("=" * 60)
    print( "Risk scoring")
    print("=" * 60)

    scorer = RiskScorer()
    lr_risk, lr_factors = scorer.score(lr_card)
    lr_card.risk_score = lr_risk
    lr_card.risk_factors = lr_factors
    gb_risk, gb_factors = scorer.score(gb_card)
    gb_card.risk_score = gb_risk
    gb_card.risk_factors = gb_factors
    print(f"LR risk score : {lr_risk:.1f} / 100 {scorer.risk_label(lr_risk)})")
    print(f"GB risk score : {gb_risk:.1f} / 100 {scorer.risk_label(gb_risk)})")