# MI Governance Framework

A lightweight, production-ready ML model governance framework for regulated industries. Register trained models, generate structured model cards with performance metrics, fairness analysis, and SHAP-based explainability, compute structured risk scores, and produce PDE-ready HTML reports. Designed around credit risk scozing models - a domain where governance, auditability, and fairness monitoring are regulatory requirements, not afterthoughts.

## 1. Project Overview

Model governance in regulated industries (banking, insurance, healthcare) means more than tracking accuracy metrics. It means:
- **Auditability**: every model version, its training data characteristics, and its approval chain are recorded
- **Fairness monitoring**: identifying demographic disparities before models reach production
- **Risk quantification**: translating model behavior into structured risk scores that non-technical stakeholders can act on
- **Explainability**: providing regulators and model risk officers with feature-level transparency via SHAP
- **Lifecycle management**: staging -> production -> archived transitions with clear ownership

This framework implements all of these in ~600 lines of Python, backed by SQLite and Jinja2 - deployable without a data platform. 

## 2. Architecture

```
ModelEvaluator              SHAPExplainer
     \                            /
      \                          /
       -----> ModelCard <--------         <----- RiskScorer
                   |
        ModelRegistry (SQLite)
                   |
        ReportGenerator (Jinja2 -> HTML)
```

**Component responsibilities:**
| Component | Role |
| --------- | ---- |
| `ModelRegistry` | SQlite-backed store; pickle models, JSON cards, lifecycle status |
| `ModelCard` | Dataclass with performance, fairness slices, feature importances, risk score |
| `ModelEvaluator` | Computes "PerformanceMetrics', fairness slices, KS curve |
| `SHAPExplainer` | Wraps 'shap.Explainer'; computes rankedfeature importances |
| `RiskScorer` | Scores 4 risk dimensions (0-25 cach); total 0-100 |
| `ReportGenerator` | Renders self-contained HTML report via Jinja2 |

## 3. Quickstart

```bash
pip install -e ".(dev]"
python examples/credit_risk_demo.py
```

Reports are written to `./output`. Open `model_report_<id>.html` in any browser or print to PDF.

## 4. Risk Scoring Methodology
Risk score 0-100 (higher = riskier), composed of four equal-weight dimensions (0-25 each):

### Performance Risk (0-25)

Penalizes models with weak discriminative power:

- Gini coefficient < 0.3 - maximum performance risk (Gini is the standard credit risk discriminator: `2 × AUC - 1`)
- Scaled linearly between 0.3 and 0.5 for partial penalty
- Additional penalty for F1 < 0.5

**Threshold rationale**: A Gini below 0.3 is widely considered unacceptable for credit decisioning in Basel II/III frameworks.

## Fairness Risk (0-25)

Based on the **disparate impact ratio** (also called the 4/5th rule):

- `disparate_impact = slice_approval_rate / overall_approval_rate`
- Slices with `disparate_impact < 0.8' are flagged as violations
- Risk scales with the fraction of slices violating the threshold

**Threshold rationale**: The 0.8 threshold originates from the EEOC Uniform Guidelines (1978) and is the most widely cited fairness criterion in US regulatory guidance, including CFPB fair lending examinations.

### Data Risk (0-25)

Penalizes insufficient training data:

- Training sets < 1,000 samples - full data risk penalty
- Scaled proportionally between 1,000 and 10,000 samples

**Threshold rationale**: Below 1,000 samples, variance in credit scorecard estimates becomes statistically unreliable for most feature sets.

## Stability Risk (0-25)

Based on the **Kolmogorov-Smirnov statistic** (KS), the standard credit model stability metric:

- KS = max separation between the cumulative distribution of predicted scores for defaulters vs. non-defaulters
- KS < 02 -> full stability risk
- KS 0.2-0.4 -> partial penalty

**Threshold rationale**: KS< 0.2 is the industry-standard "reject" threshold for credit scorecards (per standard model validation practice).

**Risk Labels:**
| Score | Label |
| ----- | ----- |
| 0-25 | Low |
| 26-50 | Medium |
| 51-75 | High |
| 76-100 | Critical | 

## 5. Fairness Analysis

The fairness module computes per-slice metrics for specified sensitive or proxy features (e.g., `age_group`, `employment_status`). For each unique value within a feature:

- **Approval rate**: fraction of samples in this slice where the model predicts approva1 (`y_pred = 0`, i.e., non-default)
- **False Positive Rate (FPR)**: traction of actual non-defaults incorrectly predicted as defaults
- **False Negative Rate (FNR)**: fraction of actual defaults missed
- **Disparate Impact**: `slice_approval_rate / overall_approval_rate`

Slices with fewer than 30 samples are excluded to avoid noisy estimates. 

The HTML report highlights any slice with `disparate_impact < 0.8' in red, providing single-glance fairness audit surface.

## 6. API Reference

### `ModelRegistry(registry_path)`

```python
registry = ModelRegistry("./my_registry")
model_id = registry.register(model, "CreditScorer", "v1.0", card, tags={"env":"prod"})
model, card = registry.get(model_id)
entries = registry.list_models(status="production")
registry.update_status(model_id, "archived")
delta = registry.compare(model_id_a, model_id_b)
```

### `ModelCard`

```python
card = ModelCard(
    model_id = "...", model_name = "CreditScorer", version = "v1.0",
    description = "...", model_type = "GradientBoostingClassifier",
    training_date = "2024-01-15", training_dataset_size = 5000,
    feature_names = [...], performance = metrics,
    intended_uses = "...", limitations = "..."
)
card_json = card.to_json()
card = ModelCard.from_json(card_json)
```

### `RiskScorer`

```python
scorer = RiskScorer()
total_score, risk_factors = scorer.score(card)
label = scorer.risk_label(total_score)
```

### `ModelEvaluator`

```python
evaluator = ModelEvaluator()
metrics = evaluator.compute_performance(y_true, y_pred, y_prob)
slices = evaluator.compute_fairness_slices(X, y_true, y_pred, y_prob, ["age_group"])
ks_curve = evaluator.compute_ks_curve(y_true, y_prob)
```

### `SHAPExplainer`

```python
explainer = SHAPExplainer(model, X_background)
importances = explainer.compute_importances(X_val)
shap_vals = explainer.compute_shap_values(X_val)
fig = explainer.plot_summary(X_val, max_features=10)
```
### `ReportGenerator`

```python
reporter = ReportGenerator("./templates")
reporter.generate(card, "./output/report.html")
```

## 7. Running Tests

```bash
pip install -o ".[dev]"
pytest tests/ -v
```

All tests are self-contained and use synthetic data - no external dependencies or network access required.