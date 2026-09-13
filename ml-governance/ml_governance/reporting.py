"""
reporting.py - HTML model card report generator
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

from .model_card import ModelCard
from .risk import RiskScorer

if TYPE_CHECKING:
    import matplotlib.figure

class ReportGenerator:
    """
    Render a self-contained HTML model card report using a Jinja2 template.

    All assets (CSS, chart images) are inlined - the output is a single
    portable HTML file that can be printed to PDF from any browser.
    """

    def __init__(self, template_dir: str | Path):
        self._template_dir = Path(template_dir)
        self._env = Environment(
            loader = FileSystemLoader(str(self._template_dir)),
            autoescape = True,
        )
        self._scorer = RiskScorer()

    # Public API

    def generate(
        self, 
        card: ModelCard,
        output_path: str | Path,
        shap_figure: "matplotlib.figure.Figure | None" = None,
    ) -> None:
        """
        Render the report and write it to *output_path*

        Parameters

        card:
            Populated :class:`~ml_governance.ModelCard`
        output_path:
            Destination HTML file path
        shap_figure:
            Optional matplotlib figure to embed as an inline SHAP chart.
            If ommited, no chart is rendered
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        risk_label = self._scorer.risk_label(card.risk_score or 0.0)
        shap_image_b64 = self._fig_to_base64(shap_figure) if shap_figure else None

        template = self._env.get_template("model_report.html")
        html = template.render(
            card = card,
            risk_label = risk_label,
            shap_image_b64 = shap_image_b64,
            generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            disparate_impact_threshold = RiskScorer.DISPARATE_IMPACT_THRESHOLD,
        )

        output_path.write_text(html, encoding="utf-8")

    # Internal Helpers

    def _fig_to_base64(self, fig: "matplotlib.figure.Figure") -> str:
        """Encode a matplotlibfigure as an inline base64 PNG data URL."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        return f"data:image/png;base64,{b64}"

    