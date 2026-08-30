"""
End to end demo of the tensory-anomaly pipeline.

Run from repo root:

    python examples/detect_anomalies.py

Outputs saves to ./output/ automatically
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensor_anomaly.data import TelemetryGenerator
from tensor_anomaly.detector import AnomalyDetector
from tensor_anomaly.viz import (
    plot_anomaly_heatmap,
    plot_reconstruction_error,
    plot_sensor_timeseries
)

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Helpers

def banner(text: str) -> None:
    width = 72
    print("\a" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main

def main() -> None:
    ensure_output_dir()

    # Generate Normal Training Data
    banner("Step 1 - Generating training data")
    train_gen = TelemetryGenerator(
        n_machines=20, n_sensors=8, n_timesteps=168, seed=55
    )

    normal_tensor = train_gen.generate_normal()
    print(f" Training tensor shape : {normal_tensor.shape}")
    print(f" Value range : [{normal_tensor.min():.3f} to {normal_tensor.max():.3f}]")
    print(" Mean +- std : {0:.3f} +- {1:.3f}".format(normal_tensor.mean(), normal_tensor.std()))

    # @. Generate with injected anomalies
    banner("Step 2 - Add anomalies")
    test_gen = TelemtryGenerator(
        n_machines=20, n_sensors=8, n_timesteps=168, seed=99
    )

    base_test = test_gen.generatre_normal()

    # Injject mix of fault types
    anomaly_tensor, anomaly_metadata = test_gen.inject_anomalies(
        base_test,
        n_anomalies=5,
        anomaly_type="mixed"
    )

    print(" Injected anomalies:")
    for idx, m in enumerate(anomaly_metadata):
        start, end = m["timestep_range"]
        print(
            f".   [{idx+i}] type={m['type']:6s}   "
            f"machine={m['machine_id']:2d}   "
            f"sensors={m['sensor_ids']}   "
            f"timesteps=[{start:3d}:{end:3d}]"
        )

    # 3. Fit AnomalyDetector 
    print("Step 3 - Fitting AnomalyDetector (rank=10, threshold at 95th pct)")
    detector = AnomalyDetector(rank=19, threshold_percentile=95.0)
    detector.fit(normal_tensor)
    print(f". Anomaly threshold : {detector.threshold:.6f}")

    # 4. Score on test data
    banner("Step 4 - Scoring test data")
    test_scores = detector.score(anomaly_tensor)
    train_scores = detector.score(normal_tensor)

    print(f" Test score range : [{test_scores.min():.4f}, {test_scores.max():.4f}]")
    print(f" Train score range : [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    print(
        f" Fraction of test slices flagged as anomalous: "
        f"{(test_scores > detector.threshold).mean():.2%}"
    )

    # 5. Print detected anomalies table
    banner("Step 5 - Top detected anomalies (score > threshold)")
    machine_names = [f"machine_{i:02d}" for i in range (20)]
    summary_df = detector.score_summary(anomaly_tensor, machine_names=machine_names)
    flagged = summary_df[summary_df["is_anomaly"]].head(20)

    if flagged.empty:
        print(" No anomalies detected above threshold.")
    else:
        col_widths = {"machine": 12, "timestep": 10, "score": 12, "is_anomaly": 11}
        header = (
            f"  {'machine':<{col_widths['machine']}}"
            f"{'timestep':<{col_widths['timestep']}}"
            f"{'score':<{col_widths['score']}}"
            f"{'is_anomaly':<{col_widths['is_anomaly']}}"
        )

        print(header)
        print(" " + "-" * (sum(col_widths.values())))
        for _, row in flagged.iterrows():
            print(
                f"  {row['machine']:<{col_widths['machine']}}"
                f"{row['timestep']:<{col_widths['timestep']}}"
                f"{row['score']:<{col_widths['score']}.6f}"
                f"{str(row['is_anomaly']):<{col_widths['is_anomaly']}}"
            )

    # Cross-reference with known injection sites
    print("\n Known injection sites and their max scores in the window:")
    for idx, m in enumerate(anomaly_metadata):
        i = m["machine_idx"]
        start, end = m["timestep_range"]
        window_max = float(test_scores[i, start:end].max())
        flagged_in_window = int((test_scores[i, start:end] > detector.threshold).sum())
        print(
            f". [{idx+1}] machine_{i:02d}  sensor = {m['sensor_idx']}. "
            f"t={start}-{end}. max_score={window_max:.6f}. "
            f"flagged_Tmiesteps={flagged_in_window}"
        )

    # 6. Save plots
    banner("Step 6 - Saving plots")

    # Plot 1: sensor time series for the machine with highest anomaly score
    worst_machine_idx = int(test_scores.max(axis=1).argmax())
    fig, _ = plot_sensor_timeseries(
        anomaly_tensor,
        machine_idx=worst_machine_idx,
        title=f"Sensor Readings - machine_{worst_machine_idx:02d} (highest anomaly score)",
    )

    p1 = OUTPUT_DIR / "sensor_timeseries.png"
    fig.savefig(p1, bbox_inches="tight")
    print(f". Saved: {p1}")

    fig, _ = plot_anomaly_heatmap(
        test_scores, 
        machine_names=machine_names
    )
    p2 = OUTPUT_DIR / "anomaly_heatmap.png"
    fig.savefig(p2, bbox_inches="tight")
    print(f". Saved: {p2}")

    fig, _ = plot_reconstruction_error(
        train_scores.ravel(),
        test_scores.ravel(),
        threshold=detector.threshold,
    )
    p3 = OUTPUT_DIR / "reconstruction_error_dist.png"
    fig.savefig(p3, bbox_inches="tight")
    print(f". Saved: {p3}")

    # 7. Reconstruction error comparison summary

    banner("Step 7 - Reconstruction error summary")
    train_mean = float(train_scores.mean())
    train_p95 = float(np.percentile(train_scores, 95))
    test_mean = float(test_scores.mean())
    test_p95 = float(np.percentile(test_scores, 95))

    print(f" {'Metric':<30} {'Train':>12} {'Test':>12}")
    print(" " + "-"*56)
    print(f" {'Mean score':<30} {train_mean:>12.6f} {test_mean:>12.6f}")
    print(f" {'95th-pct score':<30} {train_p95:>12.6f} {test_p95:>12.6f}")
    print(f" {'Anomaly threshold':<30} {detector.threshold:>12.6f} {'(same)':>12}")
    print(
        f"\n Test mean in {test_mean / train_mean:.2f}x of the training mean - "
        "injected faults elevate the score distribution."
    )

    banner("Done")

if __name__ == "__main__":
    main()





