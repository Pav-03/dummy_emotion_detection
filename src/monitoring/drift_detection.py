import pandas as pd
import json
import os
from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset


def extract_text_features(df: pd.DataFrame, text_column: str = "content") -> pd.DataFrame:
    """
    Extract numerical features from text for drift detection.

    WHY THESE FEATURES?
    Each feature catches a different type of drift:

    text_length:     Catches input format changes (tweets vs essays)
    word_count:      Catches complexity changes (simple vs detailed)
    avg_word_length: Catches vocabulary changes (slang vs formal)
    char_count:      Catches encoding or format issues
    has_question:    Catches intent changes (asking vs stating)
    exclamation_count: Catches emotion intensity changes
    unique_word_ratio: Catches repetition patterns
    """
    features = pd.DataFrame()

    # Basic text stats
    features["text_length"] = df[text_column].astype(str).str.len()
    features["word_count"] = df[text_column].astype(str).str.split().str.len()
    features["avg_word_length"] = (
        features["text_length"] / features["word_count"].replace(0, 1)
    )
    features["char_count"] = df[text_column].astype(str).str.replace(" ", "").str.len()

    # Content pattern features
    features["has_question"] = df[text_column].astype(str).str.contains(r"\?").astype(int)
    features["exclamation_count"] = df[text_column].astype(str).str.count(r"!")
    features["unique_word_ratio"] = df[text_column].astype(str).apply(
        lambda x: len(set(x.split())) / max(len(x.split()), 1)
    )

    # If sentiment/prediction column exists, include it
    if "sentiment" in df.columns:
        features["prediction"] = df["sentiment"]
    if "prediction" in df.columns:
        features["prediction"] = df["prediction"]
    if "confidence" in df.columns:
        features["confidence"] = df["confidence"]

    return features


def run_data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str = "drift_report",
    output_dir: str = "reports/drift"
) -> dict:
    """
    Run Evidently data drift report.

    WHAT THIS DOES:
    1. Compares every column's distribution between reference and current
    2. Uses statistical tests (KS test for numerical, chi-squared for categorical)
    3. Returns per-column drift scores + overall dataset drift result
    """
    os.makedirs(output_dir, exist_ok=True)

    # Run the report using DataDriftPreset
    report = Report([
        DataDriftPreset(),
    ])

    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    # Save HTML report (visual — open in browser)
    html_path = os.path.join(output_dir, f"{report_name}.html")
    snapshot.save_html(html_path)
    print(f"  HTML report saved: {html_path}")

    # Extract results as dict
    result = snapshot.dict()

    # Parse drift results
    drift_summary = parse_drift_results(result)
    drift_summary["report_name"] = report_name
    drift_summary["timestamp"] = datetime.now().isoformat()
    drift_summary["reference_rows"] = len(reference_df)
    drift_summary["current_rows"] = len(current_df)

    # Save JSON results (for automation/alerting)
    json_path = os.path.join(output_dir, f"{report_name}.json")
    with open(json_path, "w") as f:
        json.dump(drift_summary, f, indent=2)
    print(f"  JSON results saved: {json_path}")

    return drift_summary


def run_data_quality_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str = "quality_report",
    output_dir: str = "reports/drift"
) -> None:
    """
    Run Evidently data quality report using DataDriftPreset.

    In v0.7.x, DataDriftPreset also captures quality signals
    like missing values and distribution anomalies.
    """
    os.makedirs(output_dir, exist_ok=True)

    report = Report([
        DataDriftPreset(),
    ])

    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    html_path = os.path.join(output_dir, f"{report_name}.html")
    snapshot.save_html(html_path)
    print(f"  Quality report saved: {html_path}")


def parse_drift_results(result: dict) -> dict:
    """
    Parse Evidently v0.7.21 result dict into a clean summary.

    Evidently v0.7.21 structure:
    {
      "metrics": [
        {
          "metric_name": "DriftedColumnsCount(drift_share=0.5)",
          "value": {"count": 2.0, "share": 1.0}
        },
        {
          "metric_name": "ValueDrift(column=text_length,...,threshold=0.05)",
          "config": {"column": "text_length", "threshold": 0.05},
          "value": 0.0    ← p-value (lower = more drift)
        },
        ...
      ]
    }

    DRIFT LOGIC:
    - DriftedColumnsCount.value.share > 0.5 → dataset drift
    - ValueDrift.value < threshold → that column drifted
    """
    summary = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "drifted_columns": [],
    }

    try:
        metrics = result.get("metrics", [])

        for metric in metrics:
            metric_name = metric.get("metric_name", "")
            config = metric.get("config", {})
            value = metric.get("value", None)

            # Parse DriftedColumnsCount — overall dataset drift
            if "DriftedColumnsCount" in metric_name:
                if isinstance(value, dict):
                    drift_count = value.get("count", 0)
                    drift_share = value.get("share", 0.0)
                    summary["drift_share"] = drift_share
                    summary["number_of_drifted_columns"] = int(drift_count)
                    # Dataset drift if more than 50% of columns drifted
                    summary["dataset_drift"] = drift_share > 0.5

            # Parse ValueDrift — per-column drift
            if "ValueDrift" in metric_name:
                column_name = config.get("column", "unknown")
                threshold = config.get("threshold", 0.05)
                summary["number_of_columns"] = summary.get("number_of_columns", 0) + 1

                # value = p-value. If p-value < threshold → drift detected
                if isinstance(value, (int, float)) and value < threshold:
                    if column_name not in summary["drifted_columns"]:
                        summary["drifted_columns"].append(column_name)

    except Exception as e:
        print(f"  Warning: Could not parse drift results: {e}")

    return summary


def print_drift_summary(summary: dict) -> None:
    """Pretty print drift detection results."""
    print("\n" + "=" * 60)
    print(f"  DRIFT REPORT: {summary.get('report_name', 'unknown')}")
    print(f"  Timestamp: {summary.get('timestamp', 'unknown')}")
    print("=" * 60)
    print(f"  Reference rows:  {summary.get('reference_rows', 0)}")
    print(f"  Current rows:    {summary.get('current_rows', 0)}")
    print("-" * 60)

    drift_detected = summary.get("dataset_drift", False)
    drift_share = summary.get("drift_share", 0.0)
    drifted_cols = summary.get("drifted_columns", [])

    if drift_detected:
        print("  DRIFT DETECTED!")
        print(f"  Drift share: {drift_share:.1%} of columns drifted")
        print(f"  Drifted columns: {', '.join(drifted_cols)}")
    else:
        print("  NO DRIFT DETECTED")
        print(f"  Drift share: {drift_share:.1%}")
        if drifted_cols:
            print(f"  Minor drift in: {', '.join(drifted_cols)}")

    print("=" * 60 + "\n")
