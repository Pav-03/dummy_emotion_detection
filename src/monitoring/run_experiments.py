import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.monitoring.drift_detection import (
    extract_text_features,
    run_data_drift_report,
    run_data_quality_report,
    print_drift_summary,
)


def load_reference_data() -> pd.DataFrame:
    """Load training data as reference (baseline)."""
    df = pd.read_csv("data/interim/train_processed.csv")
    print(f"Reference data loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts().to_dict()}")
    return df

# Experiment -1: NO DRIFT (Baseline)


"""
SCENARIO: Normal day. Production data looks like training data.
EXPECTED: No drift detected. This is your "green" baseline.
"""


def experiment_1_no_drift(reference_df: pd.DataFrame):
    """
    Simulate: Sample from training data itself (no change).
    Expected: No drift. All green.
    """
    print("\n" + "₹" * 30)
    print("EXPERIMENT 1: NO DRIFT (Baseline)")
    print("Scenario: Production data is similar to training data")
    print("₹" * 30)

    # "Production" data = random sample from same training data
    # This simulates a normal day where inputs look like training data
    current_df = reference_df.sample(n=500, random_state=42)

    # Extract features
    ref_features = extract_text_features(reference_df)
    cur_features = extract_text_features(current_df)

    # Run drift report
    summary = run_data_drift_report(
        reference_df=ref_features,
        current_df=cur_features,
        report_name="exp1_no_drift"
    )

    print_drift_summary(summary)

    # INVESTIGATION
    print("INVESTIGATION:")
    if not summary["dataset_drift"]:
        print("  As expected — no drift. Model is operating on familiar data.")
        print("  ACTION: None needed. This is the healthy state.")
    else:
        print("  Unexpected drift detected in baseline!")
        print("  This could mean the sample is too small or data has internal variance.")
        print("  ACTION: Increase sample size or investigate reference data quality.")

    return summary


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 2: DATA DRIFT (Input text changes)
# ═══════════════════════════════════════════════════════════

"""
SCENARIO: Users start sending very short texts (tweets/SMS)
   instead of longer reviews. The MODEL hasn't changed, but the
  INPUTS have. This is the most common type of drift.

REAL COMPANY EXAMPLE:
   Sentiment analysis model trained on product reviews (avg 50 words).
   Mobile app launches → users send 5-word feedback.
   Model struggles because it was trained on longer text.
"""


def experiment_2_data_drift(reference_df: pd.DataFrame):
    """
    Simulate: Short, simple texts (tweets instead of reviews).
    Expected: Drift detected in text_length, word_count, avg_word_length.
    """
    print("\n" + "🟡" * 30)
    print("EXPERIMENT 2: DATA DRIFT (Short texts)")
    print("Scenario: Users start sending very short texts")
    print("🟡" * 30)

    # Simulate short production texts
    short_texts = [
        "happy", "sad", "love it", "hate it", "ok", "meh",
        "great", "awful", "nice", "bad", "yay", "nope",
        "good stuff", "terrible", "amazing", "horrible",
        "so good", "so bad", "love", "angry", "yes", "no",
        "cool", "boring", "fun", "worst", "best", "fine",
        "thanks", "ugh", "wow", "lol", "cry", "smile",
        "hurt", "glad", "mad", "sick", "tired", "excited",
        "happy day", "bad day", "not good", "very nice",
        "hate this", "love this", "so happy", "so sad",
        "feeling good", "feeling bad",
    ]

    # Create 500 "production" rows from short texts
    np.random.seed(42)
    current_data = pd.DataFrame({
        "sentiment": np.random.choice([0, 1], size=500),
        "content": np.random.choice(short_texts, size=500),
    })

    print(f"\nReference avg text length: {reference_df['content'].str.len().mean():.1f} chars")
    print(f"Current avg text length:   {current_data['content'].str.len().mean():.1f} chars")
    print(f"Reference avg word count:  {reference_df['content'].str.split().str.len().mean():.1f} words")
    print(f"Current avg word count:    {current_data['content'].str.split().str.len().mean():.1f} words")

    # Extract features and run drift report
    ref_features = extract_text_features(reference_df)
    cur_features = extract_text_features(current_data)

    summary = run_data_drift_report(
        reference_df=ref_features,
        current_df=cur_features,
        report_name="exp2_data_drift_short_texts"
    )

    print_drift_summary(summary)

    # INVESTIGATION
    print("INVESTIGATION:")
    print("  🔍 Root cause: Input text length dropped dramatically.")
    print(f"     Training data avg: ~{reference_df['content'].str.len().mean():.0f} chars")
    print(f"     Production data avg: ~{current_data['content'].str.len().mean():.0f} chars")
    print()
    print("  DECISION PROCESS (what a senior engineer thinks):")
    print("  1. Is the model still accurate on short texts? → Need to test")
    print("  2. If yes → Update reference data, no retraining needed")
    print("  3. If no  → Collect short text training data → Retrain")
    print()
    print("  ACTION: Test model accuracy on short text sample.")
    print("  If accuracy < 70%: retrain with short text data.")
    print("  If accuracy >= 70%: update reference dataset to include short texts.")

    return summary


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 3: PREDICTION DRIFT (Model favoring one class)
# ═══════════════════════════════════════════════════════════
"""
SCENARIO: Suddenly 90% of predictions are "negative".
   Either users genuinely got more negative (real change),
   or the model is broken (bug).

REAL COMPANY EXAMPLE:
   After a bad product launch, customer reviews genuinely
   became 80% negative. The model is working correctly!
   BUT: a corrupt model file could also cause this.
   You need to INVESTIGATE to know which one.
"""


def experiment_3_prediction_drift(reference_df: pd.DataFrame):
    """
    Simulate: 90% negative predictions.
    Expected: Drift detected in prediction distribution.
    """
    print("\n" + "🔴" * 30)
    print("EXPERIMENT 3: PREDICTION DRIFT (Imbalanced predictions)")
    print("Scenario: 90% of predictions are negative")
    print("🔴" * 30)

    # Create data where 90% is negative sentiment
    np.random.seed(42)

    # Take real texts but skew the distribution
    negative_texts = reference_df[reference_df["sentiment"] == 0].sample(
        n=450, random_state=42, replace=True
    )
    positive_texts = reference_df[reference_df["sentiment"] == 1].sample(
        n=50, random_state=42
    )
    current_data = pd.concat([negative_texts, positive_texts]).reset_index(drop=True)

    ref_dist = reference_df["sentiment"].value_counts(normalize=True).to_dict()
    cur_dist = current_data["sentiment"].value_counts(normalize=True).to_dict()

    print(f"\nReference prediction distribution: {ref_dist}")
    print(f"Current prediction distribution:   {cur_dist}")

    # Extract features
    ref_features = extract_text_features(reference_df)
    cur_features = extract_text_features(current_data)

    summary = run_data_drift_report(
        reference_df=ref_features,
        current_df=cur_features,
        report_name="exp3_prediction_drift"
    )

    print_drift_summary(summary)

    # INVESTIGATION
    print("INVESTIGATION:")
    print("  🔍 Prediction distribution shifted from ~50/50 to 90/10 (negative heavy)")
    print()
    print("  TWO POSSIBLE CAUSES:")
    print("  Cause A: Users genuinely sending more negative content")
    print("    → Check if input data ALSO drifted")
    print("    → If inputs drifted: real user behavior change")
    print("    → ACTION: Monitor, but no fix needed")
    print()
    print("  Cause B: Model is broken (corrupt file, wrong version)")
    print("    → Check if input data is NORMAL but predictions are skewed")
    print("    → If inputs normal but predictions skewed: model bug")
    print("    → ACTION: Rollback to previous model version")
    print()

    # Check if input data also drifted
    input_drifted = any(
        col in summary.get("drifted_columns", [])
        for col in ["text_length", "word_count", "avg_word_length"]
    )
    pred_drifted = "prediction" in summary.get("drifted_columns", [])

    if input_drifted and pred_drifted:
        print("  DIAGNOSIS: Both inputs AND predictions drifted.")
        print("  → Likely Cause A: Real user behavior change.")
        print("  → ACTION: Monitor closely. Consider rebalancing training data.")
    elif pred_drifted and not input_drifted:
        print("  DIAGNOSIS: Predictions drifted but inputs are normal!")
        print("  → Likely Cause B: Model is broken.")
        print("  → ACTION: Rollback model immediately. Investigate model file.")
    else:
        print("  DIAGNOSIS: Inconclusive. Need more data.")

    return summary


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 4: DATA QUALITY ISSUES (Dirty data)
# ═══════════════════════════════════════════════════════════
"""
SCENARIO: Upstream data pipeline breaks. You receive:
   - Empty strings
   - URLs instead of text
   - Numbers only
   - Special characters / HTML tags

REAL COMPANY EXAMPLE:
   Data pipeline that feeds user reviews breaks overnight.
   Instead of review text, it sends raw HTML tags.
   Model receives "<div class='review'>Great product</div>"
   Preprocessing strips it to "great product" sometimes,
   but other times it's just empty.
"""


def experiment_4_data_quality(reference_df: pd.DataFrame):
    """
    Simulate: Dirty/corrupted input data.
    Expected: Drift AND quality issues detected.
    """
    print("\n" + "🔴" * 30)
    print("EXPERIMENT 4: DATA QUALITY ISSUES (Dirty data)")
    print("Scenario: Upstream pipeline sends corrupted data")
    print("🔴" * 30)

    np.random.seed(42)

    # Mix of good data (60%) and garbage data (40%)
    good_data = reference_df.sample(n=300, random_state=42)

    # Create garbage data
    garbage_texts = (
        [""] * 50 +                                           # empty strings
        ["http://spam.com/click-here"] * 30 +                 # URLs
        ["12345678", "999", "0000", "42"] * 10 +              # numbers only
        ["!!!???..."] * 20 +                                  # special chars only
        ["<div>buy now</div>", "<p>click</p>"] * 15 +        # HTML tags
        ["a"] * 25 +                                          # single character
        ["the the the the the"] * 10 +                        # repetitive
        [""] * 20                                             # more empties
    )

    garbage_df = pd.DataFrame({
        "sentiment": np.random.choice([0, 1], size=len(garbage_texts)),
        "content": garbage_texts,
    })

    current_data = pd.concat([good_data, garbage_df]).reset_index(drop=True)

    # Stats
    empty_pct = (current_data["content"].str.strip() == "").mean()
    short_pct = (current_data["content"].str.len() < 5).mean()

    print(f"\nCurrent data: {len(current_data)} rows")
    print(f"Empty strings: {empty_pct:.1%}")
    print(f"Very short (<5 chars): {short_pct:.1%}")

    # Extract features
    ref_features = extract_text_features(reference_df)
    cur_features = extract_text_features(current_data)

    # Run BOTH drift and quality reports
    summary = run_data_drift_report(
        reference_df=ref_features,
        current_df=cur_features,
        report_name="exp4_data_quality_drift"
    )

    run_data_quality_report(
        reference_df=ref_features,
        current_df=cur_features,
        report_name="exp4_data_quality_report"
    )

    print_drift_summary(summary)

    # INVESTIGATION
    print("INVESTIGATION:")
    print(f"  🔍 {empty_pct:.1%} of inputs are empty strings")
    print(f"  🔍 {short_pct:.1%} of inputs are very short (<5 chars)")
    print()
    print("  THIS IS NOT MODEL DRIFT — THIS IS A DATA PIPELINE BUG!")
    print()
    print("  DECISION PROCESS:")
    print("  1. DON'T retrain the model (model is fine, data is broken)")
    print("  2. Fix the upstream data pipeline")
    print("  3. Add input validation to reject garbage inputs")
    print("  4. Add data quality check BEFORE model inference")
    print()
    print("  ACTION ITEMS:")
    print("  → Alert data engineering team about pipeline issue")
    print("  → Add minimum text length check (reject < 3 chars)")
    print("  → Add empty string rejection in API")
    print("  → Add data quality monitoring to catch this faster next time")

    return summary


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 5: GRADUAL DRIFT (Slow degradation over time)
# ═══════════════════════════════════════════════════════════
"""
SCENARIO: Over 5 "weeks", text patterns slowly change.
   Week 1: Normal (no drift)
   Week 2: Slightly shorter texts
   Week 3: More informal, shorter
   Week 4: Very casual, slangy
   Week 5: Mostly slang and abbreviations

REAL COMPANY EXAMPLE:
   App launches on TikTok → Gen-Z users join.
   Their language is different from the training data (formal reviews).
   Change is gradual — hard to catch without monitoring.

WHY THIS IS THE HARDEST TO DETECT:
   Each individual week looks "close enough" to the previous week.
   But week 5 vs training data = massive drift.
   Like a frog in slowly boiling water.
"""


def experiment_5_gradual_drift(reference_df: pd.DataFrame):
    """
    Simulate: Gradual text pattern change over 5 weeks.
    Expected: Drift score increases each week.
    """
    print("\n" + "🟡" * 30)
    print("EXPERIMENT 5: GRADUAL DRIFT (5 weeks)")
    print("Scenario: Text patterns slowly change over time")
    print("🟡" * 30)

    ref_features = extract_text_features(reference_df)

    # Define text patterns for each "week"
    week_texts = {
        1: reference_df["content"].tolist()[:500],  # Week 1: normal training-like data
        2: [  # Week 2: slightly shorter, some informal
            "pretty good experience overall",
            "not great honestly",
            "loved it a lot",
            "kinda disappointed",
            "was ok nothing special",
            "really enjoyed this",
            "waste of my time",
            "super happy with this",
            "not what i expected",
            "would recommend to friends",
        ],
        3: [  # Week 3: more casual
            "its good",
            "nah not for me",
            "yep love it",
            "pretty bad tbh",
            "its fine i guess",
            "omg so good",
            "meh whatever",
            "def recommend",
            "not great ngl",
            "vibes are good",
        ],
        4: [  # Week 4: very casual/slangy
            "fire",
            "mid",
            "slaps",
            "no cap this good",
            "lowkey bad",
            "bussin fr fr",
            "its giving sad",
            "slay",
            "bruh moment",
            "ngl kinda trash",
        ],
        5: [  # Week 5: abbreviations and slang
            "W",
            "L",
            "goated",
            "sus",
            "bet",
            "ong",
            "fr",
            "nah",
            "ick",
            "yeet",
        ],
    }

    weekly_results = []

    for week in range(1, 6):
        print(f"\n--- Week {week} ---")

        # Create current data for this week
        np.random.seed(week)
        texts = week_texts[week]
        current_data = pd.DataFrame({
            "sentiment": np.random.choice([0, 1], size=min(500, len(texts) * 50)),
            "content": np.random.choice(texts, size=min(500, len(texts) * 50)),
        })

        avg_len = current_data["content"].str.len().mean()
        avg_words = current_data["content"].str.split().str.len().mean()
        print(f"  Avg text length: {avg_len:.1f} chars")
        print(f"  Avg word count: {avg_words:.1f} words")

        cur_features = extract_text_features(current_data)

        summary = run_data_drift_report(
            reference_df=ref_features,
            current_df=cur_features,
            report_name=f"exp5_week_{week}"
        )

        drift_share = summary.get("drift_share", 0.0)
        drifted = summary.get("dataset_drift", False)
        print(f"  Drift share: {drift_share:.1%}")
        print(f"  Dataset drift: {'YES' if drifted else 'NO'}")

        weekly_results.append({
            "week": week,
            "drift_share": drift_share,
            "dataset_drift": drifted,
            "avg_text_length": avg_len,
            "avg_word_count": avg_words,
            "drifted_columns": summary.get("drifted_columns", []),
        })

    # INVESTIGATION — trend analysis
    print("\n" + "=" * 60)
    print("  GRADUAL DRIFT TREND ANALYSIS")
    print("=" * 60)
    print(f"  {'Week':<6} {'Drift%':<10} {'Drifted?':<10} {'Avg Length':<12} {'Avg Words':<10}")
    print("-" * 60)
    for r in weekly_results:
        print(
            f"  {r['week']:<6} {r['drift_share']:<10.1%} "
            f"{'YES' if r['dataset_drift'] else 'NO':<10} "
            f"{r['avg_text_length']:<12.1f} {r['avg_word_count']:<10.1f}"
        )
    print("=" * 60)

    print("\nINVESTIGATION:")
    print("  🔍 Drift increases each week as text gets shorter and more informal.")
    print()
    print("  DECISION PROCESS:")
    print("  1. Week 1-2: No action needed (drift within normal range)")
    print("  2. Week 3: Set a WATCH — drift is growing")
    print("  3. Week 4: ALERT — drift crossed threshold")
    print("     → Test model accuracy on current data")
    print("     → If accuracy dropped: start retraining pipeline")
    print("  4. Week 5: CRITICAL — model is operating on completely unfamiliar data")
    print("     → Immediate retraining required")
    print("     → Consider adding slang/casual text to training data")
    print()
    print("  AUTOMATED TRIGGER LOGIC:")
    print("  if drift_share > 0.3 for 3 consecutive days:")
    print("      trigger_retraining_pipeline()")
    print("      notify_ml_team()")

    return weekly_results


# ═══════════════════════════════════════════════════════════
# MAIN — Run all experiments
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  DRIFT DETECTION EXPERIMENTS")
    print("  Simulating real production scenarios")
    print("=" * 60)

    # Load reference data
    reference_df = load_reference_data()

    # Create reports directory
    os.makedirs("reports/drift", exist_ok=True)

    # Run experiments
    print("\n\n" + "=" * 60)
    print("  Running Experiment 1: No Drift (Baseline)")
    print("=" * 60)
    exp1 = experiment_1_no_drift(reference_df)

    print("\n\n" + "=" * 60)
    print("  Running Experiment 2: Data Drift (Short Texts)")
    print("=" * 60)
    exp2 = experiment_2_data_drift(reference_df)

    print("\n\n" + "=" * 60)
    print("  Running Experiment 3: Prediction Drift")
    print("=" * 60)
    exp3 = experiment_3_prediction_drift(reference_df)

    print("\n\n" + "=" * 60)
    print("  Running Experiment 4: Data Quality Issues")
    print("=" * 60)
    exp4 = experiment_4_data_quality(reference_df)

    print("\n\n" + "=" * 60)
    print("  Running Experiment 5: Gradual Drift (5 Weeks)")
    print("=" * 60)
    exp5 = experiment_5_gradual_drift(reference_df)

    # Final summary
    print("\n\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print("\n  Reports saved in: reports/drift/")
    print("  Open HTML files in browser to see visual reports:\n")
    print("    open reports/drift/exp1_no_drift.html")
    print("    open reports/drift/exp2_data_drift_short_texts.html")
    print("    open reports/drift/exp3_prediction_drift.html")
    print("    open reports/drift/exp4_data_quality_drift.html")
    print("    open reports/drift/exp4_data_quality_report.html")
    print("    open reports/drift/exp5_week_1.html")
    print("    open reports/drift/exp5_week_2.html")
    print("    open reports/drift/exp5_week_3.html")
    print("    open reports/drift/exp5_week_4.html")
    print("    open reports/drift/exp5_week_5.html")
    print()
    print("  INTERVIEW TIP: Open these reports and walk through")
    print("  the investigation process for each scenario.")
