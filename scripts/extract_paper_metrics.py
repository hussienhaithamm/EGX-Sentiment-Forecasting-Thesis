"""Extract thesis-paper metric values from final_sentiment_experiments output CSVs.

Run after run_multi_horizon_event_evaluation.py completes to get all values
needed for updating results.tex tables and prose.
"""

import json
import pandas as pd
from pathlib import Path

OUTPUTS = Path("outputs/final_sentiment_experiments")


def fmt(v, decimals=3):
    """Format a float to fixed decimals."""
    return f"{v:.{decimals}f}"


def main():
    dir_csv = OUTPUTS / "direction_metrics_by_horizon.csv"
    ret_csv = OUTPUTS / "return_metrics_by_horizon.csv"
    evt_csv = OUTPUTS / "event_vs_all_news_comparison.csv"
    cov_csv = OUTPUTS / "coverage_analysis.csv"
    leak_json = OUTPUTS / "leakage_check_report.json"
    over_csv = OUTPUTS / "overfitting_report.csv"

    dm = pd.read_csv(dir_csv)
    rm = pd.read_csv(ret_csv)
    ev = pd.read_csv(evt_csv)

    # ── Tab: main_results ────────────────────────────────────────────────────
    print("\n=== tab:main_results (all_news bilingual test) ===")
    print("Horizon & Ticker & Accuracy & Balanced Acc. & F1 & Sign Acc. & Return Corr. \\\\")
    for h in [1, 3, 5]:
        for ticker in ["CIB", "TMG", "OIH"]:
            d_row = dm[
                (dm["horizon"] == h)
                & (dm["feature_strategy"] == "all_news")
                & (dm["ticker"] == ticker)
                & (dm["view"] == "bilingual")
                & (dm["split"] == "test")
            ]
            r_row = rm[
                (rm["horizon"] == h)
                & (rm["feature_strategy"] == "all_news")
                & (rm["ticker"] == ticker)
                & (rm["view"] == "bilingual")
                & (rm["split"] == "test")
            ]
            if d_row.empty or r_row.empty:
                print(f"  MISSING: H={h} {ticker}")
                continue
            acc = d_row["accuracy"].iloc[0]
            ba = d_row["balanced_accuracy"].iloc[0]
            f1 = d_row["f1"].iloc[0]
            sign = r_row["sign_accuracy"].iloc[0]
            corr = r_row["return_correlation"].iloc[0]
            print(
                f"{h} & {ticker} & {fmt(acc)} & {fmt(ba)} & {fmt(f1)} & {fmt(sign)} & {fmt(corr)} \\\\"
            )

    # ── OIH H5 headline ──────────────────────────────────────────────────────
    print("\n=== OIH H5 all-news bilingual headline ===")
    oih5d = dm[
        (dm["horizon"] == 5) & (dm["feature_strategy"] == "all_news")
        & (dm["ticker"] == "OIH") & (dm["view"] == "bilingual") & (dm["split"] == "test")
    ]
    oih5r = rm[
        (rm["horizon"] == 5) & (rm["feature_strategy"] == "all_news")
        & (rm["ticker"] == "OIH") & (rm["view"] == "bilingual") & (rm["split"] == "test")
    ]
    if not oih5d.empty and not oih5r.empty:
        print(f"  accuracy={fmt(oih5d['accuracy'].iloc[0])}")
        print(f"  balanced_accuracy={fmt(oih5d['balanced_accuracy'].iloc[0])}")
        print(f"  f1={fmt(oih5d['f1'].iloc[0])}")
        print(f"  sign_accuracy={fmt(oih5r['sign_accuracy'].iloc[0])}")
        print(f"  return_correlation={fmt(oih5r['return_correlation'].iloc[0])}")

    # ── OIH H5 accepted validation metrics ───────────────────────────────────
    print("\n=== OIH H5 all-news bilingual VALIDATION (for acceptance prose) ===")
    oih5v_d = dm[
        (dm["horizon"] == 5) & (dm["feature_strategy"] == "all_news")
        & (dm["ticker"] == "OIH") & (dm["view"] == "bilingual") & (dm["split"] == "validation")
    ]
    oih5v_r = rm[
        (rm["horizon"] == 5) & (rm["feature_strategy"] == "all_news")
        & (rm["ticker"] == "OIH") & (rm["view"] == "bilingual") & (rm["split"] == "validation")
    ]
    if not oih5v_d.empty:
        print(f"  val_accuracy={fmt(oih5v_d['accuracy'].iloc[0])}")
        print(f"  val_balanced_accuracy={fmt(oih5v_d['balanced_accuracy'].iloc[0])}")
        print(f"  val_f1={fmt(oih5v_d['f1'].iloc[0])}")

    # ── Best balanced-accuracy row ────────────────────────────────────────────
    print("\n=== Best balanced_accuracy in all-news bilingual test ===")
    test_bl = dm[
        (dm["feature_strategy"] == "all_news") & (dm["view"] == "bilingual") & (dm["split"] == "test")
    ]
    if not test_bl.empty:
        best = test_bl.loc[test_bl["balanced_accuracy"].idxmax()]
        print(f"  H={best['horizon']} {best['ticker']} BA={fmt(best['balanced_accuracy'])}")

    # ── Event-only top performers ─────────────────────────────────────────────
    print("\n=== Event-only specific cases (CIB H1, CIB H3, OIH H5) ===")
    for (h, t) in [(1, "CIB"), (3, "CIB"), (5, "OIH")]:
        eo = dm[
            (dm["horizon"] == h) & (dm["feature_strategy"] == "event_only")
            & (dm["ticker"] == t) & (dm["view"] == "bilingual") & (dm["split"] == "test")
        ]
        if not eo.empty:
            print(f"  {t} H{h} event_only: acc={fmt(eo['accuracy'].iloc[0])} f1={fmt(eo['f1'].iloc[0])}")

    # OIH H5 event-weighted
    ew5oih = dm[
        (dm["horizon"] == 5) & (dm["feature_strategy"] == "event_weighted")
        & (dm["ticker"] == "OIH") & (dm["view"] == "bilingual") & (dm["split"] == "test")
    ]
    if not ew5oih.empty:
        print(f"  OIH H5 event_weighted: acc={fmt(ew5oih['accuracy'].iloc[0])} f1={fmt(ew5oih['f1'].iloc[0])}")

    # ── Overfitting-flagged event-weighted candidate ──────────────────────────
    print("\n=== OIH H5 event_weighted TRAIN vs VAL (validity check) ===")
    oih5_ew_train = dm[
        (dm["horizon"] == 5) & (dm["feature_strategy"] == "event_weighted")
        & (dm["ticker"] == "OIH") & (dm["view"] == "bilingual") & (dm["split"] == "train")
    ]
    oih5_ew_val = dm[
        (dm["horizon"] == 5) & (dm["feature_strategy"] == "event_weighted")
        & (dm["ticker"] == "OIH") & (dm["view"] == "bilingual") & (dm["split"] == "validation")
    ]
    if not oih5_ew_train.empty:
        print(f"  train acc={fmt(oih5_ew_train['accuracy'].iloc[0])}")
    if not oih5_ew_val.empty:
        print(f"  val acc={fmt(oih5_ew_val['accuracy'].iloc[0])}")

    # ── Tab: event_summary ───────────────────────────────────────────────────
    print("\n=== tab:event_summary (aggregate event deltas, bilingual test, per-ticker rows) ===")
    test_ev = ev[(ev["split"] == "test") & (ev["view"] == "bilingual") & (ev["ticker"] != "ALL")]
    if not test_ev.empty:
        mean_eo = test_ev["delta_accuracy_event_only"].mean()
        best_eo = test_ev["delta_accuracy_event_only"].max()
        mean_ew = test_ev["delta_accuracy_event_weighted"].mean()
        best_ew = test_ev["delta_accuracy_event_weighted"].max()
        print(f"  event-only   mean delta={fmt(mean_eo)} best positive={fmt(best_eo)}")
        print(f"  event-weighted mean delta={fmt(mean_ew)} best positive={fmt(best_ew)}")
    else:
        print("  WARNING: no per-ticker bilingual test rows found in event_vs_all_news")

    # ── Coverage ─────────────────────────────────────────────────────────────
    if cov_csv.exists():
        print("\n=== Coverage analysis ===")
        cov = pd.read_csv(cov_csv)
        print(cov.to_string(index=False))

    # ── Leakage report ───────────────────────────────────────────────────────
    if leak_json.exists():
        print("\n=== Leakage check summary ===")
        with open(leak_json) as f:
            leak = json.load(f)
        total_panels = leak.get("total_panels", "?")
        total_rows = leak.get("total_rows_checked", "?")
        violations = leak.get("total_violations", "?")
        print(f"  panels={total_panels} rows={total_rows} violations={violations}")

    # ── Overfitting report ────────────────────────────────────────────────────
    if over_csv.exists():
        print("\n=== Overfitting flags ===")
        ov = pd.read_csv(over_csv)
        if "overfitting_flag" in ov.columns:
            flagged = ov["overfitting_flag"].sum()
            total = len(ov)
            print(f"  flagged={flagged} of {total}")
        else:
            print("  Columns:", ov.columns.tolist())


if __name__ == "__main__":
    main()
