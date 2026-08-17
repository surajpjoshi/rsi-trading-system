
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

SCRIPT_FOLDER = Path(__file__).resolve().parent
SETUP_HISTORY_FILE = SCRIPT_FOLDER / "Setup_History.csv"
STATISTICS_FILE = SCRIPT_FOLDER / "Trading_Statistics.csv"
TAG_STATISTICS_FILE = SCRIPT_FOLDER / "Trading_Statistics_By_Tag.csv"
IST = ZoneInfo("Asia/Kolkata")


def load_history():
    if not SETUP_HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Setup_History.csv not found: {SETUP_HISTORY_FILE}"
        )
    return pd.read_csv(
        SETUP_HISTORY_FILE,
        encoding="utf-8-sig"
    )


def hit_rate(df, column):
    if len(df) == 0 or column not in df.columns:
        return 0.0
    return round(
        df[column].astype(str).str.upper().eq("YES").mean() * 100,
        2
    )


def avg(df, column):
    if column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return round(values.mean(), 2) if len(values) else 0.0


def build_overall(df):
    status = df["Status"].astype(str).str.upper() if "Status" in df.columns else pd.Series(dtype=str)
    completed = df[status.eq("COMPLETED")] if len(status) else df.iloc[0:0]
    active = int(status.eq("ACTIVE").sum()) if len(status) else 0

    rows = [
        ["Total Setups", len(df)],
        ["Completed Setups", len(completed)],
        ["Active Setups", active],
        ["+1% Hit Rate %", hit_rate(df, "Plus 1% Hit")],
        ["+2% Hit Rate %", hit_rate(df, "Plus 2% Hit")],
        ["+3% Hit Rate %", hit_rate(df, "Plus 3% Hit")],
        ["+5% Hit Rate %", hit_rate(df, "Plus 5% Hit")],
        ["-1% Hit Rate %", hit_rate(df, "Minus 1% Hit")],
        ["-2% Hit Rate %", hit_rate(df, "Minus 2% Hit")],
        ["-3% Hit Rate %", hit_rate(df, "Minus 3% Hit")],
        ["Average Max Gain %", avg(df, "Max Gain %")],
        ["Average Max Loss %", avg(df, "Max Loss %")],
        ["Average 1D Return %", avg(completed, "1D Return %")],
        ["Average 3D Return %", avg(completed, "3D Return %")],
        ["Average 5D Return %", avg(completed, "5D Return %")],
        ["3% Target Win Rate %", hit_rate(completed, "Plus 3% Hit")],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_by_tag(df):
    if "Tag" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["Tag"] = work["Tag"].fillna("").astype(str).str.strip()
    work.loc[work["Tag"] == "", "Tag"] = "UNTAGGED"

    rows = []
    for tag, g in work.groupby("Tag", sort=True):
        completed = g[
            g["Status"].astype(str).str.upper().eq("COMPLETED")
        ]
        rows.append({
            "Tag": tag,
            "Setups": len(g),
            "Completed": len(completed),
            "Active": len(g) - len(completed),
            "+1% Hit Rate %": hit_rate(g, "Plus 1% Hit"),
            "+2% Hit Rate %": hit_rate(g, "Plus 2% Hit"),
            "+3% Hit Rate %": hit_rate(g, "Plus 3% Hit"),
            "+5% Hit Rate %": hit_rate(g, "Plus 5% Hit"),
            "-1% Hit Rate %": hit_rate(g, "Minus 1% Hit"),
            "-2% Hit Rate %": hit_rate(g, "Minus 2% Hit"),
            "-3% Hit Rate %": hit_rate(g, "Minus 3% Hit"),
            "Average Max Gain %": avg(g, "Max Gain %"),
            "Average Max Loss %": avg(g, "Max Loss %"),
            "Average 1D Return %": avg(completed, "1D Return %"),
            "Average 3D Return %": avg(completed, "3D Return %"),
            "Average 5D Return %": avg(completed, "5D Return %"),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["+3% Hit Rate %", "Average Max Gain %", "Setups"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
    return result


def main():
    df = load_history()

    if df.empty:
        print("No setup history records found.")
        return 0

    overall = build_overall(df)
    by_tag = build_by_tag(df)

    overall.to_csv(
        STATISTICS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    if not by_tag.empty:
        by_tag.to_csv(
            TAG_STATISTICS_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    print("=" * 75)
    print("PHASE 6 — TRADING STATISTICS")
    print("=" * 75)
    print("\nOVERALL")
    print(overall.to_string(index=False))

    if not by_tag.empty:
        print("\nBY TAG")
        print(by_tag.to_string(index=False))

    print("\nFiles created:")
    print(STATISTICS_FILE)
    if not by_tag.empty:
        print(TAG_STATISTICS_FILE)

    print(
        "\nGenerated:",
        datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
