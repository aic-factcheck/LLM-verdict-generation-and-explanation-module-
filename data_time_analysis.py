import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
DATA_PATH = Path("rci_250903_0_all_archive_retry_md_bertik_csv_tmsp_expl2md_readme/combined.jsonl")
OUTPUT_DIR = Path("plots_by_year")
OUTPUT_DIR.mkdir(exist_ok=True)

# === LOAD DATA ===
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            rec = json.loads(line)
            records.append(rec)
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(records)
print(f"Loaded {len(df)} records")

# Before dropping NAs
initial_count = len(df)

# === TIME COLUMN ===
df["statement_excerpted_at"] = pd.to_datetime(
    df["statement_excerpted_at"], errors="coerce", utc=True
)

# Check for NaTs *before* dropping
nat_count = df["statement_excerpted_at"].isna().sum()
if nat_count > 0:
    print(f"WARNING: Dropping {nat_count} records ({nat_count/initial_count:.2%}) due to invalid dates.")

df = df.dropna(subset=["statement_excerpted_at"])
df = df.sort_values("statement_excerpted_at")
df = df.set_index(pd.DatetimeIndex(df["statement_excerpted_at"], name="timestamp"))

# === FEATURE ENGINEERING ===
def safe_len(x):
    return len(str(x)) if pd.notna(x) else 0

df["claim_length"] = df["statement_content"].apply(safe_len)
df["explanation_length"] = df["assessment_explanation_html_evidence_extracted"].apply(safe_len)
df["has_explanation"] = df["explanation_length"] > 0

def count_evidence(evidence):
    if not evidence:
        return 0
    if isinstance(evidence, list):
        return len(evidence)
    return 1

df["evidence_count"] = df["evidence"].apply(count_evidence)
df["has_evidence"] = df["evidence_count"] > 0

# === VERIFY CREATION ===
print("\nColumns for aggregation:", [c for c in df.columns if "length" in c or "evidence" in c])

# === YEARLY AGGREGATES ===
yearly = df.resample("YE").agg({
    "statement_id": "count",
    "claim_length": "mean",
    "explanation_length": "mean",
    "has_explanation": "mean",
    "evidence_count": "mean",
    "has_evidence": "mean"
})
yearly["has_explanation"] *= 100
yearly["has_evidence"] *= 100

# === PLOTS ===
def save_plot(series, title, ylabel, filename):
    plt.figure(figsize=(10, 4))
    series.plot(title=title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()

save_plot(yearly["statement_id"], "Statements per Year", "Count", "statements_per_year.png")
save_plot(yearly["claim_length"], "Average Claim Length per Year", "Characters", "claim_length_per_year.png")
save_plot(yearly["explanation_length"], "Average Explanation Length per Year", "Characters", "explanation_length_per_year.png")
save_plot(yearly["evidence_count"], "Average Number of Evidence URLs per Statement", "Count", "evidence_count_per_year.png")

plt.figure(figsize=(10, 4))
yearly[["has_evidence", "has_explanation"]].plot(title="Percentage with Evidence or Explanation")
plt.ylabel("% of statements")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "percentage_with_evidence_or_explanation.png", dpi=200)
plt.close()

# === VERDICT DISTRIBUTION ===
if "assessment_veracity" in df.columns:
    verdict_counts = df.groupby([pd.Grouper(freq="Y"), "assessment_veracity"]).size().unstack(fill_value=0)
    verdict_counts.plot.area(figsize=(10, 5), title="Verdict Distribution Over Time", stacked=True)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "verdict_distribution_over_time.png", dpi=200)
    plt.close()

print(f"\n✅ All yearly plots saved to: {OUTPUT_DIR.resolve()}")
