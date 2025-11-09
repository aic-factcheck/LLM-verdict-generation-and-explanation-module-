#!/usr/bin/env python3
import json
import pandas as pd
from collections import Counter
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import seaborn as sns
import re
import nltk
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works everywhere)
import matplotlib.pyplot as plt

# Ensure required tokenizers are available
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)
# Optional: nltk sentence/word tokenizers
nltk.download("punkt", quiet=True)

# ========== CONFIG ==========
INPUT_FILE = "rci_250903_0_all_archive_retry_md_bertik_csv_tmsp_expl2md_readme/combined.jsonl"
SAVE_PLOTS = True
OUT_DIR = Path("analysis_results")
OUT_DIR.mkdir(exist_ok=True)

# ========== LOAD DATA ==========
def load_jsonl(file):
    rows = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)

df = load_jsonl(INPUT_FILE)
print(f"Loaded {len(df)} statements")

# ========== CLEAN AND PREPARE ==========
# Extract date
df["statement_excerpted_at"] = pd.to_datetime(
    df["statement_excerpted_at"], errors="coerce"
)

# Speaker extraction (very rough heuristic)
def extract_speaker(text):
    # Usually "Name: rest"
    m = re.match(r"([A-ZÁÉÍÓÚÝČĎĚŇŘŠŤŽ][^:]{2,30}):", text)
    return m.group(1).strip() if m else None

df["speaker"] = df["statement_content"].apply(extract_speaker)

# Evidence count
df["evidence_count"] = df["evidence"].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Clean HTML explanations
def html_to_text(html):
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True) if html else ""

df["explanation_text"] = df["assessment_explanation_html"].apply(html_to_text)

# ========== A) Quantitative / Descriptive Statistics ==========
print("\n=== A) Descriptive Statistics ===")

# 1. Count per veracity
veracity_counts = df["assessment_veracity"].value_counts()
print("\nVeracity counts:\n", veracity_counts)

# 2. Frequency over time
# Ensure datetime conversion succeeded
df["statement_excerpted_at"] = pd.to_datetime(df["statement_excerpted_at"], errors="coerce")

# Filter out NaT and set as index
time_series = df.dropna(subset=["statement_excerpted_at"]).copy()
time_series = time_series.set_index(pd.DatetimeIndex(time_series["statement_excerpted_at"]))

# Resample by year-end ('YE')
time_counts = time_series.resample("YE")["statement_id"].count()

# 4. Evidence count distribution
evidence_stats = df["evidence_count"].describe()
print("\nEvidence count stats:\n", evidence_stats)

# 1. Statement length (in words)
df["statement_length"] = df["statement_content"].str.split().apply(len)

# 2. Explanation length (in words and sentences)
df["explanation_word_len"] = df["explanation_text"].str.split().apply(len)
df["explanation_sent_len"] = df["explanation_text"].apply(lambda t: len(nltk.sent_tokenize(t)))

# 3. Compare HTML vs text lengths
df["html_len"] = df["assessment_explanation_html"].apply(lambda t: len(t or ""))
df["text_len"] = df["explanation_text"].apply(len)
df["html_text_ratio"] = df["html_len"] / df["text_len"].replace(0, 1)

# ========== C) Fact-Checking Structure Analysis ==========
print("\n=== C) Structure Analysis ===")

# 1. Ratio of successful evidence fetches
fetched_ratio = df["evidence"].apply(
    lambda ev: sum(1 for e in ev if e.get("fetched_url")) / len(ev) if ev else 0
)
print(f"Average ratio of fetched evidence: {fetched_ratio.mean():.2f}")

# 2. Average evidence count
avg_evidence = df["evidence_count"].mean()
print(f"Average evidence per statement: {avg_evidence:.2f}")

# 3. Temporal mismatches (rough proxy)
# If evidence has a date in fetched_url, compare it to statement date
def extract_year(url):
    if not url: return None
    m = re.search(r"/(19|20)\d{2}/", url)
    return int(m.group(0).strip("/")) if m else None

df["evidence_years"] = df["evidence"].apply(
    lambda ev: [extract_year(e.get("fetched_url", "")) for e in ev if isinstance(e, dict)]
)
df["temporal_mismatch"] = df.apply(
    lambda r: any(abs(y - r["statement_excerpted_at"].year) > 1 for y in r["evidence_years"] if y and pd.notna(r["statement_excerpted_at"])), axis=1
)
print(f"Statements with temporal mismatch: {df['temporal_mismatch'].mean():.2%}")

# ========== PLOTS ==========
if SAVE_PLOTS:
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 4))
    veracity_counts.plot(kind="bar", title="Statements per Veracity")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "veracity_counts.png")

    plt.figure(figsize=(8, 4))
    time_counts.plot(title="Statements per Year")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "statements_over_time.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["evidence_count"], bins=20)
    plt.title("Evidence Count Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "evidence_count_dist.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["statement_length"], bins=30)
    plt.title("Statement Length Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "statement_length_dist.png")

print("\nAll analyses done. Results and plots saved in:", OUT_DIR)
