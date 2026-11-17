import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
REPORT_DIR = ROOT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Set clean styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_latency_breakdown():
    csv_file = RESULTS_DIR / "latency.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found. Skipping plot.")
        return

    stt_vals, llm_vals, tts_vals, tot_vals = [], [], [], []
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stt_vals.append(float(row["stt_ms"]))
            llm_vals.append(float(row["llm_ms"]))
            tts_vals.append(float(row["tts_ms"]))
            tot_vals.append(float(row["server_total_ms"]))

    if not tot_vals:
        return

    stages = ["STT (Groq)", "LLM (Agent)", "TTS (Edge-TTS)", "Server Total"]
    measured_avgs = [
        np.mean(stt_vals) / 1000.0,
        np.mean(llm_vals) / 1000.0,
        np.mean(tts_vals) / 1000.0,
        np.mean(tot_vals) / 1000.0,
    ]
    budgets = [1.0, 1.0, 0.8, 3.0]

    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, measured_avgs, width, label="Measured Average", color="#2563eb")
    bars2 = ax.bar(x + width/2, budgets, width, label="Design Budget", color="#94a3b8", alpha=0.7)

    ax.set_ylabel("Duration (Seconds)")
    ax.set_title("Voice Pipeline Latency: Measured vs. Design Budget")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(frameon=True)
    ax.axhline(3.0, color="#ef4444", linestyle="--", linewidth=1.5, label="3.0s Hard Limit")

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", weight="bold")

    plt.tight_layout()
    out_path = REPORT_DIR / "latency_breakdown.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_latency_distribution():
    csv_file = RESULTS_DIR / "latency.csv"
    if not csv_file.exists():
        return

    tot_vals = []
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tot_vals.append(float(row["server_total_ms"]) / 1000.0)

    if not tot_vals:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(tot_vals, patch_artist=True, vert=True, widths=0.4,
                     boxprops=dict(facecolor="#38bdf8", color="#0284c7"),
                     medianprops=dict(color="#0f172a", linewidth=2),
                     whiskerprops=dict(color="#0284c7", linewidth=1.5),
                     capprops=dict(color="#0284c7", linewidth=1.5))

    ax.axhline(3.0, color="#ef4444", linestyle="--", linewidth=2, label="3.0s Target Latency")
    ax.set_ylabel("Total Latency (Seconds)")
    ax.set_title("Distribution of End-to-End Voice Turn Latency")
    ax.set_xticklabels(["Arabic Voice Turns (N=%d)" % len(tot_vals)])
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_path = REPORT_DIR / "latency_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_wer_comparison():
    csv_file = RESULTS_DIR / "stt_accuracy.csv"
    if not csv_file.exists():
        return

    from collections import defaultdict
    category_model_wer = defaultdict(lambda: defaultdict(list))
    models = []

    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["model"]
            c = row["category"]
            w = float(row["wer"])
            category_model_wer[c][m].append(w)
            if m not in models:
                models.append(m)

    categories = list(category_model_wer.keys())
    if not categories:
        return

    x = np.arange(len(categories))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#0284c7", "#3b82f6", "#f59e0b"]

    for i, model in enumerate(models):
        mean_wers = [np.mean(category_model_wer[c][model]) * 100 for c in categories]
        short_name = model.replace("groq:", "").replace("local:", "")
        ax.bar(x + i*width - 0.4 + width/2, mean_wers, width, label=short_name, color=colors[i % len(colors)])

    ax.set_ylabel("Word Error Rate (WER %)")
    ax.set_title("STT Word Error Rate Across Models & Arabic Speech Categories")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.legend(frameon=True)

    plt.tight_layout()
    out_path = REPORT_DIR / "wer_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating benchmark visualization plots...")
    plot_latency_breakdown()
    plot_latency_distribution()
    plot_wer_comparison()
    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
