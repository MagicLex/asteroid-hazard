"""Run-progression chart: val_metric across kept experiments, in order.

Reads the leaderboard FG, keeps the 'keep' rows sorted by event time, and draws
the improvement curve into model/progression.png so the registry card shows the
whole search as one picture. Called at register time (on keeps).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hopsworks

TAG = "albedo"
OUT = Path(__file__).resolve().parent / "model" / "progression.png"


def main():
    fs = hopsworks.login().get_feature_store()
    df = fs.get_feature_group(f"autoresearch_experiments_{TAG}", version=1).read(
        dataframe_type="pandas")
    kept = df[df["status"] == "keep"].sort_values("ts").reset_index(drop=True)
    if kept.empty:
        print("no kept rows yet; skipping progression chart")
        return
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0b0e11")
    ax.set_facecolor("#0b0e11")
    ax.plot(range(1, len(kept) + 1), kept["val_metric"], "-o",
            color="#d97706", lw=1.8, ms=5)
    best = kept["val_metric"].max()
    ax.axhline(best, color="#34d399", ls="--", lw=1, alpha=0.6)
    for i, (_, r) in enumerate(kept.iterrows(), 1):
        ax.annotate(f"{r['val_metric']:.4f}", (i, r["val_metric"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", color="#cbd5e1", fontsize=7)
    ax.set_xlabel("kept experiment", color="#cbd5e1")
    ax.set_ylabel("CV ROC-AUC", color="#cbd5e1")
    ax.set_title(f"autoresearch {TAG}: kept-experiment progression (best {best:.4f})",
                 color="#e5e7eb", fontsize=10)
    ax.tick_params(colors="#475569")
    for s in ax.spines.values():
        s.set_color("#1f2937")
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, facecolor="#0b0e11")
    print(f"wrote {OUT} ({len(kept)} kept)")


if __name__ == "__main__":
    main()
