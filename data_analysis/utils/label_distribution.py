import pandas as pd
import matplotlib.pyplot as plt


def plot_label_distribution(csv_file: str, output: str) -> str:
    """Save a pie chart of the label distribution in csv_file to output."""
    df = pd.read_csv(csv_file)
    counts = df["label"].value_counts().sort_index()
    total = len(df)

    palette = ["#D4A017", "#B85C38", "#8C3B1F", "#6B7A2E", "#2F5D50"]
    colors = palette[:len(counts)]

    def label_with_count(pct):
        count = int(round(pct * total / 100.0))
        return f"{count}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index.astype(str),
        colors=colors,
        autopct=label_with_count,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
        pctdistance=0.72,
        labeldistance=1.05,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")

    ax.set_title(f"Label distribution (n={total})", fontsize=18, fontweight="bold", pad=20)
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output
