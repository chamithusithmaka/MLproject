"""Run end-to-end model comparison from saved result CSV files."""

from pathlib import Path
from importlib import import_module
from typing import Any

from evaluation import compare_saved_model_metrics


def save_comparison_plot(df: Any, output_path: Path) -> None:
    """Save grouped bar chart for core metrics."""
    try:
        plt = import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        print("Warning: matplotlib is not installed. Skipping comparison plot export.")
        return

    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    plot_df = df[["model", *metrics]].copy()
    plot_df = plot_df.set_index("model")

    ax = plot_df.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title("Model Comparison Across Metrics")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"

    ranked = compare_saved_model_metrics(results_dir=results_dir, export=True)

    if not ranked.empty:
        best_row = ranked.iloc[0]
        print("\nBest model by F1 score:")
        print(
            f"  {best_row['model']} | "
            f"F1={best_row['f1_score']:.4f}, "
            f"AUC={best_row['auc_roc']:.4f}, "
            f"Accuracy={best_row['accuracy']:.4f}"
        )

    plot_path = results_dir / "model_comparison_plot.png"
    save_comparison_plot(ranked, plot_path)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
