"""Plotting helpers used by the notebooks."""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def _frame(ax, color, width=6):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=color, linewidth=width, clip_on=False))


def show_retrieval(query_paths, result_paths, result_scores, result_correct,
                   query_titles=None, result_titles=None, save_path=None):
    """One row per query: query image, then its top-k results.

    Green frame = same mark as the query (correct), red frame = different mark.
    """
    n_rows = len(query_paths)
    k = len(result_paths[0])
    fig, axes = plt.subplots(n_rows, k + 1, figsize=(2.1 * (k + 1), 2.6 * n_rows),
                             squeeze=False)
    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(Image.open(query_paths[r]))
        title = query_titles[r] if query_titles else "QUERY"
        ax.set_title(textwrap.fill(title, 26), fontsize=7, fontweight="bold")
        _frame(ax, "#1f4e79")
        ax.axis("off")
        for c in range(k):
            ax = axes[r, c + 1]
            ax.imshow(Image.open(result_paths[r][c]))
            label = f"#{c + 1}  {result_scores[r][c]:.3f}"
            if result_titles:
                label += f"\n{result_titles[r][c]}"
            ax.set_title(label, fontsize=7)
            _frame(ax, "#2e9e4f" if result_correct[r][c] else "#d1342f")
            ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()


def show_families(df, image_dir, parent_ids, n_cols=7, save_path=None):
    """Grid: each row = one parent logo (first column) followed by its variants."""
    fig, axes = plt.subplots(len(parent_ids), n_cols,
                             figsize=(1.9 * n_cols, 2.1 * len(parent_ids)), squeeze=False)
    for r, pid in enumerate(parent_ids):
        fam = df[df.parent_id == pid].sort_values("is_parent", ascending=False).head(n_cols)
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(fam):
                continue
            row = fam.iloc[c]
            ax.imshow(Image.open(image_dir / row.filename))
            title = "PARENT" if row.is_parent else str(row.variant_types).replace("+", "\n")
            ax.set_title(title, fontsize=7, fontweight="bold" if row.is_parent else "normal")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
