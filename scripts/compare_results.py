"""Compare two versions of results/block_risk.csv and print a Markdown summary.
    python scripts/compare_results.py old_block_risk.csv results/block_risk.csv

Used by scripts/refresh_pipeline.sh after the model is re-run on new data. Only needs pandas.
"""
import sys

import pandas as pd

HIGH = 25  # % vacant that counts as a high-vacancy block, same as the notebooks
LEVELS = ["high", "medium", "low"]


def auc(score, positive):
    """Rank-based AUC (Mann-Whitney), no scikit-learn needed."""
    r = score.rank()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarize(path):
    d = pd.read_csv(path, dtype={"CVEGEO": str}).set_index("CVEGEO")
    high = d.target_pct_vacant >= HIGH
    top = d.risk_level == "high"
    return d, dict(blocks=len(d), auc=auc(d.risk_score, high), hit_rate_high=high[top].mean(),
                   median_vacancy=d.target_pct_vacant.median())


if __name__ == "__main__":
    old, new = (summarize(p) for p in sys.argv[1:3])
    print("| Metric | Before | After |\n|---|---|---|")
    for k in old[1]:
        print(f"| {k} | {old[1][k]:.3f} | {new[1][k]:.3f} |")

    both = old[0].index.intersection(new[0].index)
    print(f"\nBlocks only in the old file: {len(old[0]) - len(both)}; only in the new file: {len(new[0]) - len(both)}.")
    moved = pd.crosstab(old[0].loc[both, "risk_level"].rename("before"), new[0].loc[both, "risk_level"].rename("after"))
    moved = moved.reindex(index=LEVELS, columns=LEVELS, fill_value=0)
    print("\nRisk level before (rows) and after (columns):\n")
    try:
        print(moved.to_markdown())  # needs tabulate
    except ImportError:
        print("```\n" + moved.to_string() + "\n```")
    changed = (old[0].loc[both, "risk_level"] != new[0].loc[both, "risk_level"]).sum()
    print(f"\n{changed} of {len(both)} blocks changed risk level.")
