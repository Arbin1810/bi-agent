from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import settings
from app.utils.logger import logger

def _save(fig, prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = settings.charts_dir / f"{prefix}_{ts}.png"
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return str(path)

def generate_charts(df: pd.DataFrame, question: str = "") -> list[dict]:
    charts = []
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(include=["object", "category"])
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])

    # Try parse object columns as dates
    if datetime_cols.empty:
        for c in categorical.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df = df.copy()
                    df[c] = parsed
                    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])
                    break
            except Exception:
                pass

    # Time series chart
    if not datetime_cols.empty and not numeric.empty:
        dcol = datetime_cols.columns[0]
        vcol = numeric.columns[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        s = df[[dcol, vcol]].dropna().sort_values(dcol)
        ax.plot(s[dcol], s[vcol], marker="o", linewidth=1.5)
        ax.set_title(f"{vcol} over {dcol}")
        ax.tick_params(axis="x", rotation=45)
        charts.append({"type": "timeseries", "path": _save(fig, "ts")})

    # Bar chart for top categories
    if not categorical.empty and not numeric.empty:
        ccol, vcol = categorical.columns[0], numeric.columns[0]
        top = df.groupby(ccol)[vcol].sum().sort_values(ascending=False).head(10)
        if len(top) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            top.plot(kind="bar", ax=ax)
            ax.set_title(f"Top 10 {ccol} by {vcol}")
            charts.append({"type": "bar", "path": _save(fig, "bar")})

    # Distribution
    if not numeric.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[numeric.columns[0]].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {numeric.columns[0]}")
        charts.append({"type": "histogram", "path": _save(fig, "hist")})

    # Correlation heatmap
    if len(numeric.columns) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        charts.append({"type": "heatmap", "path": _save(fig, "corr")})

    logger.info(f"Generated {len(charts)} charts")
    return charts