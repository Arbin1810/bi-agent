import pandas as pd
import numpy as np

def compute_trend(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    """Compute trend metrics for a time series."""
    s = df[[date_col, value_col]].dropna().sort_values(date_col)
    if len(s) < 2:
        return {"error": "Insufficient data"}

    y = s[value_col].values.astype(float)
    x = np.arange(len(y))

    # Linear regression slope
    slope = np.polyfit(x, y, 1)[0]

    # Percent change (first vs last)
    first, last = y[0], y[-1]
    pct = ((last - first) / first * 100) if first != 0 else None

    return {
        "periods": len(s),
        "start": str(s[date_col].iloc[0]),
        "end": str(s[date_col].iloc[-1]),
        "slope": float(slope),
        "pct_change": float(pct) if pct is not None else None,
        "direction": "up" if slope > 0 else "down" if slope < 0 else "flat",
    }