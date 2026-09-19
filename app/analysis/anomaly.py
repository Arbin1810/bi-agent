import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.0, iqr_mult: float = 1.5) -> dict:
    """Detect anomalies in numeric columns using z-score and IQR."""
    result = {"zscore": {}, "iqr": {}}
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 5:
            continue

        # Z-score
        mean, std = s.mean(), s.std()
        if std > 0:
            z = (s - mean) / std
            outliers = s[np.abs(z) > z_threshold]
            if len(outliers):
                result["zscore"][col] = {
                    "count": int(len(outliers)),
                    "values": outliers.head(10).tolist(),
                    "indices": outliers.index.tolist()[:10],
                }

        # IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        iqr_out = s[(s < low) | (s > high)]
        if len(iqr_out):
            result["iqr"][col] = {
                "count": int(len(iqr_out)),
                "lower_bound": float(low),
                "upper_bound": float(high),
                "values": iqr_out.head(10).tolist(),
            }

    return result