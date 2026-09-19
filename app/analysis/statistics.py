import pandas as pd
import numpy as np

def summarize_dataframe(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "columns": list(df.columns)}

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
    }

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        summary["numeric_stats"] = {
            c: {
                "mean": float(numeric[c].mean()) if not numeric[c].isna().all() else None,
                "std": float(numeric[c].std()) if not numeric[c].isna().all() else None,
                "min": float(numeric[c].min()) if not numeric[c].isna().all() else None,
                "max": float(numeric[c].max()) if not numeric[c].isna().all() else None,
            }
            for c in numeric.columns
        }

    categorical = df.select_dtypes(include=["object", "category"])
    if not categorical.empty:
        summary["categorical_top"] = {
            c: categorical[c].value_counts().head(5).to_dict()
            for c in categorical.columns[:5]
        }

    return summary