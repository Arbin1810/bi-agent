import pandas as pd
from app.orchestration.workflow import workflow

df = pd.read_csv("data/transactions.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} cols")

state = workflow.run(
    question="Why did revenue decline in August 2026?",
    dataframe=df,
)

print("\n--- INTENT ---")
print(state.get("intent"))

print("\n--- ANALYSIS ---")
print(state.get("analysis"))

print("\n--- ANOMALIES ---")
print(state.get("anomalies"))

print("\n--- ROOT CAUSES ---")
print(state.get("root_causes"))

print("\n--- REPORT PATH ---")
print(state.get("report_path"))

print("\n--- TRACE ---")
print(state.get("trace"))