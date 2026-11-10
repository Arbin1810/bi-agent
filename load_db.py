import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load your .env file to get the DATABASE_URL
load_dotenv()

database_url = os.getenv("DATABASE_URL", "sqlite:///./bi_agent.db")
print(f"Connecting to database: {database_url}")

# Connect to SQLite
engine = create_engine(database_url)

# Read the CSV
print("Reading transactions.csv...")
df = pd.read_csv("transactions.csv")

# Convert date column to string (SQLite likes strings for dates)
df['date'] = pd.to_datetime(df['date'])

# Push to SQLite
print("Loading data into SQLite... (This might take 5-10 seconds)")
df.to_sql("transactions", con=engine, if_exists="replace", index=False)

print("Data successfully loaded into table 'transactions'!")