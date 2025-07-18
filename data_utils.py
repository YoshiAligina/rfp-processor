import pandas as pd
import os

CSV_DB = "rfp_db.csv"
CSV_COLUMNS = ["filename", "title", "sender", "decision", "probability", "excerpt"]

def load_db():
    if os.path.exists(CSV_DB):
        return pd.read_csv(CSV_DB)
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)

def add_entry_to_db(entry):
    df = load_db()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(CSV_DB, index=False)
