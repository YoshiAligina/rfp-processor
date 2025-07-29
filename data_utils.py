import pandas as pd
import os

CSV_DB = "rfp_db.csv"
CSV_COLUMNS = ["filename", "title", "sender", "decision", "probability", "summary"]

def load_db():
    if os.path.exists(CSV_DB):
        df = pd.read_csv(CSV_DB)
        # Migration: rename 'excerpt' column to 'summary' if it exists
        if 'excerpt' in df.columns and 'summary' not in df.columns:
            df = df.rename(columns={'excerpt': 'summary'})
            df.to_csv(CSV_DB, index=False)  # Save the migrated data
        # Ensure all required columns exist
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)

def add_entry_to_db(entry):
    df = load_db()
    new_df = pd.DataFrame([entry]).dropna(axis=1, how="all")
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_DB, index=False)

def update_decision(filename, new_decision):
    df = load_db()
    df.loc[df["filename"] == filename, "decision"] = new_decision
    df.to_csv(CSV_DB, index=False)

def delete_entry(filename):
    """Delete an entry from the database by filename"""
    df = load_db()
    df = df[df["filename"] != filename]
    df.to_csv(CSV_DB, index=False)

def update_probability(filename, new_probability):
    """Update the probability for a specific entry"""
    df = load_db()
    df.loc[df["filename"] == filename, "probability"] = new_probability
    df.to_csv(CSV_DB, index=False)
