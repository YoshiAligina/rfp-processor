# data_utils.py
import pandas as pd
import os

CSV_DB = "rfp_db.csv"
CSV_COLUMNS = ["filename", "title", "sender", "decision", "probability", "excerpt"]
# goal of load_db is to load the csv file into a pandas dataframe if it exists, otherwise create an empty dataframe with the correct columns
def load_db():
    if os.path.exists(CSV_DB):
        return pd.read_csv(CSV_DB)
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)
# this is to add a new entry to the csv file, it takes a dictionary as input
def add_entry_to_db(entry):
    df = load_db()
    new_df = pd.DataFrame([entry]).dropna(axis=1, how="all")  # Fix warning
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_DB, index=False)

def update_decision(filename, new_decision):
    df = load_db()
    df.loc[df["filename"] == filename, "decision"] = new_decision
    df.to_csv(CSV_DB, index=False)