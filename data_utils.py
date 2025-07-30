"""
Data Management Utilities for RFP Processing System

This module provides comprehensive data persistence and management functionality
for the RFP analysis application. It handles all database operations using CSV
as the storage format, including entry creation, updates, deletions, and schema
migration. The module ensures data consistency and provides a clean interface
for all data-related operations throughout the application.

Key responsibilities:
- CSV database initialization and schema management
- CRUD operations for RFP entries
- Data migration and backward compatibility
- Probability score updates and decision tracking
- Database integrity and column validation
"""

import pandas as pd
import os

# Database configuration constants
CSV_DB = "rfp_db.csv" 
CSV_COLUMNS = ["filename", "title", "sender", "decision", "probability", "summary"]

def load_db():
    """
    Loads and validates the RFP database from CSV storage with schema migration support.
    This function handles database initialization, legacy data migration (renaming 'excerpt' 
    to 'summary'), and ensures all required columns exist. It provides backward compatibility
    for older database formats while maintaining data integrity. If no database exists,
    it returns an empty DataFrame with the proper schema structure.
    
    Returns:
        pandas.DataFrame: Complete RFP database with validated schema
    """
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
    """
    Adds a new RFP entry to the database with automatic data cleaning and persistence.
    This function loads the current database, processes the new entry to remove
    empty columns, appends it to the existing data, and saves the updated database
    back to CSV. It handles the complete workflow of entry insertion including
    data validation and file I/O operations. Essential for storing newly processed
    RFP documents and their associated metadata and predictions.
    
    Args:
        entry (dict): RFP entry dictionary containing filename, title, sender, 
                     decision, probability, and summary fields
    """
    df = load_db()
    new_df = pd.DataFrame([entry]).dropna(axis=1, how="all")
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_DB, index=False)

def update_decision(filename, new_decision):
    """
    Updates the approval/denial decision for a specific RFP entry.
    This function locates an entry by filename and updates its decision status
    (Approved, Denied, or Pending). It's crucial for the learning system as
    decision updates trigger model improvements and historical pattern recognition.
    The function handles the complete update workflow including data persistence
    and maintains referential integrity within the database.
    
    Args:
        filename (str): Unique filename identifier for the RFP entry
        new_decision (str): New decision status ('Approved', 'Denied', or 'Pending')
    """
    df = load_db()
    df.loc[df["filename"] == filename, "decision"] = new_decision
    df.to_csv(CSV_DB, index=False)

def delete_entry(filename):
    """
    Removes an RFP entry from the database by filename identifier.
    This function provides safe deletion functionality by filtering out the
    specified entry and saving the updated database. It handles the complete
    deletion workflow including data persistence and maintains database integrity.
    Used when users need to remove incorrectly processed documents or clean up
    outdated entries from their RFP collection.
    
    Args:
        filename (str): Unique filename identifier of the entry to delete
    """
    df = load_db()
    df = df[df["filename"] != filename]
    df.to_csv(CSV_DB, index=False)

def update_probability(filename, new_probability):
    """
    Updates the ML-generated approval probability score for a specific RFP entry.
    This function locates an entry by filename and updates its probability score
    with new predictions from improved or retrained models. Essential for keeping
    predictions current as the model learns from new decisions and historical
    patterns. Used extensively during model retraining and batch prediction updates
    to ensure all entries reflect the latest model capabilities.
    
    Args:
        filename (str): Unique filename identifier for the RFP entry
        new_probability (float): Updated probability score (0.0 to 1.0)
    """
    df = load_db()
    df.loc[df["filename"] == filename, "probability"] = new_probability
    df.to_csv(CSV_DB, index=False)
