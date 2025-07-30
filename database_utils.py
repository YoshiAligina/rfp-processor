"""
Enhanced Database Management Utilities using SQLite

This module provides robust data persistence using SQLite database instead of CSV.
Offers better data integrity, concurrent access, transactions, and performance.
"""

import sqlite3
import pandas as pd
import os
from contextlib import contextmanager
from datetime import datetime
import logging

# Database configuration
DB_FILE = "rfp_database.db"

class RFPDatabase:
    """Enhanced database manager with SQLite backend for better data safety and performance."""
    
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize database with proper schema and indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create main RFP table with proper constraints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rfp_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    decision TEXT DEFAULT 'Pending' CHECK (decision IN ('Pending', 'Approved', 'Denied')),
                    probability REAL DEFAULT 0.0 CHECK (probability >= 0.0 AND probability <= 1.0),
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON rfp_entries(filename)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision ON rfp_entries(decision)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_probability ON rfp_entries(probability)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON rfp_entries(created_at)")
            
            # Create audit log table for tracking changes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    record_id INTEGER,
                    old_values TEXT,
                    new_values TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def load_all_entries(self):
        """Load all RFP entries as pandas DataFrame."""
        with self.get_connection() as conn:
            query = """
                SELECT filename, title, sender, decision, probability, summary, 
                       created_at, updated_at
                FROM rfp_entries 
                ORDER BY created_at DESC
            """
            return pd.read_sql_query(query, conn)
    
    def add_entry(self, entry):
        """Add new RFP entry with data validation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO rfp_entries (filename, title, sender, decision, probability, summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entry['filename'],
                    entry['title'], 
                    entry['sender'],
                    entry.get('decision', 'Pending'),
                    entry.get('probability', 0.0),
                    entry.get('summary', '')
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    raise ValueError(f"Entry with filename '{entry['filename']}' already exists")
                raise
    
    def update_decision(self, filename, new_decision):
        """Update decision with audit logging."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current values for audit
            cursor.execute("SELECT decision FROM rfp_entries WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No entry found with filename: {filename}")
            
            old_decision = row['decision']
            
            # Update the decision
            cursor.execute("""
                UPDATE rfp_entries 
                SET decision = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE filename = ?
            """, (new_decision, filename))
            
            if cursor.rowcount == 0:
                raise ValueError(f"No entry found with filename: {filename}")
            
            # Log the change
            cursor.execute("""
                INSERT INTO audit_log (action, table_name, old_values, new_values)
                VALUES (?, ?, ?, ?)
            """, ('UPDATE_DECISION', 'rfp_entries', old_decision, new_decision))
            
            conn.commit()
    
    def update_probability(self, filename, new_probability):
        """Update probability score with validation."""
        if not (0.0 <= new_probability <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE rfp_entries 
                SET probability = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE filename = ?
            """, (new_probability, filename))
            
            if cursor.rowcount == 0:
                raise ValueError(f"No entry found with filename: {filename}")
            
            conn.commit()
    
    def delete_entry(self, filename):
        """Delete entry with audit logging."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get entry data for audit before deletion
            cursor.execute("SELECT * FROM rfp_entries WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No entry found with filename: {filename}")
            
            # Delete the entry
            cursor.execute("DELETE FROM rfp_entries WHERE filename = ?", (filename,))
            
            # Log the deletion
            cursor.execute("""
                INSERT INTO audit_log (action, table_name, old_values)
                VALUES (?, ?, ?)
            """, ('DELETE', 'rfp_entries', str(dict(row))))
            
            conn.commit()
    
    def get_entry(self, filename):
        """Get single entry by filename."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rfp_entries WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_entries_by_decision(self, decision):
        """Get entries filtered by decision status."""
        with self.get_connection() as conn:
            query = """
                SELECT filename, title, sender, decision, probability, summary, 
                       created_at, updated_at
                FROM rfp_entries 
                WHERE decision = ?
                ORDER BY created_at DESC
            """
            return pd.read_sql_query(query, conn, params=[decision])
    
    def get_statistics(self):
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total entries
            cursor.execute("SELECT COUNT(*) as total FROM rfp_entries")
            stats['total'] = cursor.fetchone()['total']
            
            # Counts by decision
            cursor.execute("""
                SELECT decision, COUNT(*) as count 
                FROM rfp_entries 
                GROUP BY decision
            """)
            for row in cursor.fetchall():
                stats[f"{row['decision'].lower()}_count"] = row['count']
            
            # Average probability
            cursor.execute("SELECT AVG(probability) as avg_prob FROM rfp_entries")
            avg_prob = cursor.fetchone()['avg_prob']
            stats['avg_probability'] = avg_prob if avg_prob else 0.0
            
            return stats
    
    def backup_to_csv(self, filename="rfp_backup.csv"):
        """Export data to CSV for backup."""
        df = self.load_all_entries()
        df.to_csv(filename, index=False)
        return filename
    
    def migrate_from_csv(self, csv_file="rfp_db.csv"):
        """Migrate existing CSV data to SQLite database."""
        if not os.path.exists(csv_file):
            return False
        
        try:
            df = pd.read_csv(csv_file)
            
            # Rename columns if needed for backward compatibility
            if 'excerpt' in df.columns and 'summary' not in df.columns:
                df = df.rename(columns={'excerpt': 'summary'})
            
            # Ensure required columns exist
            required_cols = ['filename', 'title', 'sender', 'decision', 'probability', 'summary']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = '' if col in ['title', 'sender', 'summary'] else ('Pending' if col == 'decision' else 0.0)
            
            # Insert data
            with self.get_connection() as conn:
                for _, row in df.iterrows():
                    try:
                        self.add_entry({
                            'filename': row['filename'],
                            'title': row['title'],
                            'sender': row['sender'],
                            'decision': row['decision'],
                            'probability': row['probability'],
                            'summary': row['summary']
                        })
                    except ValueError:  # Skip duplicates
                        continue
            
            return True
            
        except Exception as e:
            logging.error(f"Migration error: {e}")
            return False

# Global database instance
_db_instance = None

def get_database():
    """Get global database instance (singleton pattern)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = RFPDatabase()
    return _db_instance

# Compatibility functions to match existing interface
def load_db():
    """Load database entries as DataFrame (compatible with existing code)."""
    return get_database().load_all_entries()

def add_entry_to_db(entry):
    """Add entry to database (compatible with existing code)."""
    return get_database().add_entry(entry)

def update_decision(filename, new_decision):
    """Update decision (compatible with existing code)."""
    return get_database().update_decision(filename, new_decision)

def update_probability(filename, new_probability):
    """Update probability (compatible with existing code)."""
    return get_database().update_probability(filename, new_probability)

def delete_entry(filename):
    """Delete entry (compatible with existing code)."""
    return get_database().delete_entry(filename)
