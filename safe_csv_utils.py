"""
Enhanced CSV Data Management with Safety Features

If you prefer to stick with CSV but want better safety, this module provides:
- Automatic backups before modifications
- Data validation and schema enforcement  
- Atomic operations with rollback capability
- Concurrent access protection
"""

import pandas as pd
import os
import shutil
import json
import fcntl  # File locking (Unix/Linux) - use msvcrt on Windows
import time
from datetime import datetime
from contextlib import contextmanager
import tempfile

# Database configuration
CSV_DB = "rfp_db.csv"
BACKUP_DIR = "backups"
SCHEMA_FILE = "db_schema.json"

# Schema definition for validation
SCHEMA = {
    "filename": {"type": "string", "required": True, "unique": True},
    "title": {"type": "string", "required": True},
    "sender": {"type": "string", "required": True},
    "decision": {"type": "string", "required": False, "allowed": ["Pending", "Approved", "Denied"]},
    "probability": {"type": "float", "required": False, "min": 0.0, "max": 1.0},
    "summary": {"type": "string", "required": False}
}

class SafeCSVDatabase:
    """Enhanced CSV database with safety features."""
    
    def __init__(self, csv_file=CSV_DB):
        self.csv_file = csv_file
        self.backup_dir = BACKUP_DIR
        self.schema = SCHEMA
        self._ensure_backup_dir()
        self._ensure_schema_file()
    
    def _ensure_backup_dir(self):
        """Ensure backup directory exists."""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
    
    def _ensure_schema_file(self):
        """Create schema file if it doesn't exist."""
        if not os.path.exists(SCHEMA_FILE):
            with open(SCHEMA_FILE, 'w') as f:
                json.dump(self.schema, f, indent=2)
    
    def _create_backup(self):
        """Create timestamped backup of current database."""
        if os.path.exists(self.csv_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"rfp_db_backup_{timestamp}.csv"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copy2(self.csv_file, backup_path)
            
            # Keep only last 10 backups
            backups = [f for f in os.listdir(self.backup_dir) if f.startswith("rfp_db_backup_")]
            backups.sort()
            while len(backups) > 10:
                oldest = backups.pop(0)
                os.remove(os.path.join(self.backup_dir, oldest))
            
            return backup_path
        return None
    
    def _validate_entry(self, entry):
        """Validate entry against schema."""
        errors = []
        
        for field, rules in self.schema.items():
            value = entry.get(field)
            
            # Check required fields
            if rules.get("required", False) and (value is None or str(value).strip() == ""):
                errors.append(f"Field '{field}' is required")
                continue
            
            # Skip validation for empty optional fields
            if value is None or str(value).strip() == "":
                continue
            
            # Type validation
            expected_type = rules.get("type")
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{field}' must be a string")
            elif expected_type == "float":
                try:
                    float_val = float(value)
                    # Range validation
                    if "min" in rules and float_val < rules["min"]:
                        errors.append(f"Field '{field}' must be >= {rules['min']}")
                    if "max" in rules and float_val > rules["max"]:
                        errors.append(f"Field '{field}' must be <= {rules['max']}")
                except (ValueError, TypeError):
                    errors.append(f"Field '{field}' must be a number")
            
            # Allowed values validation
            if "allowed" in rules and value not in rules["allowed"]:
                errors.append(f"Field '{field}' must be one of: {rules['allowed']}")
        
        return errors
    
    @contextmanager
    def _file_lock(self, timeout=10):
        """Context manager for file locking (prevents concurrent access issues)."""
        lock_file = f"{self.csv_file}.lock"
        
        # Wait for lock to be available
        start_time = time.time()
        while os.path.exists(lock_file):
            if time.time() - start_time > timeout:
                raise TimeoutError("Could not acquire file lock")
            time.sleep(0.1)
        
        # Create lock file
        try:
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            yield
        finally:
            # Remove lock file
            if os.path.exists(lock_file):
                os.remove(lock_file)
    
    def _atomic_write(self, df):
        """Atomic write operation - write to temp file first, then move."""
        temp_file = f"{self.csv_file}.tmp"
        try:
            df.to_csv(temp_file, index=False)
            # Atomic move (works on most filesystems)
            if os.path.exists(self.csv_file):
                os.replace(temp_file, self.csv_file)
            else:
                os.rename(temp_file, self.csv_file)
        except Exception:
            # Cleanup temp file on error
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
    
    def load_db(self):
        """Load database with validation and migration."""
        if os.path.exists(self.csv_file):
            try:
                df = pd.read_csv(self.csv_file)
                
                # Migration: rename 'excerpt' to 'summary'
                if 'excerpt' in df.columns and 'summary' not in df.columns:
                    df = df.rename(columns={'excerpt': 'summary'})
                    self._create_backup()
                    self._atomic_write(df)
                
                # Ensure all required columns exist
                for field in self.schema.keys():
                    if field not in df.columns:
                        if field == 'decision':
                            df[field] = 'Pending'
                        elif field == 'probability':
                            df[field] = 0.0
                        else:
                            df[field] = ''
                
                return df
            except Exception as e:
                print(f"Error loading database: {e}")
                return pd.DataFrame(columns=list(self.schema.keys()))
        else:
            return pd.DataFrame(columns=list(self.schema.keys()))
    
    def add_entry(self, entry):
        """Add entry with validation and backup."""
        # Validate entry
        errors = self._validate_entry(entry)
        if errors:
            raise ValueError(f"Validation errors: {'; '.join(errors)}")
        
        with self._file_lock():
            # Create backup
            self._create_backup()
            
            # Load current data
            df = self.load_db()
            
            # Check for duplicates
            if entry['filename'] in df['filename'].values:
                raise ValueError(f"Entry with filename '{entry['filename']}' already exists")
            
            # Add new entry
            new_df = pd.DataFrame([entry])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Atomic write
            self._atomic_write(df)
    
    def update_decision(self, filename, new_decision):
        """Update decision with validation and backup."""
        if new_decision not in self.schema['decision']['allowed']:
            raise ValueError(f"Decision must be one of: {self.schema['decision']['allowed']}")
        
        with self._file_lock():
            self._create_backup()
            df = self.load_db()
            
            if filename not in df['filename'].values:
                raise ValueError(f"No entry found with filename: {filename}")
            
            df.loc[df['filename'] == filename, 'decision'] = new_decision
            self._atomic_write(df)
    
    def update_probability(self, filename, new_probability):
        """Update probability with validation and backup."""
        try:
            prob_float = float(new_probability)
            if not (0.0 <= prob_float <= 1.0):
                raise ValueError("Probability must be between 0.0 and 1.0")
        except (ValueError, TypeError):
            raise ValueError("Probability must be a valid number")
        
        with self._file_lock():
            self._create_backup()
            df = self.load_db()
            
            if filename not in df['filename'].values:
                raise ValueError(f"No entry found with filename: {filename}")
            
            df.loc[df['filename'] == filename, 'probability'] = prob_float
            self._atomic_write(df)
    
    def delete_entry(self, filename):
        """Delete entry with backup."""
        with self._file_lock():
            self._create_backup()
            df = self.load_db()
            
            if filename not in df['filename'].values:
                raise ValueError(f"No entry found with filename: {filename}")
            
            df = df[df['filename'] != filename]
            self._atomic_write(df)
    
    def get_backup_list(self):
        """Get list of available backups."""
        if not os.path.exists(self.backup_dir):
            return []
        
        backups = [f for f in os.listdir(self.backup_dir) if f.startswith("rfp_db_backup_")]
        backups.sort(reverse=True)  # Most recent first
        return backups
    
    def restore_from_backup(self, backup_name):
        """Restore database from backup."""
        backup_path = os.path.join(self.backup_dir, backup_name)
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup file not found: {backup_name}")
        
        with self._file_lock():
            # Create backup of current state before restoring
            self._create_backup()
            shutil.copy2(backup_path, self.csv_file)

# Global database instance
_safe_db = None

def get_safe_database():
    """Get safe database instance."""
    global _safe_db
    if _safe_db is None:
        _safe_db = SafeCSVDatabase()
    return _safe_db

# Compatibility functions
def load_db():
    return get_safe_database().load_db()

def add_entry_to_db(entry):
    return get_safe_database().add_entry(entry)

def update_decision(filename, new_decision):
    return get_safe_database().update_decision(filename, new_decision)

def update_probability(filename, new_probability):
    return get_safe_database().update_probability(filename, new_probability)

def delete_entry(filename):
    return get_safe_database().delete_entry(filename)
