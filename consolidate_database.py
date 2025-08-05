#!/usr/bin/env python3
"""
Database Cleanup Utility
Consolidates legacy project entries into single project entries
"""

import pandas as pd
import os
from data_utils import load_db, add_entry_to_db, delete_entry
from model_utils import predict_document_probability
from upload_handler import generate_document_summary

def consolidate_legacy_projects():
    """
    Find and consolidate legacy project entries that were saved as individual files
    but belong to the same project.
    """
    df = load_db()
    if df.empty:
        print("No entries to consolidate")
        return
    
    # Find potential project groups by looking for entries with similar titles
    # that contain "RFP PROJECT:" in their summary
    project_groups = {}
    
    for idx, row in df.iterrows():
        if pd.notna(row['summary']) and "🎯 RFP PROJECT:" in row['summary']:
            # Extract project name from summary
            summary_lines = row['summary'].split('\n')
            project_line = summary_lines[0] if summary_lines else ""
            if "🎯 RFP PROJECT:" in project_line:
                project_name = project_line.replace("🎯 RFP PROJECT:", "").strip()
                
                if project_name not in project_groups:
                    project_groups[project_name] = []
                project_groups[project_name].append(row)
    
    # Process each project group
    consolidated_count = 0
    for project_name, entries in project_groups.items():
        if len(entries) > 1:  # Only consolidate if there are multiple entries
            print(f"Consolidating project: {project_name} ({len(entries)} files)")
            
            # Use the first entry as the base
            base_entry = entries[0]
            all_files = []
            
            # Collect all filenames from the entries
            for entry in entries:
                # Extract original filename from title or filename
                if " - " in entry['title']:
                    original_filename = entry['title'].split(" - ", 1)[1]
                else:
                    original_filename = entry['filename']
                all_files.append(original_filename)
            
            # Create consolidated entry
            project_filename = f"PROJECT_{project_name.replace(' ', '_')}"
            consolidated_entry = {
                "filename": project_filename,
                "title": project_name,
                "sender": base_entry['sender'],
                "decision": base_entry['decision'],
                "probability": base_entry['probability'],  # Keep the same probability
                "summary": base_entry['summary'],
                "file_list": ", ".join(all_files)
            }
            
            # Delete the old individual entries
            for entry in entries:
                delete_entry(entry['filename'])
            
            # Add the new consolidated entry
            add_entry_to_db(consolidated_entry)
            consolidated_count += 1
            print(f"✅ Consolidated {len(entries)} entries into: {project_filename}")
    
    if consolidated_count > 0:
        print(f"\n🎉 Successfully consolidated {consolidated_count} projects!")
    else:
        print("No legacy projects found to consolidate")

if __name__ == "__main__":
    print("🔧 Starting database consolidation...")
    consolidate_legacy_projects()
    print("✅ Database consolidation complete!")
