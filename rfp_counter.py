# RFP Counter - Utilities for counting unique RFPs
# Handles the logic for counting projects vs individual files

def count_unique_rfps(df):
    """Count unique RFPs, treating project files as single entities"""
    if df.empty:
        return 0
    
    unique_rfps = set()
    
    for _, row in df.iterrows():
        # Check if this is part of a multi-file project
        is_project_file = "RFP Project:" in str(row['summary'])
        
        if is_project_file:
            # Extract project identifier from title (before the " - filename" part)
            if ' - ' in row['title']:
                project_key = f"{row['title'].split(' - ')[0]}_{row['sender']}_{row['probability']:.3f}"
            else:
                project_key = f"{row['title']}_{row['sender']}_{row['probability']:.3f}"
            unique_rfps.add(project_key)
        else:
            # Individual file - use full filename as unique identifier
            individual_key = f"individual_{row['filename']}_{row['sender']}"
            unique_rfps.add(individual_key)
    
    return len(unique_rfps)

def count_unique_rfps_by_decision(df, decision):
    """Count unique RFPs by decision, treating project files as single entities"""
    filtered_df = df[df['decision'] == decision]
    return count_unique_rfps(filtered_df)
