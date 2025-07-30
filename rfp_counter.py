# RFP Counter - Utilities for counting unique RFPs
# Handles the logic for counting projects vs individual files

"""
RFP Counting Utilities for Database Metrics and Analytics

This module provides specialized counting functions that handle the complexity
of distinguishing between individual RFP documents and grouped project submissions.
It ensures accurate metrics by treating multi-file projects as single entities
while properly counting standalone documents. Essential for dashboard metrics
and analytics that need to reflect true RFP counts rather than raw file counts.

Key features:
- Intelligent project vs. individual document detection
- Unique RFP identification using composite keys
- Decision-based filtering and counting
- Consistent metrics across grouped and individual submissions
"""

def count_unique_rfps(df):
    """
    Counts unique RFP submissions, properly handling both individual documents and grouped projects.
    This function analyzes the database to distinguish between standalone RFP documents
    and multi-file project submissions, ensuring that grouped projects are counted as
    single entities rather than multiple separate RFPs. It uses intelligent key generation
    based on document metadata to identify unique submissions across different processing
    modes, providing accurate metrics for dashboard displays and analytics.
    
    Args:
        df (pandas.DataFrame): RFP database containing all processed documents
        
    Returns:
        int: Count of unique RFP submissions (projects counted as single entities)
    """
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
    """
    Counts unique RFP submissions filtered by specific decision status.
    This function extends the unique counting functionality to provide metrics
    for specific decision categories (Approved, Denied, Pending). It filters
    the database by decision status and then applies the same intelligent
    project grouping logic to ensure accurate counts. Essential for dashboard
    metrics that show approval rates and decision distribution while maintaining
    proper distinction between individual documents and project groups.
    
    Args:
        df (pandas.DataFrame): RFP database containing all processed documents
        decision (str): Decision status to filter by ('Approved', 'Denied', 'Pending')
        
    Returns:
        int: Count of unique RFP submissions with the specified decision status
    """
    filtered_df = df[df['decision'] == decision]
    return count_unique_rfps(filtered_df)
