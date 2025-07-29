# Main RFP Analyzer Application
# Streamlined main application logic using modular components

import streamlit as st
import pandas as pd
import os
import re
from data_utils import load_db, update_probability
from document_utils import extract_text_from_file
from model_utils import predict_document_probability, manual_fine_tune, get_model_info

# Import our modular components
from ui_styles import UI_STYLES
from ui_components import render_header, render_empty_state
from rfp_counter import count_unique_rfps, count_unique_rfps_by_decision
from display_utils import display_project_entry, display_individual_entry
from upload_handler import handle_file_upload_form, process_uploaded_files

PDF_FOLDER = "documents"

def render_upload_tab():
    """Render the Upload & Process tab"""
    st.markdown("### Upload RFP Documents")
    st.markdown("Upload PDF, Word (DOCX), or Excel files to analyze their potential value..")
    
    # Upload area with better styling
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose document files",
        type=["pdf", "docx", "xlsx", "xls"], 
        accept_multiple_files=True,
        help="Select PDF, Word (DOCX), or Excel files to analyze. "
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        submitted, group_mode, titles, senders, decisions, project_title, project_sender, project_decision = handle_file_upload_form(uploaded_files)
        
        if submitted:
            success = process_uploaded_files(
                uploaded_files, group_mode, titles, senders, decisions,
                project_title, project_sender, project_decision
            )

def render_database_tab():
    """Render the RFP Database tab"""
    st.markdown("### RFP Database Overview")
    df = load_db()
    
    if df.empty:
        render_empty_state()
        return
    
    # Render metrics and controls
    render_database_metrics(df)
    
    # Get filter and sort options from controls
    filter_option, sort_option = render_database_controls(df)
    
    # Render entries with filtering and sorting
    render_database_entries(df, filter_option, sort_option)

def render_database_metrics(df):
    """Render the database metrics section"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unique_rfp_count = count_unique_rfps(df)
        st.metric("Total RFPs", unique_rfp_count)
    
    with col2:
        st.metric("Avg Score", f"{df['probability'].mean():.1%}")
    
    with col3:
        approved_count = count_unique_rfps_by_decision(df, 'Approved')
        st.metric("Approved", approved_count)
    
    with col4:
        csv = df.to_csv(index=False)
        st.download_button(
            "Export CSV",
            data=csv,
            file_name="rfp_database.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Model status info
    model_info = get_model_info()
    approved_count = model_info['approved_count']
    denied_count = model_info['denied_count']
    pending_count = count_unique_rfps_by_decision(df, 'Pending')
    has_finetuned = model_info['has_fine_tuned_model']
    
    # Model status display
    if has_finetuned:
        st.success(f"🎯 **Fine-tuned Model Active**: Trained on {approved_count} approved and {denied_count} denied RFPs. Model predictions are optimized for your decision patterns!")
    elif approved_count + denied_count > 0:
        st.info(f"📚 **Learning Mode**: Using similarity learning from {approved_count} approved and {denied_count} denied RFPs. Consider fine-tuning for better accuracy!")
    else:
        st.warning("🔄 **Initial Mode**: No historical decisions yet. Start approving/denying RFPs to improve prediction accuracy!")
    
    if pending_count > 0:
        st.info(f"⏳ {pending_count} pending RFPs will get more accurate scores as you make more decisions.")

def render_database_controls(df):
    """Render the database control section"""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        if st.button("Rerun Model for All Entries", use_container_width=True):
            handle_model_rerun(df)
    
    with col2:
        model_info = get_model_info()
        historical_count = model_info['historical_decisions']
        
        # Show fine-tuning button if we have enough data
        if historical_count >= 2:
            if st.button("🎯 Fine-tune Model", use_container_width=True, 
                        help=f"Train model on your {historical_count} decisions for better accuracy"):
                handle_fine_tuning()
        else:
            st.button("🎯 Fine-tune Model", use_container_width=True, 
                     disabled=True, help="Need at least 2 decisions to fine-tune")
    
    with col3:
        filter_option = st.selectbox(
            "Filter by Status",
            ["All", "Pending", "Approved", "Denied"],
            help="Filter RFPs by their current status"
        )
    
    with col4:
        sort_option = st.selectbox(
            "Sort by",
            ["Score (High to Low)", "Score (Low to High)", "Title", "Date Added"]
        )
    
    return filter_option, sort_option

def handle_model_rerun(df):
    """Handle the model rerun for all entries"""
    with st.spinner("Rerunning predictions with updated learning..."):
        rerun_count = 0
        for idx, row in df.iterrows():
            file_path = os.path.join(PDF_FOLDER, row['filename'])
            if os.path.exists(file_path):
                try:
                    text = extract_text_from_file(file_path)
                    old_prob = row['probability']
                    new_prob = predict_document_probability(text)
                    update_probability(row['filename'], new_prob)
                    
                    change = new_prob - old_prob
                    change_text = f"({change:+.1%})" if abs(change) > 0.01 else ""
                    st.success(f"Updated {row['filename']}: {new_prob:.2%} {change_text}")
                    rerun_count += 1
                except Exception as e:
                    st.error(f"Error updating {row['filename']}: {str(e)}")
            else:
                st.warning(f"File not found: {row['filename']}")
    
    if rerun_count > 0:
        st.success(f"Model rerun complete! Updated {rerun_count} entries with improved learning from your decisions.")
    st.rerun()

def handle_fine_tuning():
    """Handle the fine-tuning process"""
    with st.spinner("Fine-tuning model on your decisions... This may take a few minutes."):
        try:
            success = manual_fine_tune()
            if success:
                st.success("🎯 Model fine-tuning completed successfully! The model is now optimized for your decision patterns.")
                st.info("💡 Future predictions will be more accurate. Consider rerunning predictions for existing entries.")
            else:
                st.error("❌ Fine-tuning failed. Please check the logs for details.")
        except Exception as e:
            st.error(f"❌ Fine-tuning error: {str(e)}")
    st.rerun()

def render_database_entries(df, filter_option, sort_option):
    """Render the database entries with filtering and sorting"""
    filtered_df = df.copy()
    if filter_option != "All":
        filtered_df = filtered_df[filtered_df['decision'] == filter_option]
    
    if sort_option == "Score (High to Low)":
        filtered_df = filtered_df.sort_values("probability", ascending=False)
    elif sort_option == "Score (Low to High)":
        filtered_df = filtered_df.sort_values("probability", ascending=True)
    elif sort_option == "Title":
        filtered_df = filtered_df.sort_values("title")
    
    st.markdown("---")
    
    # Group project files together and show individual files separately
    displayed_projects = set()  # Track displayed projects to avoid duplicates
    
    for _, row in filtered_df.iterrows():
        # Check if this is part of a multi-file project
        is_project_file = "RFP Project:" in str(row['summary'])
        
        if is_project_file:
            # Extract project identifier from title (before the " - filename" part)
            if ' - ' in row['title']:
                project_title_base = row['title'].split(' - ')[0]
                project_key = f"{project_title_base}_{row['sender']}"
            else:
                project_title_base = row['title']
                project_key = f"{row['title']}_{row['sender']}"
            
            # Skip if we already displayed this project
            if project_key in displayed_projects:
                continue
            
            displayed_projects.add(project_key)
            
            # Find all files in this project - use more flexible matching
            if ' - ' in row['title']:
                # For projects with multiple files, find all files with the same base title and sender
                project_files = filtered_df[
                    (filtered_df['title'].str.startswith(f"{project_title_base} - ", na=False) | 
                     (filtered_df['title'] == project_title_base)) &
                    (filtered_df['sender'] == row['sender']) &
                    (filtered_df['summary'].str.contains("RFP Project:", na=False))
                ]
            else:
                # For single-file projects, just match by title and sender
                project_files = filtered_df[
                    (filtered_df['title'] == row['title']) &
                    (filtered_df['sender'] == row['sender']) &
                    (filtered_df['summary'].str.contains("RFP Project:", na=False))
                ]
            
            # Display project as a single entry
            display_project_entry(row, project_files, project_title_base)
        
        else:
            # Display individual file entry
            display_individual_entry(row)

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="RFP Analyzer",
        page_icon="RFP",
        layout="wide"
    )
    
    # Apply styles
    st.markdown(UI_STYLES, unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Upload & Process", "RFP Database"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_database_tab()

if __name__ == "__main__":
    main()
