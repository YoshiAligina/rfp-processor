import streamlit as st
import pandas as pd
import os
import re
from ocr_utils import is_scanned, extract_text, convert_scanned
from model_utils import predict_document_probability
from data_utils import add_entry_to_db, load_db, update_decision, delete_entry, update_probability
from document_utils import extract_text_from_file, get_file_type, get_appropriate_mime_type

PDF_FOLDER = "documents"

def generate_document_summary(text, max_sentences=3):

    if not text or len(text.strip()) < 100:
        return "Document too short to summarize effectively."
    
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= max_sentences:
        return ' '.join(sentences[:max_sentences]) + '.'
    
    # Simple scoring based on length and position (earlier sentences often more important)
    scored_sentences = []
    for i, sentence in enumerate(sentences[:10]):  # Only consider first 10 sentences
        # Score based on length (prefer medium-length sentences) and position (prefer earlier)
        length_score = min(len(sentence) / 100, 1.0)  # Normalize to 0-1
        position_score = (10 - i) / 10  # Earlier sentences get higher score
        
        # Look for key RFP-related terms
        key_terms = ['request', 'proposal', 'requirements', 'services', 'project', 'deadline', 'budget', 'scope']
        term_score = sum(1 for term in key_terms if term.lower() in sentence.lower()) / len(key_terms)
        
        total_score = length_score * 0.3 + position_score * 0.4 + term_score * 0.3
        scored_sentences.append((sentence, total_score))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    summary_sentences = [s[0] for s in scored_sentences[:max_sentences]]
    
    original_order = []
    for sentence in sentences:
        if sentence in summary_sentences:
            original_order.append(sentence)
            if len(original_order) == max_sentences:
                break
    
    return ' '.join(original_order) + '.'

def _count_unique_rfps(df):
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

def _count_unique_rfps_by_decision(df, decision):
    """Count unique RFPs by decision, treating project files as single entities"""
    filtered_df = df[df['decision'] == decision]
    return _count_unique_rfps(filtered_df)

def _display_project_entry(representative_row, project_files, project_title):
    """Display a single entry for a grouped project"""
    # Create a card-like layout
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        
        with col1:
            if representative_row['decision'] == 'Approved':
                badge_class = 'success-badge'
            elif representative_row['decision'] == 'Pending':
                badge_class = 'pending-badge'
            else:
                badge_class = 'denied-badge'
            
            # Show project indicator
            project_indicator = f"📁 **Project** ({len(project_files)} files) "
            
            st.markdown(f"""
            {project_indicator}**{project_title}**  
            <span class="{badge_class}">{representative_row['decision']}</span>  
            From: {representative_row['sender']}
            """, unsafe_allow_html=True)
        
        with col2:
            prob_color = "#4A7637" if representative_row['probability'] > 0.7 else "#B9C930" if representative_row['probability'] > 0.4 else "#8b5a3c"
            st.markdown(f"**Score:** <span style='color: {prob_color}'>{representative_row['probability']:.1%}</span>", unsafe_allow_html=True)
            st.markdown(f"<small>Combined analysis</small>", unsafe_allow_html=True)
        
        with col3:
            project_key = f"project_{representative_row['sender']}_{representative_row['probability']:.3f}"
            if st.button("View", key=f"view_{project_key}", help="View project details"):
                st.session_state[f"show_details_{project_key}"] = not st.session_state.get(f"show_details_{project_key}", False)
        
        with col4:
            # Create a zip download for all project files
            if st.button("Download All", key=f"download_{project_key}", help="Download all project files"):
                st.info("Individual file downloads available in project details below")
    
    # Show details if requested
    if st.session_state.get(f"show_details_{project_key}", False):
        with st.expander("Project Details", expanded=True):
            st.markdown("**📁 Project Files:**")
            
            # Display all files in the project
            for idx, (_, pf) in enumerate(project_files.iterrows()):
                file_exists = os.path.exists(os.path.join(PDF_FOLDER, pf['filename']))
                status_icon = "✅" if file_exists else "❌"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{status_icon} **{pf['filename']}**")
                with col2:
                    if file_exists:
                        file_path = os.path.join(PDF_FOLDER, pf['filename'])
                        with open(file_path, "rb") as f:
                            mime_type = get_appropriate_mime_type(pf['filename'])
                            st.download_button(
                                "Download",
                                data=f.read(),
                                file_name=pf['filename'],
                                mime=mime_type,
                                key=f"download_file_{pf['filename']}_{idx}",
                                help="Download file"
                            )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Project Summary:**")
                st.text_area("", value=representative_row['summary'], height=150, disabled=True, key=f"summary_{project_key}")
                
                current_decision = representative_row['decision']
                new_decision = st.selectbox(
                    "Update Project Status",
                    ["Pending", "Approved", "Denied"],
                    index=["Pending", "Approved", "Denied"].index(current_decision),
                    key=f"decision_update_{project_key}",
                    help="This will update the status for all files in the project"
                )
                
                if new_decision != current_decision:
                    if st.button(f"Save Status Change for All Files", key=f"save_decision_{project_key}"):
                        # Update all files in the project
                        for _, pf in project_files.iterrows():
                            update_decision(pf['filename'], new_decision)
                        st.success(f"Status updated to '{new_decision}' for all {len(project_files)} files")
                        st.rerun()
            
            with col2:
                st.markdown("**Project Actions:**")
                
                # Rerun analysis for entire project
                if st.button("Rerun Project Analysis", key=f"rerun_{project_key}"):
                    with st.spinner(f"Reanalyzing project with updated learning..."):
                        updated_count = 0
                        for _, pf in project_files.iterrows():
                            file_path = os.path.join(PDF_FOLDER, pf['filename'])
                            if os.path.exists(file_path):
                                try:
                                    text = extract_text_from_file(file_path)
                                    old_prob = pf['probability']
                                    new_prob = predict_document_probability(text)
                                    update_probability(pf['filename'], new_prob)
                                    updated_count += 1
                                except Exception as e:
                                    st.error(f"Error updating {pf['filename']}: {str(e)}")
                        
                        if updated_count > 0:
                            st.success(f"Updated analysis for {updated_count} files in project")
                            st.rerun()
                
                # Delete entire project
                if st.button("Delete Entire Project", key=f"delete_{project_key}"):
                    if st.session_state.get(f"confirm_delete_{project_key}", False):
                        # Delete all files in project
                        for _, pf in project_files.iterrows():
                            delete_entry(pf['filename'])
                            file_path = os.path.join(PDF_FOLDER, pf['filename'])
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        st.success(f"Deleted entire project ({len(project_files)} files)")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{project_key}"] = True
                        st.warning(f"Click again to confirm deletion of all {len(project_files)} files")
                
                if st.session_state.get(f"confirm_delete_{project_key}", False):
                    if st.button("Cancel Delete", key=f"cancel_delete_{project_key}"):
                        st.session_state[f"confirm_delete_{project_key}"] = False
                        st.rerun()

def _display_individual_entry(row):
    """Display a single entry for an individual file"""
    # Create a card-like layout
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        
        with col1:
            if row['decision'] == 'Approved':
                badge_class = 'success-badge'
            elif row['decision'] == 'Pending':
                badge_class = 'pending-badge'
            else:
                badge_class = 'denied-badge'
            
            st.markdown(f"""
            **{row['title']}**  
            <span class="{badge_class}">{row['decision']}</span>  
            From: {row['sender']}
            """, unsafe_allow_html=True)
        
        with col2:
            prob_color = "#4A7637" if row['probability'] > 0.7 else "#B9C930" if row['probability'] > 0.4 else "#8b5a3c"
            st.markdown(f"**Score:** <span style='color: {prob_color}'>{row['probability']:.1%}</span>", unsafe_allow_html=True)
        
        with col3:
            if st.button("View", key=f"view_{row['filename']}", help="View details"):
                st.session_state[f"show_details_{row['filename']}"] = not st.session_state.get(f"show_details_{row['filename']}", False)
        
        with col4:
            file_path = os.path.join(PDF_FOLDER, row['filename'])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    mime_type = get_appropriate_mime_type(row['filename'])
                    st.download_button(
                        "Download",
                        data=f.read(),
                        file_name=row['filename'],
                        mime=mime_type,
                        key=f"download_{row['filename']}",
                        help="Download file"
                    )
    
    # Show details if requested
    if st.session_state.get(f"show_details_{row['filename']}", False):
        with st.expander("Document Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Document Summary:**")
                st.text_area("", value=row['summary'], height=150, disabled=True, key=f"summary_{row['filename']}")
                
                current_decision = row['decision']
                new_decision = st.selectbox(
                    "Update Status",
                    ["Pending", "Approved", "Denied"],
                    index=["Pending", "Approved", "Denied"].index(current_decision),
                    key=f"decision_update_{row['filename']}"
                )
                
                if new_decision != current_decision:
                    if st.button(f"Save Status Change", key=f"save_decision_{row['filename']}"):
                        update_decision(row['filename'], new_decision)
                        st.success(f"Status updated to '{new_decision}'")
                        st.rerun()
            
            with col2:
                st.markdown("**Actions:**")
                
                # Individual rerun model button
                if st.button("Rerun Analysis", key=f"rerun_{row['filename']}"):
                    file_path = os.path.join(PDF_FOLDER, row['filename'])
                    if os.path.exists(file_path):
                        try:
                            with st.spinner(f"Reanalyzing {row['filename']} with updated learning..."):
                                text = extract_text_from_file(file_path)
                                old_prob = row['probability']
                                new_prob = predict_document_probability(text)
                                update_probability(row['filename'], new_prob)
                                
                                change = new_prob - old_prob
                                if abs(change) > 0.01:
                                    change_text = f" (changed {change:+.1%})"
                                    st.success(f"Updated score: {new_prob:.2%}{change_text}")
                                else:
                                    st.success(f"Updated score: {new_prob:.2%} (no significant change)")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error rerunning analysis: {str(e)}")
                    else:
                        st.error("File not found")
                
                # Delete button with confirmation
                if st.button("Delete Entry", key=f"delete_{row['filename']}"):
                    if st.session_state.get(f"confirm_delete_{row['filename']}", False):
                        delete_entry(row['filename'])
                        file_path = os.path.join(PDF_FOLDER, row['filename'])
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        st.success(f"Deleted {row['filename']}")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{row['filename']}"] = True
                        st.warning("Click again to confirm deletion")
                
                if st.session_state.get(f"confirm_delete_{row['filename']}", False):
                    if st.button("Cancel Delete", key=f"cancel_delete_{row['filename']}"):
                        st.session_state[f"confirm_delete_{row['filename']}"] = False
                        st.rerun()

def main():
    # Page configuration
    st.set_page_config(
        page_title="RFP Analyzer",
        page_icon="RFP",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    /* Main background and body styling */
    .stApp {
        background-color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(90deg, #4A7637 0%, #B9C930 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(74, 118, 55, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(74, 118, 55, 0.1);
        border-left: 4px solid #4A7637;
        border: 1px solid #e8e8e8;
    }
    
    .upload-area {
        border: 2px dashed #B9C930;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #ffffff;
        box-shadow: 0 2px 4px rgba(185, 201, 48, 0.1);
    }
    
    .success-badge {
        background: #4A7637;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .pending-badge {
        background: #B9C930;
        color: #2d4a1f;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .denied-badge {
        background: #8b5a3c;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Button styling with maximum specificity */
    .stButton > button, 
    .stButton button,
    button[data-testid="stButton"],
    div[data-testid="stButton"] button,
    .stForm button[type="submit"],
    button[kind="secondary"],
    button[kind="primary"] {
        background-color: #4A7637 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    
    .stButton > button:hover,
    .stButton button:hover,
    button[data-testid="stButton"]:hover,
    div[data-testid="stButton"] button:hover,
    .stForm button[type="submit"]:hover,
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover {
        background-color: #3a5e2b !important;
        box-shadow: 0 2px 4px rgba(74, 118, 55, 0.3) !important;
        color: white !important;
    }
    
    /* Primary button specific styling */
    .stButton > button[kind="primary"],
    button[kind="primary"],
    .stForm button[type="submit"] {
        background-color: #B9C930 !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover,
    .stForm button[type="submit"]:hover {
        background-color: #a6b82a !important;
        box-shadow: 0 2px 4px rgba(185, 201, 48, 0.3) !important;
        color: white !important;
    }
    
    /* Universal button text color override */
    button, 
    button *,
    .stButton button,
    .stButton button *,
    button[data-testid="stButton"],
    button[data-testid="stButton"] *,
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button * {
        color: white !important;
    }
    
    /* Form elements styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        color: #2d4a1f;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #B9C930;
        box-shadow: 0 0 0 1px #B9C930;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        color: #2d4a1f;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #B9C930;
        box-shadow: 0 0 0 1px #B9C930;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        border: 1px solid #d0d0d0 !important;
        color: #000000 !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #B9C930 !important;
        box-shadow: 0 0 0 1px #B9C930 !important;
        color: #000000 !important;
    }
    
    /* Ensure disabled text areas show black text */
    .stTextArea > div > div > textarea[disabled] {
        color: #000000 !important;
        background-color: #f8f9fa !important;
    }
    
    .stRadio > div {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Tabs styling with better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f9fa;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #4A7637 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa !important;
        color: #4A7637 !important;
        border-color: #B9C930 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4A7637 !important;
        color: white !important;
        border-color: #4A7637 !important;
        font-weight: bold !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    
    /* Force tab text to be visible */
    .stTabs [data-baseweb="tab"] > div,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
    }
    
    /* Tab content styling */
    .stTabs > div[data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    /* Metrics styling */
    .stMetric > div {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric > div > div {
        color: #4A7637;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #2d4a1f;
    }
    
    /* Download button styling with maximum specificity */
    .stDownloadButton > button,
    .stDownloadButton button,
    button[data-testid="stDownloadButton"],
    div[data-testid="stDownloadButton"] button,
    [data-testid="stDownloadButton"] button {
        background-color: #B9C930 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }
    
    .stDownloadButton > button:hover,
    .stDownloadButton button:hover,
    button[data-testid="stDownloadButton"]:hover,
    div[data-testid="stDownloadButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        background-color: #a6b82a !important;
        color: white !important;
    }
    
    /* Download button text color override */
    .stDownloadButton button,
    .stDownloadButton button *,
    button[data-testid="stDownloadButton"],
    button[data-testid="stDownloadButton"] *,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button *,
    [data-testid="stDownloadButton"] button,
    [data-testid="stDownloadButton"] button * {
        color: white !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4A7637;
    }
    
    /* Alert styling */
    .stAlert > div {
        border-left: 4px solid #4A7637;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }
    
    .stSuccess {
        background-color: #ffffff;
        border-left: 4px solid #4A7637;
        border: 1px solid #d4edda;
        color: #2d4a1f;
    }
    
    .stInfo {
        background-color: #ffffff;
        border-left: 4px solid #B9C930;
        border: 1px solid #d1ecf1;
        color: #4A7637;
    }
    
    .stWarning {
        background-color: #ffffff;
        border-left: 4px solid #B9C930;
        border: 1px solid #ffeaa7;
        color: #6b7d28;
    }
    
    .stError {
        background-color: #ffffff;
        border-left: 4px solid #8b5a3c;
        border: 1px solid #f8d7da;
        color: #8b5a3c;
    }
    
    /* Expander styling */
    .stExpander > div > div > div {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    
    .stExpander [data-testid="stExpander"] > div:first-child {
        background-color: #f8f9fa;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Container styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="metric-container"] > div {
        color: #4A7637;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #4A7637;
    }
    
    /* Text colors */
    .stMarkdown {
        color: #2d4a1f;
    }
    
    /* Text area styling for summaries with maximum specificity */
    .stTextArea textarea,
    .stTextArea textarea[disabled],
    .stTextArea textarea[readonly],
    textarea,
    textarea[disabled],
    textarea[readonly],
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextArea"] textarea[disabled],
    div[data-testid="stTextArea"] textarea[readonly],
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextArea"] textarea[disabled],
    [data-testid="stTextArea"] textarea[readonly] {
        color: #000000 !important;
        background-color: #f8f9fa !important;
        border: 1px solid #d0d0d0 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Force black text in all text areas with maximum specificity */
    textarea,
    textarea::placeholder,
    textarea::-webkit-input-placeholder,
    textarea::-moz-placeholder,
    textarea:-ms-input-placeholder,
    .stTextArea textarea,
    .stTextArea textarea::placeholder,
    .stTextArea textarea::-webkit-input-placeholder,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Override any inherited text colors */
    .stTextArea,
    .stTextArea *,
    div[data-testid="stTextArea"],
    div[data-testid="stTextArea"] *,
    [data-testid="stTextArea"],
    [data-testid="stTextArea"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Ensure text areas have proper contrast on focus */
    .stTextArea textarea:focus,
    div[data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextArea"] textarea:focus {
        color: #000000 !important;
        background-color: #ffffff !important;
        border-color: #B9C930 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Force visible text in text areas */
    .stTextArea textarea::selection,
    div[data-testid="stTextArea"] textarea::selection,
    [data-testid="stTextArea"] textarea::selection {
        background-color: #B9C930 !important;
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Form styling */
    .stForm {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    
    /* Improve readability */
    h1, h2, h3, h4, h5, h6 {
        color: #2d4a1f;
    }
    
    p {
        color: #4A7637;
    }
    
    /* Card-like containers */
    .stContainer > div {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>RFP Analyzer & Processor</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Upload & Process", "RFP Database"])
    
    with tab1:
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
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
            os.makedirs(PDF_FOLDER, exist_ok=True)
            
            with st.form("batch_metadata_form"):
                st.markdown("### RFP Project Information")
                
                # Option to group files as one RFP or process individually
                col1, col2 = st.columns([2, 1])
                with col1:
                    group_mode = st.radio(
                        "How would you like to process these files?",
                        ["Process as separate RFPs", "Group as one RFP project"],
                        help="Choose 'Group as one RFP project' if all files belong to the same RFP from one client"
                    )
                
                if group_mode == "Group as one RFP project":
                    st.markdown("#### Project-Level Information")
                    st.markdown("*This information will apply to all files in the project*")
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        project_title = st.text_input(
                            "RFP Project Title",
                            placeholder="e.g., Hospital Lab Services RFP 2025",
                            help="Main title for this RFP project"
                        )
                    
                    with col2:
                        project_sender = st.text_input(
                            "Client/Organization",
                            placeholder="e.g., City Hospital System",
                            help="The organization that sent this RFP"
                        )
                    
                    with col3:
                        project_decision = st.selectbox(
                            "Project Status",
                            ["Pending", "Approved", "Denied"],
                            help="Overall status for this RFP project"
                        )
                    
                    # Only show file listing if project info is provided
                    if project_title and project_sender:
                        st.markdown("#### Files in this Project")
                        st.markdown("*All files will be analyzed together and receive the same overall score and status*")
                        
                        # Show file listing with file types
                        files_info = []
                        for i, uploaded_file in enumerate(uploaded_files):
                            file_type = get_file_type(uploaded_file.name)
                            file_type_display = {
                                'pdf': '📄 PDF',
                                'docx': '📝 Word Document', 
                                'excel': '📊 Excel Spreadsheet',
                                'unknown': '❓ Unknown'
                            }.get(file_type, '❓ Unknown')
                            
                            files_info.append(f"{file_type_display} **{uploaded_file.name}**")
                        
                        # Display files in columns for better layout
                        cols = st.columns(2)
                        for i, file_info in enumerate(files_info):
                            with cols[i % 2]:
                                st.markdown(f"{i+1}. {file_info}")
                        
                        # Prepare data for all files using project information
                        titles, senders, decisions = [], [], []
                        for uploaded_file in uploaded_files:
                            titles.append(f"{project_title} - {uploaded_file.name}")
                            senders.append(project_sender)
                            decisions.append(project_decision)
                    else:
                        # Show individual form when project info is not complete
                        st.markdown("### Individual Document Information")
                        st.info("Complete the project information above to group these files, or fill out individual information below:")
                        
                        titles, senders, decisions = [], [], []
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            file_type = get_file_type(uploaded_file.name)
                            file_type_display = {
                                'pdf': 'PDF',
                                'docx': 'Word Document',
                                'excel': 'Excel Spreadsheet',
                                'unknown': 'Unknown'
                            }.get(file_type, 'Unknown')
                            
                            with st.expander(f"{uploaded_file.name} ({file_type_display})", expanded=True):
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    title = st.text_input(
                                        "Document Title",
                                        value=os.path.splitext(uploaded_file.name)[0],
                                        key=f"title_{uploaded_file.name}",
                                        placeholder="Enter a descriptive title..."
                                    )
                                    titles.append(title)
                                
                                with col2:
                                    sender = st.text_input(
                                        "From (Sender/Company)",
                                        key=f"sender_{uploaded_file.name}",
                                        placeholder="Company or sender name..."
                                    )
                                    senders.append(sender)
                                
                                with col3:
                                    decision = st.selectbox(
                                        "Initial Status",
                                        ["Pending", "Approved", "Denied"],
                                        key=f"decision_{uploaded_file.name}",
                                        help="Set initial status for this RFP"
                                    )
                                    decisions.append(decision)
                
                else:
                    st.markdown("### Individual Document Information")
                    st.markdown("Please provide information for each document:")
                    
                    titles, senders, decisions = [], [], []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        file_type = get_file_type(uploaded_file.name)
                        file_type_display = {
                            'pdf': 'PDF',
                            'docx': 'Word Document',
                            'excel': 'Excel Spreadsheet',
                            'unknown': 'Unknown'
                        }.get(file_type, 'Unknown')
                        
                        with st.expander(f"{uploaded_file.name} ({file_type_display})", expanded=True):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                title = st.text_input(
                                    "Document Title",
                                    value=os.path.splitext(uploaded_file.name)[0],
                                    key=f"title_{uploaded_file.name}",
                                    placeholder="Enter a descriptive title..."
                                )
                                titles.append(title)
                            
                            with col2:
                                sender = st.text_input(
                                    "From (Sender/Company)",
                                    key=f"sender_{uploaded_file.name}",
                                    placeholder="Company or sender name..."
                                )
                                senders.append(sender)
                            
                            with col3:
                                decision = st.selectbox(
                                    "Initial Status",
                                    ["Pending", "Approved", "Denied"],
                                    key=f"decision_{uploaded_file.name}",
                                    help="Set initial status for this RFP"
                                )
                                decisions.append(decision)
                
                st.markdown("---")
                submitted = st.form_submit_button(
                    "Process All Documents", 
                    use_container_width=True,
                    type="primary"
                )
            
            if submitted:
                # Validation for grouped projects
                if group_mode == "Group as one RFP project":
                    # If project info is complete, validate it
                    if project_title and project_sender:
                        if not project_title.strip() or not project_sender.strip():
                            st.error("⚠️ Project Title and Client/Organization cannot be empty.")
                            st.stop()
                    else:
                        # If project info is incomplete, treat as individual processing
                        group_mode = "Process as separate RFPs"
                        st.warning("⚠️ Switching to individual processing since project information is incomplete.")
                
                # Validation for individual processing
                if group_mode == "Process as separate RFPs":
                    empty_titles = [i for i, title in enumerate(titles) if not title or not title.strip()]
                    empty_senders = [i for i, sender in enumerate(senders) if not sender or not sender.strip()]
                    
                    if empty_titles:
                        file_names = [uploaded_files[i].name for i in empty_titles]
                        st.error(f"⚠️ Please fill in titles for: {', '.join(file_names)}")
                        st.stop()
                    
                    if empty_senders:
                        file_names = [uploaded_files[i].name for i in empty_senders]
                        st.error(f"⚠️ Please fill in sender/company for: {', '.join(file_names)}")
                        st.stop()
                
                # Initialize progress tracking in session state to prevent tab switching issues
                if 'processing_in_progress' not in st.session_state:
                    st.session_state.processing_in_progress = False
                
                st.session_state.processing_in_progress = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Handle grouped vs individual processing
                if group_mode == "Group as one RFP project" and project_title and project_sender:
                    st.info(f"Processing {len(uploaded_files)} files as one RFP project: '{project_title}'")
                    
                    # Combine all text from files for analysis
                    combined_text = ""
                    file_summaries = []
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        progress = (idx + 0.5) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Extracting text from {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
                        
                        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        try:
                            file_type = get_file_type(uploaded_file.name)
                            
                            if file_type == 'pdf':
                                if is_scanned(file_path):
                                    st.info(f"Applying OCR to scanned PDF: {uploaded_file.name}")
                                    text = extract_text(file_path, use_easyocr=True)
                                else:
                                    text = extract_text(file_path)
                            elif file_type == 'docx':
                                text = extract_text_from_file(file_path)
                            elif file_type == 'excel':
                                text = extract_text_from_file(file_path)
                            else:
                                st.warning(f"Unsupported file type: {uploaded_file.name}")
                                continue
                            
                            if text:
                                combined_text += f"\n\n--- Content from {uploaded_file.name} ---\n" + text
                                file_summaries.append(f"📄 {uploaded_file.name}: {generate_document_summary(text, 1)}")
                                
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            continue
                    
                    # Run AI analysis on combined text
                    status_text.text(f"Running AI analysis on combined RFP project...")
                    progress_bar.progress(0.8)
                    
                    with st.spinner(f"Analyzing complete RFP project: '{project_title}'..."):
                        prob = predict_document_probability(combined_text) if combined_text else 0.5
                    
                    # Create combined summary
                    project_summary = f"RFP Project: {project_title}\n\nContains {len(uploaded_files)} files:\n" + "\n".join(file_summaries)
                    if combined_text:
                        project_summary += f"\n\nOverall Summary: {generate_document_summary(combined_text, 3)}"
                    
                    # Save each file with project information
                    for idx, uploaded_file in enumerate(uploaded_files):
                        entry = {
                            "filename": uploaded_file.name,
                            "title": titles[idx],
                            "sender": senders[idx], 
                            "decision": decisions[idx],
                            "probability": prob,  # Same probability for all files in project
                            "summary": project_summary  # Same project summary for all files
                        }
                        add_entry_to_db(entry)
                    
                    progress_bar.progress(1.0)
                    status_text.text("RFP project processed successfully!")
                    st.session_state.processing_in_progress = False  # Reset processing flag
                    
                    # Show detailed success message
                    st.success(f"✅ **RFP Project Completed!**")
                    st.info(f"""
                    **Project:** {project_title}  
                    **Client:** {project_sender}  
                    **Status:** {project_decision}  
                    **Files Processed:** {len(uploaded_files)}  
                    **Combined AI Score:** {prob:.1%}
                    """)
                    
                    # Show historical learning info
                    historical_df = load_db()
                    num_approved = len(historical_df[historical_df['decision'] == 'Approved'])
                    num_denied = len(historical_df[historical_df['decision'] == 'Denied'])
                    
                    if num_approved + num_denied > 0:
                        st.info(f"📚 AI learned from {num_approved} approved and {num_denied} denied RFPs in your history")
                    else:
                        st.info("📚 This is one of your first RFPs! The AI will get smarter as you approve/deny more RFPs.")
                    
                else:
                    # Individual file processing (original logic)
                    for idx, uploaded_file in enumerate(uploaded_files):
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
                        
                        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        try:
                            file_type = get_file_type(uploaded_file.name)
                            
                            if file_type == 'pdf':
                                if is_scanned(file_path):
                                    st.info(f"Applying OCR to scanned PDF: {uploaded_file.name}")
                                    text = extract_text(file_path, use_easyocr=True)
                                else:
                                    text = extract_text(file_path)
                            elif file_type == 'docx':
                                st.info(f"Extracting text from Word document: {uploaded_file.name}")
                                text = extract_text_from_file(file_path)
                            elif file_type == 'excel':
                                st.info(f"Extracting text from Excel spreadsheet: {uploaded_file.name}")
                                text = extract_text_from_file(file_path)
                            else:
                                st.warning(f"Unsupported file type: {uploaded_file.name}")
                                continue
                                
                            summary = generate_document_summary(text) if text else "No text content available"
                            
                            # Show progress
                            with st.spinner(f"Running AI analysis on {uploaded_file.name}..."):
                                prob = predict_document_probability(text)
                            
                            # Show brief analysis info
                            historical_df = load_db()
                            num_approved = len(historical_df[historical_df['decision'] == 'Approved'])
                            num_denied = len(historical_df[historical_df['decision'] == 'Denied'])
                            
                            if num_approved + num_denied > 0:
                                st.info(f"Learned from {num_approved} approved and {num_denied} denied RFPs in your history")
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            continue
                        
                        entry = {
                            "filename": uploaded_file.name,
                            "title": titles[idx],
                            "sender": senders[idx],
                            "decision": decisions[idx],
                            "probability": prob,
                            "summary": summary
                        }
                        
                        add_entry_to_db(entry)
                        st.success(f"{uploaded_file.name} processed successfully! Score: {prob:.1%}")
                
                progress_bar.progress(1.0)
                status_text.text("All documents processed successfully!")
                st.session_state.processing_in_progress = False  # Reset processing flag
                st.balloons()
    
    with tab2:
        st.markdown("### RFP Database Overview")
        df = load_db()
        
        if not df.empty:
            # Add export button and summary stats at the top
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                unique_rfp_count = _count_unique_rfps(df)
                st.metric("Total RFPs", unique_rfp_count)
            with col2:
                st.metric("Avg Score", f"{df['probability'].mean():.1%}")
            with col3:
                approved_count = _count_unique_rfps_by_decision(df, 'Approved')
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
            
            approved_count = _count_unique_rfps_by_decision(df, 'Approved')
            denied_count = _count_unique_rfps_by_decision(df, 'Denied')
            pending_count = _count_unique_rfps_by_decision(df, 'Pending')
            
            if approved_count + denied_count > 0:
                st.info(f"Model Status: Learning from {approved_count} approved and {denied_count} denied RFPs. {pending_count} pending RFPs will get more accurate scores as you make decisions.")
            else:
                st.warning("Model Status: No historical decisions yet. Start approving/denying RFPs to improve prediction accuracy!")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                if st.button("Rerun Model for All Entries", use_container_width=True):
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
            
            with col2:
                filter_option = st.selectbox(
                    "Filter by Status",
                    ["All", "Pending", "Approved", "Denied"],
                    help="Filter RFPs by their current status"
                )
            
            with col3:
                sort_option = st.selectbox(
                    "Sort by",
                    ["Score (High to Low)", "Score (Low to High)", "Title", "Date Added"]
                )
            
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
                        project_key = f"{row['title'].split(' - ')[0]}_{row['sender']}_{row['probability']:.3f}"
                    else:
                        project_key = f"{row['title']}_{row['sender']}_{row['probability']:.3f}"
                    
                    # Skip if we already displayed this project
                    if project_key in displayed_projects:
                        continue
                    
                    displayed_projects.add(project_key)
                    
                    # Find all files in this project
                    if ' - ' in row['title']:
                        project_title_base = row['title'].split(' - ')[0]
                        project_files = filtered_df[
                            (filtered_df['title'].str.contains(f"^{re.escape(project_title_base)} - ", na=False, regex=True)) &
                            (filtered_df['sender'] == row['sender']) &
                            (abs(filtered_df['probability'] - row['probability']) < 0.01)
                        ]
                    else:
                        project_files = filtered_df[
                            (filtered_df['title'] == row['title']) &
                            (filtered_df['sender'] == row['sender']) &
                            (abs(filtered_df['probability'] - row['probability']) < 0.01)
                        ]
                    
                    # Display project as a single entry
                    _display_project_entry(row, project_files, project_title_base if ' - ' in row['title'] else row['title'])
                
                else:
                    # Display individual file entry
                    _display_individual_entry(row)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #ffffff; border-radius: 10px; border: 2px dashed #B9C930; box-shadow: 0 2px 4px rgba(185, 201, 48, 0.1);">
                <h3 style="color: #4A7637;">No documents uploaded yet</h3>
                <p style="color: #4A7637;">Upload your first RFP document (PDF, Word, or Excel) in the "Upload & Process" tab to get started!</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
