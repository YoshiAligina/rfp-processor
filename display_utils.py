# Display Utils - Functions for displaying RFP entries and project details

"""
Display Utilities for RFP Database Management Interface

This module provides comprehensive display functionality for the RFP database interface,
handling both individual document entries and grouped project displays. It manages
complex UI interactions including decision updates, file downloads, probability
recalculations, and detailed document views. The module integrates with data
management, document processing, and ML prediction systems to provide a complete
database management experience.

Key features:
- Project-based grouping for multi-file RFP submissions
- Individual document entry displays with full interactivity
- Real-time decision updates with ML model integration
- File download capabilities with proper MIME type handling
- Expandable detail views with document summaries
- Integrated prediction recalculation and model feedback
"""

import streamlit as st
import os
from data_utils import update_decision, delete_entry, update_probability
from document_utils import extract_text_from_file, get_appropriate_mime_type
from model_utils import predict_document_probability

PDF_FOLDER = "documents"

def display_project_entry(representative_row, project_files, project_title):
    """
    Renders a comprehensive project entry interface for multi-file RFP submissions.
    This function creates a grouped display for RFP projects consisting of multiple
    related documents, showing combined project information, individual file details,
    and interactive controls for project management. It provides expandable details,
    batch download capabilities, and decision management for the entire project
    while maintaining access to individual file operations and metadata.
    """
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

def display_individual_entry(row):
    """
    Renders a complete individual document entry interface with full interactivity.
    This function creates a comprehensive display for standalone RFP documents,
    providing document information, approval probability, decision controls,
    file download capabilities, and expandable detail views. It integrates
    real-time ML prediction updates, decision tracking, and document management
    features into a cohesive interface that allows users to efficiently review
    and manage individual RFP submissions with immediate feedback and actions.
    """
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
