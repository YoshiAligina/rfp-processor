# Upload Handler - Handles file upload and processing logic

"""
File Upload and Processing Handler for RFP Analysis System

This module manages the complete file upload workflow including metadata collection,
document processing, ML prediction generation, and database storage. It supports
both individual document processing and project-grouped submissions where multiple
files belong to a single RFP. The module integrates text extraction, OCR processing,
and intelligent summarization to create comprehensive RFP entries.

Key features:
- Flexible upload modes (individual documents vs. grouped projects)
- Comprehensive metadata collection with intelligent defaults
- Multi-format document processing (PDF, DOCX, Excel)
- Real-time ML probability prediction
- Intelligent document summarization
- Progress tracking and error handling
- Database integration with duplicate detection
"""

import streamlit as st
import os
from ocr_utils import is_scanned, extract_text
from model_utils import predict_document_probability
from data_utils import add_entry_to_db, load_db
from document_utils import extract_text_from_file, get_file_type
from ui_components import generate_document_summary

PDF_FOLDER = "documents"

def handle_file_upload_form(uploaded_files):
    """
    Creates and manages the comprehensive file upload form interface with metadata collection.
    This function orchestrates the complete upload workflow, providing users with options
    for individual or grouped processing, collecting necessary metadata, and preparing
    files for processing. It creates an intuitive form interface that adapts based on
    user selections and provides clear guidance for both single document and multi-file
    project submissions. Returns all collected metadata for downstream processing.
    
    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects
        
    Returns:
        tuple: Contains submission status, processing mode, and all collected metadata
    """
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
        
        titles, senders, decisions = [], [], []
        project_title, project_sender, project_decision = None, None, None
        
        if group_mode == "Group as one RFP project":
            project_title, project_sender, project_decision, titles, senders, decisions = _handle_project_mode(uploaded_files)
        else:
            titles, senders, decisions = _handle_individual_mode(uploaded_files)
        
        st.markdown("---")
        submitted = st.form_submit_button(
            "Process All Documents", 
            use_container_width=True,
            type="primary"
        )
        
        return submitted, group_mode, titles, senders, decisions, project_title, project_sender, project_decision

def _handle_project_mode(uploaded_files):
    """
    Manages metadata collection for grouped project submissions with shared information.
    This function creates an interface for collecting project-level metadata that
    applies to all files in a multi-document RFP submission. It simplifies the
    user experience by allowing shared title, sender, and decision information
    while still providing individual file identification. Essential for processing
    complex RFPs that consist of multiple related documents from a single client
    or project, maintaining logical grouping while ensuring proper file tracking.
    
    Args:
        uploaded_files (list): List of files to be processed as a single project
        
    Returns:
        tuple: Project metadata including title, sender, decision, and file lists
    """
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
        titles, senders, decisions = _collect_individual_metadata(uploaded_files)
    
    return project_title, project_sender, project_decision, titles, senders, decisions

def _handle_individual_mode(uploaded_files):
    """Handle individual file processing mode"""
    st.markdown("### Individual Document Information")
    st.markdown("Please provide information for each document:")
    return _collect_individual_metadata(uploaded_files)

def _collect_individual_metadata(uploaded_files):
    """Collect metadata for individual files"""
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
    
    return titles, senders, decisions

def process_uploaded_files(uploaded_files, group_mode, titles, senders, decisions, 
                          project_title=None, project_sender=None, project_decision=None):
    """Process the uploaded files based on the selected mode"""
    
    # Validation
    if group_mode == "Group as one RFP project":
        if project_title and project_sender:
            if not project_title.strip() or not project_sender.strip():
                st.error("⚠️ Project Title and Client/Organization cannot be empty.")
                return False
        else:
            group_mode = "Process as separate RFPs"
            st.warning("⚠️ Switching to individual processing since project information is incomplete.")
    
    if group_mode == "Process as separate RFPs":
        empty_titles = [i for i, title in enumerate(titles) if not title or not title.strip()]
        empty_senders = [i for i, sender in enumerate(senders) if not sender or not sender.strip()]
        
        if empty_titles:
            file_names = [uploaded_files[i].name for i in empty_titles]
            st.error(f"⚠️ Please fill in titles for: {', '.join(file_names)}")
            return False
        
        if empty_senders:
            file_names = [uploaded_files[i].name for i in empty_senders]
            st.error(f"⚠️ Please fill in sender/company for: {', '.join(file_names)}")
            return False
    
    # Initialize progress tracking
    if 'processing_in_progress' not in st.session_state:
        st.session_state.processing_in_progress = False
    
    st.session_state.processing_in_progress = True
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if group_mode == "Group as one RFP project" and project_title and project_sender:
            _process_as_project(uploaded_files, titles, senders, decisions, 
                              project_title, project_sender, project_decision, 
                              progress_bar, status_text)
        else:
            _process_individually(uploaded_files, titles, senders, decisions, 
                                progress_bar, status_text)
        
        progress_bar.progress(1.0)
        status_text.text("All documents processed successfully!")
        st.session_state.processing_in_progress = False
        st.balloons()
        return True
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.session_state.processing_in_progress = False
        return False

def _process_as_project(uploaded_files, titles, senders, decisions, 
                       project_title, project_sender, project_decision, 
                       progress_bar, status_text):
    """
    Process files as a single RFP project with intelligent document combination.
    This function treats multiple files as components of one cohesive RFP,
    combining their content strategically for unified analysis while maintaining
    individual file tracking for reference purposes.
    """
    st.info(f"🔗 Processing {len(uploaded_files)} files as unified RFP project: '{project_title}'")
    
    # Enhanced document combination with structured approach
    combined_sections = {
        'executive_summary': [],
        'technical_specs': [],
        'financial_info': [],
        'general_content': []
    }
    
    file_summaries = []
    total_word_count = 0
    
    for idx, uploaded_file in enumerate(uploaded_files):
        progress = (idx + 0.5) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"📄 Processing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
        
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Extract text based on file type
            file_type = get_file_type(uploaded_file.name)
            
            if file_type == 'pdf':
                if is_scanned(file_path):
                    st.info(f"🔍 Applying OCR to scanned PDF: {uploaded_file.name}")
                    text = extract_text(file_path, use_easyocr=True)
                else:
                    text = extract_text(file_path)
            elif file_type == 'docx':
                text = extract_text_from_file(file_path)
            elif file_type == 'excel':
                text = extract_text_from_file(file_path)
            else:
                st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
                continue
            
            if text and text.strip():
                # Categorize content based on filename and content patterns
                file_category = _categorize_document_content(uploaded_file.name, text)
                combined_sections[file_category].append({
                    'filename': uploaded_file.name,
                    'content': text,
                    'word_count': len(text.split())
                })
                
                total_word_count += len(text.split())
                file_summaries.append(f"📄 {uploaded_file.name} ({file_category}): {generate_document_summary(text, 1)}")
                st.success(f"✅ Processed {uploaded_file.name} - {len(text.split())} words")
            else:
                st.warning(f"⚠️ No readable text found in {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    # Create intelligently structured combined text
    combined_text = _create_structured_combination(combined_sections, project_title)
    
    # Run AI analysis on intelligently combined text
    status_text.text(f"🤖 Running AI analysis on unified project ({total_word_count:,} words)...")
    progress_bar.progress(0.8)
    
    with st.spinner(f"🔍 Analyzing complete RFP project: '{project_title}'..."):
        prob = predict_document_probability(combined_text) if combined_text else 0.5
    
    # Create comprehensive project summary
    project_summary = _create_project_summary(project_title, uploaded_files, file_summaries, combined_text, total_word_count)
    
    # Save each file with unified project information
    for idx, uploaded_file in enumerate(uploaded_files):
        entry = {
            "filename": uploaded_file.name,
            "title": titles[idx],
            "sender": senders[idx], 
            "decision": decisions[idx],
            "probability": prob,  # Unified probability for all files in project
            "summary": project_summary  # Comprehensive project summary for all files
        }
        add_entry_to_db(entry)
    
    progress_bar.progress(1.0)
    status_text.text("RFP project processed successfully!")
    
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

def _process_individually(uploaded_files, titles, senders, decisions, progress_bar, status_text):
    """Process files individually"""
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

def _categorize_document_content(filename, text):
    """
    Categorizes document content based on filename patterns and content analysis.
    Helps organize multi-document RFPs into logical sections for better analysis.
    """
    filename_lower = filename.lower()
    text_lower = text.lower()
    
    # Check filename patterns first
    if any(term in filename_lower for term in ['summary', 'executive', 'overview', 'intro']):
        return 'executive_summary'
    elif any(term in filename_lower for term in ['spec', 'technical', 'requirement', 'scope']):
        return 'technical_specs'
    elif any(term in filename_lower for term in ['budget', 'cost', 'financial', 'price', 'billing']):
        return 'financial_info'
    
    # Check content patterns if filename doesn't match
    executive_terms = ['executive summary', 'overview', 'project description', 'background']
    technical_terms = ['specifications', 'requirements', 'technical', 'scope of work', 'deliverables']
    financial_terms = ['budget', 'cost', 'price', 'financial', 'payment terms', 'billing']
    
    if any(term in text_lower for term in executive_terms):
        return 'executive_summary'
    elif any(term in text_lower for term in technical_terms):
        return 'technical_specs'
    elif any(term in text_lower for term in financial_terms):
        return 'financial_info'
    
    return 'general_content'

def _create_structured_combination(combined_sections, project_title):
    """
    Creates an intelligently structured combination of all project documents.
    Organizes content in a logical order for optimal AI analysis.
    """
    structured_text = f"RFP PROJECT: {project_title}\n\n"
    
    # Combine in logical order: Executive Summary → Technical → Financial → General
    section_order = ['executive_summary', 'technical_specs', 'financial_info', 'general_content']
    section_titles = {
        'executive_summary': 'EXECUTIVE SUMMARY & OVERVIEW',
        'technical_specs': 'TECHNICAL SPECIFICATIONS & REQUIREMENTS', 
        'financial_info': 'FINANCIAL & BUDGET INFORMATION',
        'general_content': 'ADDITIONAL PROJECT INFORMATION'
    }
    
    for section_key in section_order:
        if combined_sections[section_key]:
            structured_text += f"\n\n=== {section_titles[section_key]} ===\n\n"
            
            for doc in combined_sections[section_key]:
                structured_text += f"--- From {doc['filename']} ---\n"
                structured_text += doc['content']
                structured_text += "\n\n"
    
    return structured_text

def _create_project_summary(project_title, uploaded_files, file_summaries, combined_text, total_word_count):
    """
    Creates a comprehensive summary for the entire RFP project.
    """
    summary_parts = [
        f"🎯 RFP PROJECT: {project_title}",
        f"📁 Multi-Document Project ({len(uploaded_files)} files, {total_word_count:,} words)",
        "",
        "📄 PROJECT COMPONENTS:",
    ]
    
    summary_parts.extend(file_summaries)
    
    if combined_text:
        summary_parts.extend([
            "",
            "📋 OVERALL PROJECT SUMMARY:",
            generate_document_summary(combined_text, 4)  # Longer summary for projects
        ])
    
    return "\n".join(summary_parts)
