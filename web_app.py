#!/usr/bin/env python3
"""
RFP Analyzer Web Application - Flask Version
Converts the Streamlit RFP processor into a professional HTML web application
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
import json

# Import your existing modules
from data_utils import load_db, update_probability, add_entry_to_db, update_decision, delete_entry
from document_utils import extract_text_from_file, get_file_type
from model_utils import predict_document_probability, manual_fine_tune, get_model_info
from rfp_counter import count_unique_rfps, count_unique_rfps_by_decision
from upload_handler import generate_document_summary

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'documents'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main dashboard page"""
    df = load_db()
    
    # Calculate metrics
    total_rfps = count_unique_rfps(df) if not df.empty else 0
    avg_score = df['probability'].mean() if not df.empty else 0
    approved_count = count_unique_rfps_by_decision(df, 'Approved') if not df.empty else 0
    pending_count = count_unique_rfps_by_decision(df, 'Pending') if not df.empty else 0
    
    # Get model info
    model_info = get_model_info()
    
    # Recent entries (last 10)
    recent_entries = df.tail(10).to_dict('records') if not df.empty else []
    
    return render_template('index.html', 
                         total_rfps=total_rfps,
                         avg_score=avg_score,
                         approved_count=approved_count,
                         pending_count=pending_count,
                         model_info=model_info,
                         recent_entries=recent_entries)

@app.route('/upload')
def upload_page():
    """File upload page"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    if 'files' not in request.files:
        flash('No files selected')
        return redirect(request.url)
    
    files = request.files.getlist('files')
    processing_mode = request.form.get('processing_mode', 'individual')
    
    if processing_mode == 'project':
        return process_project_files(files)
    else:
        return process_individual_files(files)

def process_individual_files(files):
    """Process files individually"""
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Extract text and analyze
                text = extract_text_from_file(filepath)
                probability = predict_document_probability(text)
                summary = generate_document_summary(text)
                
                # Get metadata from form
                title = request.form.get(f'title_{file.filename}', filename)
                sender = request.form.get(f'sender_{file.filename}', '')
                decision = request.form.get(f'decision_{file.filename}', 'Pending')
                
                # Save to database
                entry = {
                    "filename": filename,
                    "title": title,
                    "sender": sender,
                    "decision": decision,
                    "probability": probability,
                    "summary": summary
                }
                add_entry_to_db(entry)
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'probability': probability,
                    'message': f'Processed successfully! Score: {probability:.1%}'
                })
                
            except Exception as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'message': f'Error: {str(e)}'
                })
    
    return render_template('upload_results.html', results=results)

def process_project_files(files):
    """Process files as a single project"""
    project_title = request.form.get('project_title', '')
    project_sender = request.form.get('project_sender', '')
    project_decision = request.form.get('project_decision', 'Pending')
    
    if not project_title or not project_sender:
        flash('Project title and sender are required for project mode')
        return redirect(url_for('upload_page'))
    
    # Combine all file content
    combined_text = ""
    file_summaries = []
    processed_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                text = extract_text_from_file(filepath)
                if text:
                    combined_text += f"\n\n--- Content from {filename} ---\n" + text
                    file_summaries.append(f"📄 {filename}: {generate_document_summary(text, 1)}")
                    processed_files.append(filename)
            except Exception as e:
                flash(f'Error processing {filename}: {str(e)}')
    
    if combined_text:
        # Analyze combined content
        probability = predict_document_probability(combined_text)
        
        # Create project summary
        project_summary = f"🎯 RFP PROJECT: {project_title}\n\nContains {len(processed_files)} files:\n" + "\n".join(file_summaries)
        project_summary += f"\n\nOverall Summary: {generate_document_summary(combined_text, 3)}"
        
        # Save each file with project information
        for filename in processed_files:
            entry = {
                "filename": filename,
                "title": f"{project_title} - {filename}",
                "sender": project_sender,
                "decision": project_decision,
                "probability": probability,
                "summary": project_summary
            }
            add_entry_to_db(entry)
        
        flash(f'Project "{project_title}" processed successfully! Score: {probability:.1%}')
        return redirect(url_for('database'))
    else:
        flash('No valid content found in uploaded files')
        return redirect(url_for('upload_page'))

@app.route('/database')
def database():
    """Database view page"""
    df = load_db()
    
    # Get filter and sort parameters
    filter_status = request.args.get('filter', 'All')
    sort_by = request.args.get('sort', 'Score (High to Low)')
    
    # Apply filters
    if filter_status != 'All' and not df.empty:
        df = df[df['decision'] == filter_status]
    
    # Apply sorting
    if not df.empty:
        if sort_by == 'Score (High to Low)':
            df = df.sort_values('probability', ascending=False)
        elif sort_by == 'Score (Low to High)':
            df = df.sort_values('probability', ascending=True)
        elif sort_by == 'Title':
            df = df.sort_values('title')
        elif sort_by == 'Date Added':
            df = df.sort_values('filename')  # Approximate by filename
    
    entries = df.to_dict('records') if not df.empty else []
    
    return render_template('database.html', 
                         entries=entries,
                         filter_status=filter_status,
                         sort_by=sort_by)

@app.route('/update_decision', methods=['POST'])
def update_decision_route():
    """Update RFP decision"""
    filename = request.form.get('filename')
    new_decision = request.form.get('decision')
    
    if filename and new_decision:
        update_decision(filename, new_decision)
        flash(f'Updated {filename} to {new_decision}')
    
    return redirect(url_for('database'))

@app.route('/delete_entry/<filename>', methods=['POST'])
def delete_entry_route(filename):
    """Delete RFP entry"""
    try:
        # Delete the file from filesystem if it exists
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete from database
        delete_entry(filename)
        flash(f'✅ Successfully deleted {filename}')
    except Exception as e:
        flash(f'❌ Error deleting {filename}: {str(e)}')
    
    return redirect(url_for('database'))

@app.route('/edit_entry/<filename>')
def edit_entry_page(filename):
    """Show edit form for RFP entry"""
    df = load_db()
    entry = df[df['filename'] == filename]
    
    if entry.empty:
        flash(f'Entry {filename} not found')
        return redirect(url_for('database'))
    
    entry_data = entry.iloc[0].to_dict()
    return render_template('edit_entry.html', entry=entry_data)

@app.route('/edit_entry/<filename>', methods=['POST'])
def update_entry_route(filename):
    """Update RFP entry details"""
    try:
        df = load_db()
        entry_index = df[df['filename'] == filename].index
        
        if len(entry_index) == 0:
            flash(f'Entry {filename} not found')
            return redirect(url_for('database'))
        
        # Update entry fields
        df.loc[entry_index[0], 'title'] = request.form.get('title', '')
        df.loc[entry_index[0], 'sender'] = request.form.get('sender', '')
        df.loc[entry_index[0], 'decision'] = request.form.get('decision', 'Pending')
        df.loc[entry_index[0], 'summary'] = request.form.get('summary', '')
        
        # Save updated data
        df.to_csv('rfp_db.csv', index=False)
        flash(f'✅ Successfully updated {filename}')
        
    except Exception as e:
        flash(f'❌ Error updating {filename}: {str(e)}')
    
    return redirect(url_for('database'))

@app.route('/train_model', methods=['POST'])
def train_model():
    """Trigger model training"""
    try:
        success = manual_fine_tune()
        if success:
            flash('🎉 Model training completed successfully!')
        else:
            flash('❌ Model training failed. Check logs for details.')
    except Exception as e:
        flash(f'Error during training: {str(e)}')
    
    return redirect(url_for('index'))

@app.route('/rerun_model', methods=['POST'])
def rerun_model():
    """Rerun model predictions for all entries"""
    try:
        df = load_db()
        updated_count = 0
        
        for _, row in df.iterrows():
            filepath = os.path.join(UPLOAD_FOLDER, row['filename'])
            if os.path.exists(filepath):
                try:
                    text = extract_text_from_file(filepath)
                    new_prob = predict_document_probability(text)
                    update_probability(row['filename'], new_prob)
                    updated_count += 1
                except Exception as e:
                    print(f"Error updating {row['filename']}: {e}")
        
        flash(f'✅ Updated {updated_count} entries with improved predictions!')
    except Exception as e:
        flash(f'Error during model rerun: {str(e)}')
    
    return redirect(url_for('database'))

@app.route('/api/model_info')
def api_model_info():
    """API endpoint for model information"""
    return jsonify(get_model_info())

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    df = load_db()
    
    stats = {
        'total_rfps': count_unique_rfps(df) if not df.empty else 0,
        'avg_score': float(df['probability'].mean()) if not df.empty else 0,
        'approved_count': count_unique_rfps_by_decision(df, 'Approved') if not df.empty else 0,
        'denied_count': count_unique_rfps_by_decision(df, 'Denied') if not df.empty else 0,
        'pending_count': count_unique_rfps_by_decision(df, 'Pending') if not df.empty else 0
    }
    
    return jsonify(stats)

@app.route('/api/regenerate_score', methods=['POST'])
def api_regenerate_score():
    """API endpoint to regenerate AI score for a specific entry"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'})
        
        # Check if file exists
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        # Extract text and get new prediction
        text = extract_text_from_file(filepath)
        new_probability = predict_document_probability(text)
        
        # Update in database
        update_probability(filename, new_probability)
        
        return jsonify({
            'success': True, 
            'score': f"{new_probability * 100:.1f}",
            'probability': new_probability
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    """Download an uploaded file"""
    try:
        # Verify file exists in database
        df = load_db()
        if df.empty or filename not in df['filename'].values:
            flash('File not found in database')
            return redirect(url_for('database'))
        
        # Secure the filename and check if file exists
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        if not os.path.exists(file_path):
            flash('Physical file not found on server')
            return redirect(url_for('database'))
        
        # Send the file for download
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('database'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
