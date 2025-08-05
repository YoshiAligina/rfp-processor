#!/usr/bin/env python3
"""
GitHub Pages HTML Generator for RFP Analyzer
Creates a professional static HTML report perfect for GitHub Pages deployment
"""

import pandas as pd
import os
import json
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_utils import load_db
    from model_utils import get_model_info
    from rfp_counter import count_unique_rfps, count_unique_rfps_by_decision
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Creating demo data for GitHub Pages...")

def create_demo_data():
    """Create demo data for GitHub Pages when actual data isn't available"""
    demo_data = {
        'total_rfps': 15,
        'avg_score': 0.72,
        'approved_count': 8,
        'denied_count': 4,
        'pending_count': 3,
        'model_info': {
            'has_fine_tuned_model': True,
            'historical_decisions': 12,
            'approved_count': 8,
            'denied_count': 4
        },
        'entries': [
            {
                'title': 'Healthcare IT Infrastructure Upgrade',
                'filename': 'healthcare_it_rfp.pdf',
                'sender': 'Metro Health System',
                'probability': 0.85,
                'decision': 'Approved',
                'summary': 'Comprehensive IT infrastructure modernization for healthcare network including cloud migration, security upgrades, and system integration.'
            },
            {
                'title': 'Municipal Water Treatment Facility',
                'filename': 'water_treatment_rfp.pdf', 
                'sender': 'City of Springfield',
                'probability': 0.42,
                'decision': 'Denied',
                'summary': 'Design and construction of new water treatment facility with advanced filtration systems and environmental monitoring capabilities.'
            },
            {
                'title': 'Educational Technology Platform',
                'filename': 'edu_tech_rfp.pdf',
                'sender': 'Regional School District',
                'probability': 0.78,
                'decision': 'Approved',
                'summary': 'Development of integrated learning management system with student portal, teacher tools, and parent communication features.'
            },
            {
                'title': 'Transportation Management System',
                'filename': 'transport_mgmt_rfp.pdf',
                'sender': 'Transit Authority',
                'probability': 0.63,
                'decision': 'Pending',
                'summary': 'Implementation of real-time transit tracking, route optimization, and passenger information system across metro area.'
            },
            {
                'title': 'Cybersecurity Assessment Services',
                'filename': 'cybersecurity_rfp.pdf',
                'sender': 'State Government',
                'probability': 0.91,
                'decision': 'Approved',
                'summary': 'Comprehensive cybersecurity audit, penetration testing, and risk assessment for critical government infrastructure.'
            }
        ]
    }
    return demo_data

def generate_github_pages_html():
    """Generate a professional HTML report optimized for GitHub Pages"""
    
    try:
        # Try to load real data
        df = load_db()
        model_info = get_model_info()
        
        # Calculate metrics
        total_rfps = count_unique_rfps(df) if not df.empty else 0
        avg_score = df['probability'].mean() if not df.empty else 0
        approved_count = count_unique_rfps_by_decision(df, 'Approved') if not df.empty else 0
        denied_count = count_unique_rfps_by_decision(df, 'Denied') if not df.empty else 0
        pending_count = count_unique_rfps_by_decision(df, 'Pending') if not df.empty else 0
        
        entries = df.to_dict('records') if not df.empty else []
        is_demo = False
        
    except Exception as e:
        print(f"Using demo data for GitHub Pages: {e}")
        demo_data = create_demo_data()
        total_rfps = demo_data['total_rfps']
        avg_score = demo_data['avg_score']
        approved_count = demo_data['approved_count']
        denied_count = demo_data['denied_count']
        pending_count = demo_data['pending_count']
        model_info = demo_data['model_info']
        entries = demo_data['entries']
        is_demo = True
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RFP Analyzer Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f8f9fa; }}
        .metric-card {{ text-align: center; padding: 2rem; }}
        .metric-value {{ font-size: 2.5rem; font-weight: bold; color: #2E86C1; }}
        .metric-label {{ color: #6C757D; font-size: 0.9rem; text-transform: uppercase; }}
        .status-approved {{ background-color: #D5F4E6; color: #27AE60; padding: 0.4rem 0.8rem; border-radius: 15px; }}
        .status-denied {{ background-color: #FADBD8; color: #E74C3C; padding: 0.4rem 0.8rem; border-radius: 15px; }}
        .status-pending {{ background-color: #FCF3CF; color: #F39C12; padding: 0.4rem 0.8rem; border-radius: 15px; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-primary">
        <div class="container">
            <span class="navbar-brand">
                <i class="fas fa-file-contract me-2"></i>RFP Analyzer Report
            </span>
            <span class="navbar-text">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
    </nav>

    <div class="container my-4">
        <!-- Metrics -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="metric-value">{total_rfps}</div>
                    <div class="metric-label">Total RFPs</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="metric-value">{avg_score:.1%}</div>
                    <div class="metric-label">Avg Score</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="metric-value text-success">{approved_count}</div>
                    <div class="metric-label">Approved</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="metric-value text-warning">{pending_count}</div>
                    <div class="metric-label">Pending</div>
                </div>
            </div>
        </div>

        <!-- Model Status -->
        <div class="card mb-4">
            <div class="card-header">
                <h5><i class="fas fa-brain me-2"></i>Model Status</h5>
            </div>
            <div class="card-body">
                <p><strong>Model Type:</strong> {"Fine-tuned" if model_info['has_fine_tuned_model'] else "Base"}</p>
                <p><strong>Historical Decisions:</strong> {model_info['historical_decisions']}</p>
                <p><strong>Training Data:</strong> {model_info['approved_count']} approved, {model_info['denied_count']} denied</p>
            </div>
        </div>

        <!-- RFP Entries -->
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-database me-2"></i>RFP Entries</h5>
            </div>
            <div class="card-body">
    """
    
    if df.empty:
        html_content += """
                <div class="text-center py-4">
                    <i class="fas fa-folder-open fa-3x text-muted mb-3"></i>
                    <h4>No RFPs Found</h4>
                </div>
        """
    else:
        html_content += """
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Title</th>
                                <th>Sender</th>
                                <th>Score</th>
                                <th>Status</th>
                                <th>Summary</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for _, entry in df.iterrows():
            score_color = "success" if entry['probability'] >= 0.7 else "warning" if entry['probability'] >= 0.4 else "danger"
            summary_preview = entry['summary'][:200] + "..." if len(entry['summary']) > 200 else entry['summary']
            
            html_content += f"""
                            <tr>
                                <td>
                                    <strong>{entry['title']}</strong><br>
                                    <small class="text-muted">{entry['filename']}</small>
                                </td>
                                <td>{entry['sender']}</td>
                                <td>
                                    <div class="progress" style="height: 20px;">
                                        <div class="progress-bar bg-{score_color}" 
                                             style="width: {entry['probability']*100:.0f}%">
                                            {entry['probability']:.1%}
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <span class="status-{entry['decision'].lower()}">{entry['decision']}</span>
                                </td>
                                <td>{summary_preview}</td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    
    # Save HTML file
    output_file = "rfp_analyzer_report.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Static HTML report generated: {output_file}")
    print(f"📁 Open {os.path.abspath(output_file)} in your browser")
    
    return output_file

if __name__ == "__main__":
    generate_static_html()
