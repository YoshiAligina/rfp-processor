# UI Styles for RFP Analyzer
# Contains all CSS styling for the Streamlit application

UI_STYLES = """
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
"""
