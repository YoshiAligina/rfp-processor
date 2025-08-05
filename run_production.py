#!/usr/bin/env python3
"""
Production Flask Runner - No Auto-Reload
This version disables file watching to prevent interruptions during model processing
"""

from web_app import app
import os

if __name__ == '__main__':
    print("="*60)
    print("   RFP ANALYZER - PRODUCTION MODE")
    print("="*60)
    print("🚀 Starting Flask server without file watching...")
    print("💡 This prevents interruptions during model processing")
    print("🌐 Server will be available at:")
    print("   - Local:   http://127.0.0.1:5000")
    print("   - Network: http://0.0.0.0:5000")
    print("📝 Note: Server won't auto-reload on file changes")
    print("="*60)
    
    # Run without debug mode and auto-reloader
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
