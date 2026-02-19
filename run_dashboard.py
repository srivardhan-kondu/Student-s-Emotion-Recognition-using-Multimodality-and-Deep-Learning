#!/usr/bin/env python3
"""
Main entry point to run the Multimodal Emotion Recognition Dashboard.
Usage: python run_dashboard.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    app_path = os.path.join(os.path.dirname(__file__), "src", "dashboard", "app.py")
    
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=localhost",
        "--theme.base=light"
    ]
    
    print("🚀 Starting Multimodal Emotion Recognition Dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    
    sys.exit(stcli.main())
