"""
Script to run the Streamlit dashboard.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    sys.exit(stcli.main())
