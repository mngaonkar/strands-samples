#!/usr/bin/env python3
"""
Launch script for the Strands Agent Web UI
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit web UI"""
    
    # Check if we're in the right directory
    if not os.path.exists("web_ui.py"):
        print("❌ Error: web_ui.py not found in current directory")
        print("Please run this script from the strands-agent-samples directory")
        sys.exit(1)
    
    print("🚀 Starting Strands Agent Web UI...")
    print("📝 The web interface will open in your browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "web_ui.py", 
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web UI...")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()