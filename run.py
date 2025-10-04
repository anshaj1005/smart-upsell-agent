#!/usr/bin/env python3
"""
🚀 Smart Upsell Agent - Enhanced Startup Script
AI-Powered Revenue Optimization Platform
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        print("   Try running: pip install -r requirements.txt")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("📁 Directory structure verified")

def display_banner():
    """Display startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🚀 Smart Upsell Agent - Enhanced Edition                  ║
║    AI-Powered Revenue Optimization Platform                  ║
║                                                              ║
║    Features:                                                 ║
║    • Customer Segmentation with ML                          ║
║    • Churn Prediction & Prevention                          ║
║    • Multi-Channel Campaign Management                      ║
║    • Real-time Analytics & ROI Tracking                     ║
║    • A/B Testing Framework                                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main startup function"""
    display_banner()

    # Check Python version
    check_python_version()

    # Create directories
    create_directories()

    # Ask user about package installation
    if not os.path.exists('venv') and '--skip-install' not in sys.argv:
        install = input("📦 Install/update required packages? (y/n): ").lower().strip()
        if install in ['y', 'yes']:
            if not install_requirements():
                return

    print("\n🌟 Starting Enhanced Smart Upsell Agent...")
    print("\n📊 Dashboard URLs:")
    print("   • Main Dashboard: http://localhost:5000")
    print("   • Advanced Analytics: http://localhost:5000/analytics")
    print("   • Campaign Manager: http://localhost:5000/campaigns") 
    print("   • Customer Segments: http://localhost:5000/segments")
    print("\n💡 Tips:")
    print("   • Click 'Load Enhanced Demo Data' to get started")
    print("   • Explore AI predictions and customer segments")
    print("   • Create targeted campaigns with high conversion rates")
    print("   • Press Ctrl+C to stop the server\n")

    # Import and run the Flask app
    try:
        from app import app, init_enhanced_db

        # Initialize database
        init_enhanced_db()
        print("✅ Enhanced database initialized")

        # Start the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)

    except ImportError as e:
        print(f"❌ Failed to import app: {e}")
        print("   Make sure all required files are present")
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using Smart Upsell Agent!")
        print("🌟 Your AI-powered revenue optimization platform")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
