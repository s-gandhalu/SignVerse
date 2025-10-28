"""
SignVerse Pro - Enhanced Setup and Run Script
Includes optimized detection parameters and new features
"""

import os
import sys
import subprocess

def print_banner():
    print("\n" + "="*70)
    print("🚀 SignVerse Pro v2.0 - Enhanced Setup & Launch")
    print("="*70)
    print("✨ New Features:")
    print("   • Faster detection (8-frame buffer)")
    print("   • Text-to-Speech for accessibility")
    print("   • Text-to-Sign converter")
    print("   • Improved UI with Apple-style design")
    print("="*70 + "\n")

def check_file_exists(filename):
    """Check if required file exists"""
    exists = os.path.exists(filename)
    status = "✅" if exists else "❌"
    print(f"{status} {filename}: {'Found' if exists else 'MISSING'}")
    return exists

def check_requirements():
    """Check if all required files exist"""
    print("📋 Checking required files...")
    
    files = {
        'app.py': check_file_exists('app.py'),
        'templates/index.html': check_file_exists('templates/index.html'),
        'best.pt': check_file_exists('best.pt')
    }
    
    if not all(files.values()):
        print("\n❌ Missing required files! Please ensure all files are in the correct folders.")
        print("\n📂 Expected structure:")
        print("   signverse-project/")
        print("   ├── app.py")
        print("   ├── best.pt")
        print("   ├── setup_and_run.py")
        print("   └── templates/")
        print("       └── index.html")
        return False
    
    print("\n✅ All required files found!\n")
    return True

def check_python_packages():
    """Check if required packages are installed"""
    print("📦 Checking Python packages...")
    
    required = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'ultralytics': 'ultralytics'
    }
    
    missing = []
    installed_versions = {}
    
    for module, package in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            installed_versions[package] = version
            print(f"✅ {package} ({version})")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\n📥 Install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All packages installed!\n")
    return True

def fix_pytorch_issue():
    """Set environment variables to fix PyTorch 2.6+ loading issues"""
    print("🔧 Configuring PyTorch environment...")
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print("✅ Environment configured\n")

def print_optimizations():
    """Display optimization information"""
    print("⚡ Performance Optimizations:")
    print("   • Buffer Size: 8 frames (reduced from 15)")
    print("   • Stability Threshold: 6 frames (reduced from 12)")
    print("   • Min Confidence: 0.45 (optimized)")
    print("   • Processing Interval: 100ms")
    print("   • Expected response time: ~0.8-1.2 seconds")
    print()

def start_server():
    """Start the Flask server"""
    print("="*70)
    print("🚀 Starting SignVerse Pro Server...")
    print("="*70)
    print("\n📌 Access the application at: http://127.0.0.1:5000")
    print("📌 Press CTRL+C to stop the server")
    print("\n🎯 Features Available:")
    print("   • Real-time sign language detection")
    print("   • Text-to-Speech (for blind users)")
    print("   • Text-to-Sign converter (for learning)")
    print("   • Manual space insertion")
    print("   • Copy sentence to clipboard")
    print("   • Screenshot capture")
    print("\n💡 Tips:")
    print("   • Hold gesture steady for ~1 second")
    print("   • Use 'Add Space' button between words")
    print("   • Click 'Speak' to hear the sentence")
    print("   • Switch to 'Text → Sign' tab to learn signs")
    print("="*70 + "\n")
    
    # Set environment variables
    env = os.environ.copy()
    env['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Run app.py
    try:
        subprocess.run([sys.executable, 'app.py'], env=env)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

def main():
    print_banner()
    
    # Check everything
    if not check_requirements():
        input("\nPress Enter to exit...")
        return
    
    if not check_python_packages():
        input("\nPress Enter to exit...")
        return
    
    fix_pytorch_issue()
    print_optimizations()
    
    # Start server
    try:
        start_server()
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()