"""
Build E-z Photo Organizer as a standalone Windows .exe

Usage:
    python build_exe.py

Output:
    dist/E-z Photo Organizer.exe
"""

import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 60)
    print("  Building E-z Photo Organizer .exe")
    print("=" * 60)

    # Ensure PyInstaller is available
    try:
        import PyInstaller
        print(f"  PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("  Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Core app files that get imported at runtime
    extra_modules = [
        "auth",
        "curate",
        "event_agent",
    ]

    # Data files to bundle (templates dir)
    templates_dir = os.path.join(PROJECT_DIR, "templates")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",           # onedir is faster to build and start than onefile
        "--console",          # console so user can see startup messages
        "--name", "E-z Photo Organizer",

        # App icon (if exists)
    ]

    icon_path = os.path.join(PROJECT_DIR, "icon.ico")
    if os.path.exists(icon_path):
        cmd += ["--icon", icon_path]

    # Add data files
    if os.path.isdir(templates_dir):
        cmd += ["--add-data", f"{templates_dir};templates"]

    # Hidden imports (modules loaded dynamically at runtime)
    hidden = [
        "flask",
        "PIL", "PIL.Image", "PIL.ImageStat", "PIL.ExifTags",
        "cv2",
        "numpy",
        "face_recognition", "dlib",
        "reverse_geocoder",
        "auth", "curate", "event_agent",
        "werkzeug", "werkzeug.serving", "werkzeug.debug",
        "jinja2", "markupsafe", "itsdangerous", "click", "blinker",
    ]
    for h in hidden:
        cmd += ["--hidden-import", h]

    # Collect all submodules for packages that PyInstaller misses
    for pkg in ["flask", "werkzeug", "jinja2", "cv2", "PIL"]:
        cmd += ["--collect-submodules", pkg]

    # Extra .py files that are imported at runtime
    for mod in extra_modules:
        mod_path = os.path.join(PROJECT_DIR, f"{mod}.py")
        if os.path.exists(mod_path):
            cmd += ["--add-data", f"{mod_path};."]

    # Main script
    cmd.append(os.path.join(PROJECT_DIR, "app.py"))

    print(f"\n  Running PyInstaller...")
    print(f"  Command: {' '.join(cmd[-5:])}")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_DIR)

    if result.returncode == 0:
        dist_dir = os.path.join(PROJECT_DIR, "dist", "E-z Photo Organizer")
        exe_path = os.path.join(dist_dir, "E-z Photo Organizer.exe")
        print()
        print("=" * 60)
        print("  BUILD SUCCESSFUL!")
        print(f"  Exe: {exe_path}")
        print(f"  Folder: {dist_dir}")
        print()
        print("  To run: double-click 'E-z Photo Organizer.exe'")
        print("  To distribute: copy the entire folder.")
        print("=" * 60)
    else:
        print("\n  BUILD FAILED. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
