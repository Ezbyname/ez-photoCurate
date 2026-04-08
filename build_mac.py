"""
Build E-z Photo Organizer as a macOS .app bundle

Usage:
    python build_mac.py

Output:
    dist/E-z Photo Organizer.app
"""

import subprocess
import sys
import os
import platform

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60)
    print("  Building E-z Photo Organizer .app (macOS)")
    print(f"  Architecture: {platform.machine()}")
    print("=" * 60)

    # Ensure PyInstaller is available
    try:
        import PyInstaller
        print(f"  PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("  Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Core app files imported at runtime
    extra_modules = [
        "auth",
        "curate",
        "event_agent",
    ]

    templates_dir = os.path.join(PROJECT_DIR, "templates")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",           # .app bundle (no terminal window)
        "--name", "E-z Photo Organizer",
    ]

    # App icon — convert PNG to icns on macOS
    icon_icns = os.path.join(PROJECT_DIR, "app_icon.icns")
    icon_png = os.path.join(PROJECT_DIR, "app_icon.png")
    if not os.path.exists(icon_icns) and os.path.exists(icon_png):
        print("  Converting PNG icon to icns...")
        iconset = os.path.join(PROJECT_DIR, "app_icon.iconset")
        os.makedirs(iconset, exist_ok=True)
        from PIL import Image
        img = Image.open(icon_png)
        for size in [16, 32, 64, 128, 256, 512]:
            resized = img.resize((size, size), Image.LANCZOS)
            resized.save(os.path.join(iconset, f"icon_{size}x{size}.png"))
            double = img.resize((size * 2, size * 2), Image.LANCZOS)
            double.save(os.path.join(iconset, f"icon_{size}x{size}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icon_icns], check=True)
        import shutil
        shutil.rmtree(iconset)
        print("  Created app_icon.icns")
    if os.path.exists(icon_icns):
        cmd += ["--icon", icon_icns]

    # Data files — macOS uses : as separator (not ;)
    if os.path.isdir(templates_dir):
        cmd += ["--add-data", f"{templates_dir}:templates"]

    # Hidden imports
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

    # Collect submodules
    for pkg in ["flask", "werkzeug", "jinja2", "cv2", "PIL"]:
        cmd += ["--collect-submodules", pkg]

    # Extra .py files
    for mod in extra_modules:
        mod_path = os.path.join(PROJECT_DIR, f"{mod}.py")
        if os.path.exists(mod_path):
            cmd += ["--add-data", f"{mod_path}:."]

    # macOS-specific: set bundle identifier
    cmd += ["--osx-bundle-identifier", "com.ezphoto.organizer"]

    # Main script
    cmd.append(os.path.join(PROJECT_DIR, "app.py"))

    print(f"\n  Running PyInstaller...")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_DIR)

    if result.returncode == 0:
        app_path = os.path.join(PROJECT_DIR, "dist", "E-z Photo Organizer.app")
        print()
        print("=" * 60)
        print("  BUILD SUCCESSFUL!")
        print(f"  App: {app_path}")
        print()
        print("  To run: double-click 'E-z Photo Organizer.app'")
        print("  To distribute: zip the .app and share it.")
        print("=" * 60)
    else:
        print("\n  BUILD FAILED. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
