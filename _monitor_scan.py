"""Monitor scan and restart if it stops before completion."""
import json
import os
import time
import requests
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)

BASE = "http://localhost:5000"
SCAN_DB = "scan_db.json"
SESSION = requests.Session()

def login():
    """Try to get an authenticated session by reading existing session cookie."""
    # First check if we're already authenticated
    r = SESSION.get(f"{BASE}/api/auth/me")
    if r.status_code == 200:
        data = r.json()
        if data.get("authenticated"):
            print(f"Authenticated as: {data.get('name', '?')}")
            return True
    print("Not authenticated - cannot monitor via API. Will monitor scan_db.json file instead.")
    return False

def get_scan_status_api():
    """Get scan status via API."""
    try:
        r = SESSION.get(f"{BASE}/api/scan/status", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def get_jobs_api():
    """Get jobs via API."""
    try:
        r = SESSION.get(f"{BASE}/api/jobs", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def start_scan_api():
    """Start/continue scan via API."""
    try:
        r = SESSION.post(f"{BASE}/api/scan/start", json={"full": False}, timeout=10)
        print(f"  Start scan response: {r.status_code} - {r.text[:200]}")
        return r.status_code == 200
    except Exception as e:
        print(f"  Failed to start scan: {e}")
        return False

def get_scan_db_stats():
    """Read scan_db.json for current state."""
    if not os.path.exists(SCAN_DB):
        return None
    try:
        with open(SCAN_DB, 'r', encoding='utf-8') as f:
            db = json.load(f)
        stats = db.get("stats", {})
        return {
            "total_scanned": stats.get("total_scanned", 0),
            "total_kept": stats.get("total_kept", 0),
            "partial": stats.get("partial", False),
            "images": len(db.get("images", [])),
            "scan_date": db.get("scan_date", ""),
        }
    except:
        return None

def monitor_with_api():
    """Monitor using API calls."""
    print("Monitoring scan via API...")
    last_scanned = 0
    stall_count = 0

    while True:
        st = get_scan_status_api()
        jobs = get_jobs_api()

        if st:
            progress = st.get("progress", "")
            done = st.get("done", False)
            error = st.get("error")
            cancelled = st.get("cancelled", False)

            print(f"  [{time.strftime('%H:%M:%S')}] {progress} | done={done} error={error} cancelled={cancelled}")

            if done and not error and not cancelled:
                print("SCAN COMPLETED SUCCESSFULLY!")
                # Check final stats
                db_stats = get_scan_db_stats()
                if db_stats:
                    print(f"  Final: {db_stats['total_scanned']} scanned, {db_stats['images']} images, partial={db_stats['partial']}")
                return True

            if done and (error == "Cancelled" or cancelled):
                print("Scan was cancelled/stopped. Restarting...")
                time.sleep(3)
                if start_scan_api():
                    print("  Scan restarted!")
                    time.sleep(5)
                    continue
                else:
                    print("  Could not restart scan.")
                    return False

            if done and error:
                print(f"Scan errored: {error}. Attempting restart...")
                time.sleep(3)
                if start_scan_api():
                    print("  Scan restarted!")
                    time.sleep(5)
                    continue
                else:
                    print(f"  Could not restart. Error was: {error}")
                    return False

        # Check if any scan job is running
        if jobs:
            running = [j for j in jobs.get("jobs", []) if j.get("running")]
            if not running:
                # No running jobs - check if scan is complete
                db_stats = get_scan_db_stats()
                if db_stats and not db_stats["partial"]:
                    print(f"Scan complete! {db_stats['total_scanned']} scanned, {db_stats['images']} images")
                    return True
                elif db_stats and db_stats["partial"]:
                    print(f"Scan stopped (partial). {db_stats['total_scanned']} scanned so far. Restarting...")
                    time.sleep(3)
                    start_scan_api()
                    time.sleep(5)

        time.sleep(15)

def monitor_file_based():
    """Monitor by watching scan_db.json changes."""
    print("Monitoring scan via scan_db.json file changes...")
    last_mtime = 0
    last_scanned = 0
    no_change_count = 0

    while True:
        db_stats = get_scan_db_stats()
        if not db_stats:
            print(f"  [{time.strftime('%H:%M:%S')}] No scan_db.json yet...")
            time.sleep(30)
            continue

        current_mtime = os.path.getmtime(SCAN_DB)
        scanned = db_stats["total_scanned"]

        print(f"  [{time.strftime('%H:%M:%S')}] Scanned: {scanned}, Kept: {db_stats['images']}, Partial: {db_stats['partial']}")

        if not db_stats["partial"] and scanned > 202:
            print(f"SCAN COMPLETED! {scanned} scanned, {db_stats['images']} images kept.")
            return True

        if current_mtime == last_mtime and scanned == last_scanned:
            no_change_count += 1
            if no_change_count >= 8:  # 4 minutes with no change
                print("No progress for 4 minutes. Scan may have stopped. Attempting restart...")
                try:
                    r = requests.post(f"{BASE}/api/scan/start", json={"full": False}, timeout=10)
                    print(f"  Restart response: {r.status_code}")
                    no_change_count = 0
                except Exception as e:
                    print(f"  Restart failed: {e}")
        else:
            no_change_count = 0

        last_mtime = current_mtime
        last_scanned = scanned
        time.sleep(30)

if __name__ == "__main__":
    print("=== Scan Monitor ===")
    print(f"Watching project scan at {BASE}")

    authenticated = login()
    if authenticated:
        monitor_with_api()
    else:
        monitor_file_based()
