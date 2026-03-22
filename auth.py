"""
Authentication module for E-z Photo Organizer.
SQLite-based user management with email/SMS verification.
"""

import os
import re
import sqlite3
import secrets
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps

from werkzeug.security import generate_password_hash, check_password_hash
from flask import request, jsonify, session, redirect

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
_db_lock = threading.Lock()

# ── Password policy ──────────────────────────────────────────────────────────

PASSWORD_POLICY = {
    "min_length": 8,
    "require_upper": True,
    "require_special": True,
    "special_chars": "!@#$%^&*()_+-=[]{}|;:',.<>?/~`",
}


def validate_password(pw):
    """Returns (ok, error_message)."""
    if len(pw) < PASSWORD_POLICY["min_length"]:
        return False, f"Password must be at least {PASSWORD_POLICY['min_length']} characters."
    if PASSWORD_POLICY["require_upper"] and not any(c.isupper() for c in pw):
        return False, "Password must contain at least 1 uppercase letter."
    if PASSWORD_POLICY["require_special"] and not any(c in PASSWORD_POLICY["special_chars"] for c in pw):
        return False, "Password must contain at least 1 special character (!@#$ etc.)."
    return True, ""


def validate_contact(contact):
    """Validate email or phone number. Returns (type, normalized, error)."""
    contact = contact.strip()
    # Email
    if "@" in contact:
        if re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', contact):
            return "email", contact.lower(), ""
        return None, None, "Invalid email address."
    # Phone — strip non-digits, allow + prefix
    digits = re.sub(r'[^\d+]', '', contact)
    if len(digits) >= 8:
        return "phone", digits, ""
    return None, None, "Invalid email or phone number."


# ── Database ─────────────────────────────────────────────────────────────────

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _db_lock:
        conn = _get_db()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL DEFAULT '',
                age INTEGER,
                contact TEXT UNIQUE NOT NULL,
                contact_type TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                verified INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                last_login TEXT
            );
            CREATE TABLE IF NOT EXISTS verification_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contact TEXT NOT NULL,
                code TEXT NOT NULL,
                purpose TEXT DEFAULT 'signup',
                created_at TEXT DEFAULT (datetime('now')),
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0
            );
        """)
        conn.close()


def get_user(contact):
    with _db_lock:
        conn = _get_db()
        user = conn.execute("SELECT * FROM users WHERE contact = ?", (contact,)).fetchone()
        conn.close()
        return dict(user) if user else None


def create_user(contact, contact_type, password, full_name="", age=None):
    pw_hash = generate_password_hash(password)
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                "INSERT INTO users (contact, contact_type, password_hash, full_name, age) VALUES (?, ?, ?, ?, ?)",
                (contact, contact_type, pw_hash, full_name, age))
            conn.commit()
            conn.close()
            return True, ""
        except sqlite3.IntegrityError:
            conn.close()
            return False, "Account already exists."


def verify_user(contact):
    with _db_lock:
        conn = _get_db()
        conn.execute("UPDATE users SET verified = 1 WHERE contact = ?", (contact,))
        conn.commit()
        conn.close()


def reset_password(contact, new_password):
    pw_hash = generate_password_hash(new_password)
    with _db_lock:
        conn = _get_db()
        conn.execute("UPDATE users SET password_hash = ? WHERE contact = ?", (pw_hash, contact))
        conn.commit()
        conn.close()


def generate_temp_password():
    """Generate a readable temporary password that meets policy."""
    import string
    alpha = string.ascii_lowercase
    upper = string.ascii_uppercase
    digits = string.digits
    specials = "!@#$"
    # Build: 4 lowercase + 1 upper + 1 digit + 2 special = 8 chars
    pw = (secrets.choice(upper) +
          ''.join(secrets.choice(alpha) for _ in range(4)) +
          secrets.choice(digits) +
          ''.join(secrets.choice(specials) for _ in range(2)))
    # Shuffle
    pw_list = list(pw)
    secrets.SystemRandom().shuffle(pw_list)
    return ''.join(pw_list)


def update_last_login(contact):
    with _db_lock:
        conn = _get_db()
        conn.execute("UPDATE users SET last_login = datetime('now') WHERE contact = ?", (contact,))
        conn.commit()
        conn.close()


# ── Verification codes ───────────────────────────────────────────────────────

def generate_code(contact, purpose="signup"):
    code = f"{secrets.randbelow(1000000):06d}"
    expires = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    with _db_lock:
        conn = _get_db()
        # Invalidate old codes
        conn.execute(
            "UPDATE verification_codes SET used = 1 WHERE contact = ? AND purpose = ? AND used = 0",
            (contact, purpose))
        conn.execute(
            "INSERT INTO verification_codes (contact, code, purpose, expires_at) VALUES (?, ?, ?, ?)",
            (contact, code, purpose, expires))
        conn.commit()
        conn.close()
    return code


def check_code(contact, code, purpose="signup"):
    """Returns (valid, error_message)."""
    with _db_lock:
        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM verification_codes WHERE contact = ? AND code = ? AND purpose = ? AND used = 0 ORDER BY id DESC LIMIT 1",
            (contact, code, purpose)).fetchone()
        if not row:
            conn.close()
            return False, "Invalid verification code."
        if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
            conn.close()
            return False, "Code has expired. Request a new one."
        conn.execute("UPDATE verification_codes SET used = 1 WHERE id = ?", (row["id"],))
        conn.commit()
        conn.close()
        return True, ""


# ── Send verification ────────────────────────────────────────────────────────

# SMTP config — set via environment variables or auth_config.json
def _load_send_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth_config.json")
    if os.path.isfile(cfg_path):
        import json
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {
        "smtp_host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
        "smtp_user": os.environ.get("SMTP_USER", ""),
        "smtp_pass": os.environ.get("SMTP_PASS", ""),
        "smtp_from": os.environ.get("SMTP_FROM", ""),
        "twilio_sid": os.environ.get("TWILIO_SID", ""),
        "twilio_token": os.environ.get("TWILIO_TOKEN", ""),
        "twilio_from": os.environ.get("TWILIO_FROM", ""),
    }


def send_email_code(email, code):
    cfg = _load_send_config()
    if not cfg.get("smtp_user"):
        # Dev mode — return code so UI can show it
        print(f"[AUTH] Verification code for {email}: {code}")
        return True, code  # second value = dev_code
    try:
        msg = MIMEMultipart()
        msg["From"] = cfg.get("smtp_from") or cfg["smtp_user"]
        msg["To"] = email
        msg["Subject"] = "E-z Photo Organizer — Verification Code"
        body = f"Your verification code is: {code}\n\nThis code expires in 10 minutes."
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_pass"])
            server.send_message(msg)
        return True, ""
    except Exception as e:
        print(f"[AUTH] Email send failed: {e}")
        return False, f"Failed to send email: {e}"


def send_sms_code(phone, code):
    cfg = _load_send_config()
    if not cfg.get("twilio_sid"):
        # Dev mode — return code so UI can show it
        print(f"[AUTH] Verification code for {phone}: {code}")
        return True, code
    try:
        from twilio.rest import Client
        client = Client(cfg["twilio_sid"], cfg["twilio_token"])
        client.messages.create(
            body=f"E-z Photo Organizer verification code: {code}",
            from_=cfg["twilio_from"],
            to=phone)
        return True, ""
    except Exception as e:
        print(f"[AUTH] SMS send failed: {e}")
        return False, f"Failed to send SMS: {e}"


def send_verification(contact, contact_type, code):
    if contact_type == "email":
        return send_email_code(contact, code)
    else:
        return send_sms_code(contact, code)


# ── Flask integration ────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Not authenticated"}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


def register_auth_routes(app):
    """Register all auth-related routes on the Flask app."""

    @app.route("/api/auth/signup", methods=["POST"])
    def auth_signup():
        data = request.json or {}
        full_name = data.get("full_name", "").strip()
        age_raw = data.get("age")
        contact_raw = data.get("contact", "").strip()
        password = data.get("password", "")
        confirm = data.get("confirm_password", "")

        if not full_name:
            return jsonify({"error": "Full name is required."}), 400

        age = None
        if age_raw not in (None, "", 0):
            try:
                age = int(age_raw)
                if age < 1 or age > 150:
                    return jsonify({"error": "Please enter a valid age."}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "Please enter a valid age."}), 400

        ctype, contact, err = validate_contact(contact_raw)
        if err:
            return jsonify({"error": err}), 400

        ok, err = validate_password(password)
        if not ok:
            return jsonify({"error": err}), 400

        if password != confirm:
            return jsonify({"error": "Passwords do not match."}), 400

        # Check existing
        existing = get_user(contact)
        if existing and existing["verified"]:
            return jsonify({"error": "Account already exists. Please log in."}), 400
        if existing and not existing["verified"]:
            # Re-send verification for unverified account
            code = generate_code(contact, "signup")
            ok, dev_code = send_verification(contact, ctype, code)
            if not ok:
                return jsonify({"error": dev_code}), 500
            resp = {"ok": True, "contact": contact, "type": ctype,
                    "message": "Verification code re-sent."}
            if dev_code and dev_code == code:
                resp["dev_code"] = code
            return jsonify(resp)

        ok, err = create_user(contact, ctype, password, full_name, age)
        if not ok:
            return jsonify({"error": err}), 400

        code = generate_code(contact, "signup")
        ok, dev_code = send_verification(contact, ctype, code)
        if not ok:
            return jsonify({"error": dev_code, "warning": "Account created but verification failed."}), 500

        resp = {"ok": True, "contact": contact, "type": ctype,
                "message": f"Verification code sent to {contact}."}
        if dev_code and dev_code == code:
            resp["dev_code"] = code
        return jsonify(resp)

    @app.route("/api/auth/verify", methods=["POST"])
    def auth_verify():
        data = request.json or {}
        contact_raw = data.get("contact", "").strip()
        code = data.get("code", "").strip()

        ctype, contact, err = validate_contact(contact_raw)
        if err:
            return jsonify({"error": err}), 400

        ok, err = check_code(contact, code, "signup")
        if not ok:
            return jsonify({"error": err}), 400

        verify_user(contact)
        update_last_login(contact)
        session["user"] = contact
        session.permanent = True
        return jsonify({"ok": True, "message": "Account verified! Welcome."})

    @app.route("/api/auth/login", methods=["POST"])
    def auth_login():
        data = request.json or {}
        contact_raw = data.get("contact", "").strip()
        password = data.get("password", "")

        ctype, contact, err = validate_contact(contact_raw)
        if err:
            return jsonify({"error": err}), 400

        user = get_user(contact)
        if not user:
            return jsonify({"error": "Account not found. Please sign up."}), 400

        if not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "Invalid password."}), 400

        if not user["verified"]:
            # Send new verification code
            code = generate_code(contact, "signup")
            ok, dev_code = send_verification(contact, ctype, code)
            resp = {"error": "Account not verified.", "needs_verify": True,
                    "contact": contact, "type": ctype}
            if dev_code and dev_code == code:
                resp["dev_code"] = code
            return jsonify(resp), 400

        update_last_login(contact)
        session["user"] = contact
        session.permanent = True
        return jsonify({"ok": True, "message": "Logged in."})

    @app.route("/api/auth/resend", methods=["POST"])
    def auth_resend():
        data = request.json or {}
        contact_raw = data.get("contact", "").strip()

        ctype, contact, err = validate_contact(contact_raw)
        if err:
            return jsonify({"error": err}), 400

        code = generate_code(contact, "signup")
        ok, dev_code = send_verification(contact, ctype, code)
        if not ok:
            return jsonify({"error": dev_code}), 500

        resp = {"ok": True, "message": f"New code sent to {contact}."}
        if dev_code and dev_code == code:
            resp["dev_code"] = code
        return jsonify(resp)

    @app.route("/api/auth/forgot", methods=["POST"])
    def auth_forgot():
        data = request.json or {}
        contact_raw = data.get("contact", "").strip()

        ctype, contact, err = validate_contact(contact_raw)
        if err:
            return jsonify({"error": err}), 400

        user = get_user(contact)
        if not user:
            # Don't reveal if account exists — always say "sent"
            return jsonify({"ok": True, "message": "If an account exists, a new password has been sent."})

        # Generate temp password and send it
        temp_pw = generate_temp_password()
        reset_password(contact, temp_pw)

        cfg = _load_send_config()
        if ctype == "email":
            if not cfg.get("smtp_user"):
                print(f"[AUTH] New password for {contact}: {temp_pw}")
                ok = True
            else:
                try:
                    msg = MIMEMultipart()
                    msg["From"] = cfg.get("smtp_from") or cfg["smtp_user"]
                    msg["To"] = contact
                    msg["Subject"] = "E-z Photo Organizer — Password Reset"
                    body = f"Your new temporary password is: {temp_pw}\n\nPlease log in and change it."
                    msg.attach(MIMEText(body, "plain"))
                    with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                        server.starttls()
                        server.login(cfg["smtp_user"], cfg["smtp_pass"])
                        server.send_message(msg)
                    ok = True
                except Exception as e:
                    return jsonify({"error": f"Failed to send email: {e}"}), 500
        else:
            if not cfg.get("twilio_sid"):
                print(f"[AUTH] New password for {contact}: {temp_pw}")
                ok = True
            else:
                try:
                    from twilio.rest import Client
                    client = Client(cfg["twilio_sid"], cfg["twilio_token"])
                    client.messages.create(
                        body=f"E-z Photo Organizer new password: {temp_pw}",
                        from_=cfg["twilio_from"], to=contact)
                    ok = True
                except Exception as e:
                    return jsonify({"error": f"Failed to send SMS: {e}"}), 500

        return jsonify({"ok": True, "message": "If an account exists, a new password has been sent."})

    @app.route("/api/auth/logout", methods=["POST"])
    def auth_logout():
        session.pop("user", None)
        return jsonify({"ok": True})

    @app.route("/api/auth/me")
    def auth_me():
        contact = session.get("user")
        if contact:
            user_data = get_user(contact)
            name = user_data["full_name"] if user_data else contact
            return jsonify({"authenticated": True, "user": contact, "name": name or contact})
        return jsonify({"authenticated": False})

    @app.route("/login")
    def login_page():
        return LOGIN_HTML


# ── Login/Signup HTML ────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-z Photo Organizer — Sign In</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
}
.auth-card {
    background: white; border-radius: 16px; padding: 40px 36px; width: 400px;
    box-shadow: 0 20px 60px rgba(0,0,0,.2);
}
.auth-card h1 { font-size: 1.6em; color: #2b6cb0; text-align: center; margin-bottom: 4px; }
.auth-card .subtitle { text-align: center; color: #718096; font-size: .9em; margin-bottom: 28px; }
.form-group { margin-bottom: 16px; }
.form-group label { display: block; font-size: .85em; font-weight: 600; color: #4a5568; margin-bottom: 5px; }
.form-group input {
    width: 100%; padding: 10px 14px; border: 1px solid #e2e8f0; border-radius: 8px;
    font-size: .95em; transition: border-color .2s; outline: none;
}
.form-group input:focus { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102,126,234,.15); }
.pw-rules { font-size: .75em; color: #a0aec0; margin-top: 4px; }
.pw-rules span.ok { color: #38a169; }
.pw-rules span.fail { color: #e53e3e; }
.btn {
    width: 100%; padding: 12px; border: none; border-radius: 8px; font-size: 1em;
    font-weight: 600; cursor: pointer; transition: all .2s;
}
.btn-primary { background: #667eea; color: white; }
.btn-primary:hover { background: #5a67d8; }
.btn-primary:disabled { background: #a0aec0; cursor: not-allowed; }
.btn-link { background: none; color: #667eea; font-size: .9em; margin-top: 12px; padding: 8px; }
.btn-link:hover { text-decoration: underline; }
.error-msg { background: #fff5f5; color: #e53e3e; font-size: .85em; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; display: none; }
.success-msg { background: #f0fff4; color: #38a169; font-size: .85em; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; display: none; }
.verify-section { display: none; }
.code-input {
    display: flex; gap: 8px; justify-content: center; margin: 16px 0;
}
.code-input input {
    width: 46px; height: 52px; text-align: center; font-size: 1.4em; font-weight: 700;
    border: 2px solid #e2e8f0; border-radius: 10px; outline: none;
}
.code-input input:focus { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102,126,234,.15); }
.tabs { display: flex; margin-bottom: 24px; border-bottom: 2px solid #e2e8f0; }
.tab {
    flex: 1; padding: 10px; text-align: center; font-size: .95em; font-weight: 600;
    color: #a0aec0; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all .2s;
}
.tab.active { color: #667eea; border-bottom-color: #667eea; }
.resend-link { font-size: .85em; color: #667eea; cursor: pointer; text-align: center; margin-top: 8px; }
.resend-link:hover { text-decoration: underline; }
.resend-link.disabled { color: #a0aec0; cursor: not-allowed; }
</style>
</head>
<body>
<div class="auth-card">
    <h1>E-z Photo Organizer</h1>
    <p class="subtitle">Sign in to manage your collections</p>

    <div class="tabs" id="auth-tabs">
        <div class="tab active" onclick="switchTab('login')">Log In</div>
        <div class="tab" onclick="switchTab('signup')">Sign Up</div>
    </div>

    <div class="error-msg" id="auth-error"></div>
    <div class="success-msg" id="auth-success"></div>

    <!-- Login / Signup form -->
    <div id="form-section">
        <div class="form-group signup-only" style="display:none;">
            <label>Full Name</label>
            <input type="text" id="auth-name" placeholder="Your full name" autocomplete="name">
        </div>
        <div class="form-group signup-only" style="display:none;">
            <label>Age <span style="font-weight:400; color:#a0aec0;">(optional)</span></label>
            <input type="number" id="auth-age" placeholder="" min="1" max="150" style="width:100px;">
        </div>
        <div class="form-group">
            <label>Email or Phone Number</label>
            <input type="text" id="auth-contact" placeholder="you@example.com or +1234567890" autocomplete="username">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" id="auth-password" placeholder="Enter password" autocomplete="current-password" oninput="checkPwRules()">
            <div class="pw-rules" id="pw-rules" style="display:none;">
                <span id="rule-len">8+ characters</span> &middot;
                <span id="rule-upper">1 uppercase</span> &middot;
                <span id="rule-special">1 special char (!@#$)</span>
            </div>
        </div>
        <div class="form-group signup-only" style="display:none;">
            <label>Confirm Password</label>
            <input type="password" id="auth-confirm" placeholder="Re-enter password" autocomplete="new-password">
        </div>
        <button class="btn btn-primary" id="auth-submit" onclick="submitAuth()">Log In</button>
        <div id="forgot-link" style="text-align:center; margin-top:10px;">
            <span style="font-size:.85em; color:#667eea; cursor:pointer;" onclick="showForgot()">Forgot password?</span>
        </div>
    </div>

    <!-- Forgot password section -->
    <div id="forgot-section" style="display:none;">
        <p style="text-align:center; color:#4a5568; font-size:.9em; margin-bottom:16px;">
            Enter your email or phone number and we'll send you a new password.
        </p>
        <div class="form-group">
            <label>Email or Phone Number</label>
            <input type="text" id="forgot-contact" placeholder="you@example.com or +1234567890">
        </div>
        <button class="btn btn-primary" onclick="submitForgot()">Send New Password</button>
        <button class="btn btn-link" onclick="backToForm()">Back to login</button>
    </div>

    <!-- Verification section -->
    <div class="verify-section" id="verify-section">
        <p style="text-align:center; color:#4a5568; font-size:.9em; margin-bottom:4px;">
            Enter the 6-digit code sent to
        </p>
        <p style="text-align:center; font-weight:600; color:#2d3748; margin-bottom:16px;" id="verify-contact-display"></p>

        <div class="code-input" id="code-inputs">
            <input type="text" maxlength="1" inputmode="numeric" pattern="[0-9]" data-idx="0">
            <input type="text" maxlength="1" inputmode="numeric" pattern="[0-9]" data-idx="1">
            <input type="text" maxlength="1" inputmode="numeric" pattern="[0-9]" data-idx="2">
            <input type="text" maxlength="1" inputmode="numeric" pattern="[0-9]" data-idx="3">
            <input type="text" maxlength="1" inputmode="numeric" pattern="[0-9]" data-idx="4">
            <input type="text" maxlength="1" inputmode="numeric" pattern="[0-9]" data-idx="5">
        </div>

        <button class="btn btn-primary" id="verify-submit" onclick="submitVerify()">Verify</button>
        <div class="resend-link" id="resend-link" onclick="resendCode()">Resend code</div>
        <button class="btn btn-link" onclick="backToForm()">Back</button>
    </div>
</div>

<script>
let currentTab = 'login';
let pendingContact = '';
let pendingType = '';

function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach((t, i) => {
        t.classList.toggle('active', (i === 0 && tab === 'login') || (i === 1 && tab === 'signup'));
    });
    document.getElementById('auth-submit').textContent = tab === 'login' ? 'Log In' : 'Sign Up';
    document.getElementById('pw-rules').style.display = tab === 'signup' ? 'block' : 'none';
    document.getElementById('forgot-link').style.display = tab === 'login' ? 'block' : 'none';
    document.getElementById('auth-password').autocomplete = tab === 'login' ? 'current-password' : 'new-password';
    document.querySelectorAll('.signup-only').forEach(el => el.style.display = tab === 'signup' ? 'block' : 'none');
    hideMessages();
}

function hideMessages() {
    document.getElementById('auth-error').style.display = 'none';
    document.getElementById('auth-success').style.display = 'none';
}

function showError(msg) {
    const el = document.getElementById('auth-error');
    el.textContent = msg;
    el.style.display = 'block';
    document.getElementById('auth-success').style.display = 'none';
}

function showSuccess(msg) {
    const el = document.getElementById('auth-success');
    el.textContent = msg;
    el.style.display = 'block';
    document.getElementById('auth-error').style.display = 'none';
}

function checkPwRules() {
    const pw = document.getElementById('auth-password').value;
    const len = pw.length >= 8;
    const upper = /[A-Z]/.test(pw);
    const special = /[!@#$%^&*()_+\-=\[\]{}|;:',.<>?\/~`]/.test(pw);
    document.getElementById('rule-len').className = len ? 'ok' : 'fail';
    document.getElementById('rule-upper').className = upper ? 'ok' : 'fail';
    document.getElementById('rule-special').className = special ? 'ok' : 'fail';
}

async function submitAuth() {
    hideMessages();
    const contact = document.getElementById('auth-contact').value.trim();
    const password = document.getElementById('auth-password').value;

    if (currentTab === 'signup') {
        const name = document.getElementById('auth-name').value.trim();
        if (!name) { showError('Please enter your full name.'); return; }
        const confirm = document.getElementById('auth-confirm').value;
        if (password !== confirm) { showError('Passwords do not match.'); return; }
    }

    if (!contact) { showError('Please enter your email or phone number.'); return; }
    if (!password) { showError('Please enter your password.'); return; }

    const btn = document.getElementById('auth-submit');
    btn.disabled = true;
    btn.textContent = 'Please wait...';

    try {
        const endpoint = currentTab === 'login' ? '/api/auth/login' : '/api/auth/signup';
        const payload = {contact, password};
        if (currentTab === 'signup') {
            payload.full_name = document.getElementById('auth-name').value.trim();
            payload.age = document.getElementById('auth-age').value || null;
            payload.confirm_password = document.getElementById('auth-confirm').value;
        }
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (res.ok && data.ok) {
            if (currentTab === 'login') {
                window.location.href = '/';
            } else {
                pendingContact = data.contact;
                pendingType = data.type;
                showVerifySection();
                if (data.dev_code) {
                    showSuccess('Dev mode — your code is: ' + data.dev_code);
                } else {
                    showSuccess(data.message);
                }
            }
        } else {
            if (data.needs_verify) {
                pendingContact = data.contact;
                pendingType = data.type;
                showVerifySection();
                if (data.dev_code) {
                    showSuccess('Dev mode — your code is: ' + data.dev_code);
                } else {
                    showSuccess('Verification code sent. Please check your ' + (data.type === 'email' ? 'email' : 'phone') + '.');
                }
            } else {
                showError(data.error || 'Something went wrong.');
            }
        }
    } catch(e) {
        showError('Connection error. Please try again.');
    }

    btn.disabled = false;
    btn.textContent = currentTab === 'login' ? 'Log In' : 'Sign Up';
}

function showVerifySection() {
    document.getElementById('form-section').style.display = 'none';
    document.getElementById('auth-tabs').style.display = 'none';
    document.getElementById('verify-section').style.display = 'block';
    document.getElementById('verify-contact-display').textContent = pendingContact;
    // Focus first input
    const inputs = document.querySelectorAll('#code-inputs input');
    inputs.forEach(inp => inp.value = '');
    inputs[0].focus();
}

function backToForm() {
    document.getElementById('form-section').style.display = 'block';
    document.getElementById('auth-tabs').style.display = 'flex';
    document.getElementById('verify-section').style.display = 'none';
    document.getElementById('forgot-section').style.display = 'none';
    hideMessages();
}

// Code input auto-advance and paste support
document.querySelectorAll('#code-inputs input').forEach((inp, i, all) => {
    inp.addEventListener('input', function() {
        this.value = this.value.replace(/[^0-9]/g, '');
        if (this.value && i < 5) all[i + 1].focus();
        if (i === 5 && this.value) submitVerify();
    });
    inp.addEventListener('keydown', function(e) {
        if (e.key === 'Backspace' && !this.value && i > 0) all[i - 1].focus();
    });
    inp.addEventListener('paste', function(e) {
        e.preventDefault();
        const text = (e.clipboardData || window.clipboardData).getData('text').replace(/\D/g, '');
        for (let j = 0; j < 6 && j < text.length; j++) {
            all[j].value = text[j];
        }
        if (text.length >= 6) submitVerify();
        else all[Math.min(text.length, 5)].focus();
    });
});

async function submitVerify() {
    hideMessages();
    const inputs = document.querySelectorAll('#code-inputs input');
    const code = Array.from(inputs).map(i => i.value).join('');
    if (code.length < 6) { showError('Please enter all 6 digits.'); return; }

    const btn = document.getElementById('verify-submit');
    btn.disabled = true;
    btn.textContent = 'Verifying...';

    try {
        const res = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({contact: pendingContact, code})
        });
        const data = await res.json();
        if (res.ok && data.ok) {
            showSuccess(data.message);
            setTimeout(() => { window.location.href = '/'; }, 1000);
        } else {
            showError(data.error || 'Verification failed.');
        }
    } catch(e) {
        showError('Connection error.');
    }

    btn.disabled = false;
    btn.textContent = 'Verify';
}

async function resendCode() {
    const link = document.getElementById('resend-link');
    if (link.classList.contains('disabled')) return;
    link.classList.add('disabled');
    link.textContent = 'Sending...';

    try {
        const res = await fetch('/api/auth/resend', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({contact: pendingContact})
        });
        const data = await res.json();
        if (res.ok) {
            if (data.dev_code) {
                showSuccess('Dev mode — your code is: ' + data.dev_code);
            } else {
                showSuccess(data.message);
            }
        } else {
            showError(data.error || 'Failed to resend.');
        }
    } catch(e) {
        showError('Connection error.');
    }

    // Cooldown 30s
    let sec = 30;
    link.textContent = 'Resend code (' + sec + 's)';
    const timer = setInterval(() => {
        sec--;
        if (sec <= 0) {
            clearInterval(timer);
            link.textContent = 'Resend code';
            link.classList.remove('disabled');
        } else {
            link.textContent = 'Resend code (' + sec + 's)';
        }
    }, 1000);
}

function showForgot() {
    hideMessages();
    document.getElementById('form-section').style.display = 'none';
    document.getElementById('auth-tabs').style.display = 'none';
    document.getElementById('forgot-section').style.display = 'block';
    document.getElementById('forgot-contact').value = document.getElementById('auth-contact').value;
    document.getElementById('forgot-contact').focus();
}

async function submitForgot() {
    hideMessages();
    const contact = document.getElementById('forgot-contact').value.trim();
    if (!contact) { showError('Please enter your email or phone number.'); return; }

    try {
        const res = await fetch('/api/auth/forgot', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({contact})
        });
        const data = await res.json();
        if (res.ok) {
            showSuccess(data.message + ' You can now log in with the new password.');
            setTimeout(() => backToForm(), 3000);
        } else {
            showError(data.error || 'Something went wrong.');
        }
    } catch(e) {
        showError('Connection error.');
    }
}

// Enter key submits
document.getElementById('auth-contact').addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('auth-password').focus(); });
document.getElementById('auth-password').addEventListener('keydown', e => { if (e.key === 'Enter') submitAuth(); });
</script>
</body>
</html>"""
