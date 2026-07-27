import os, sqlite3, random, string, hashlib, base64, pickle, json
from datetime import datetime, timedelta
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, flash, g, Response)
from flask_mail import Mail, Message
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = "ATM_FLASK_SECRET_KEY_2025_SECURE"

# ─────────────────────────────────────────────
# MAIL CONFIG
# ─────────────────────────────────────────────
app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 465
app.config['MAIL_USERNAME'] = 'hariviki7895@gmail.com'
app.config['MAIL_PASSWORD'] = 'kmvwrwphnjsfamtu'
app.config['MAIL_USE_TLS']  = False
app.config['MAIL_USE_SSL']  = True
app.config['MAIL_DEFAULT_SENDER'] = ('SecureATM', 'hariviki7895@gmail.com')

mail = Mail(app)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATABASE        = os.path.join(app.root_path, "instance", "atm.db")
FACE_DATA_DIR   = os.path.join(app.root_path, "face_data")
MODEL_PATH      = os.path.join(FACE_DATA_DIR, "face_model.pkl")
OTP_EXPIRE_MIN  = 5
MAX_FACE_FAILS  = 3
MAX_OTP_FAILS   = 3
LOCK_MINUTES    = 10

os.makedirs(FACE_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(app.root_path, "instance"), exist_ok=True)

# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
    return db

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def query_db(sql, args=(), one=False):
    cur = get_db().execute(sql, args)
    rv  = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def exec_db(sql, args=()):
    db  = get_db()
    cur = db.execute(sql, args)
    db.commit()
    return cur.lastrowid

# ─────────────────────────────────────────────
# INIT DB
# ─────────────────────────────────────────────
def init_db():
    db = sqlite3.connect(DATABASE)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS bank_user (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT    NOT NULL UNIQUE,
        dob             TEXT    NOT NULL,
        account_number  TEXT    NOT NULL UNIQUE,
        phone           TEXT    NOT NULL,
        email           TEXT    NOT NULL,
        address         TEXT,
        saving_type     TEXT    DEFAULT 'SAVINGS',
        pin_hash        TEXT    NOT NULL,
        face_id         INTEGER UNIQUE,
        is_active       INTEGER DEFAULT 1,
        locked_until    TEXT,
        face_fail_total INTEGER DEFAULT 0,
        otp_fail_total  INTEGER DEFAULT 0,
        created_at      TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS otp_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL REFERENCES bank_user(id) ON DELETE CASCADE,
        otp_code   TEXT    NOT NULL,
        created_at TEXT    DEFAULT (datetime('now')),
        is_used    INTEGER DEFAULT 0,
        attempts   INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS balance (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE REFERENCES bank_user(id) ON DELETE CASCADE,
        amount  REAL    DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS txn_history (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER NOT NULL REFERENCES bank_user(id) ON DELETE CASCADE,
        tx_type       TEXT    NOT NULL,
        amount        REAL    NOT NULL,
        balance_after REAL    NOT NULL,
        created_at    TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS auth_fail_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL REFERENCES bank_user(id) ON DELETE CASCADE,
        fail_type  TEXT    NOT NULL,
        message    TEXT    DEFAULT '',
        created_at TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS admin_user (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT    NOT NULL UNIQUE,
        password TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS deposit_request (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL REFERENCES bank_user(id) ON DELETE CASCADE,
        amount     REAL    NOT NULL,
        status     TEXT    DEFAULT 'PENDING',
        created_at TEXT    DEFAULT (datetime('now')),
        reviewed_at TEXT
    );
    """)
    # default admin  (username: admin  |  password: 123)
    pw = hashlib.sha256("123".encode()).hexdigest()
    db.execute("INSERT OR IGNORE INTO admin_user(username,password) VALUES(?,?)", ("admin", pw))
    db.commit()
    db.close()
    print("[DB] Initialized.")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def gen_account():
    while True:
        acc = "".join(str(random.randint(0, 9)) for _ in range(12))
        if not query_db("SELECT id FROM bank_user WHERE account_number=?", (acc,), one=True):
            return acc

def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()

def check_pin(pin, hashed):
    return hashlib.sha256(pin.encode()).hexdigest() == hashed

def gen_otp():
    return "".join(random.choices(string.digits, k=6))

def is_locked(user):
    if not user["is_active"]:
        if user["locked_until"]:
            if datetime.now() < datetime.fromisoformat(user["locked_until"]):
                return True
        return True
    return False

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# FACE RECOGNITION ENGINE (Strong — DeepFace / dlib)
# ─────────────────────────────────────────────
# We use face_recognition (128-d dlib embeddings) for highest accuracy
try:
    import face_recognition
    FACE_LIB = "face_recognition"
except ImportError:
    FACE_LIB = "opencv"

print(f"[FACE] Using library: {FACE_LIB}")

def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face_encoding(img_bgr):
    """Extract 128-d face embedding using face_recognition (dlib). Returns numpy array or None."""
    if FACE_LIB == "face_recognition":
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            return None, "No face detected"
        if len(locs) > 1:
            return None, "Multiple faces detected — please be alone"
        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            return None, "Could not encode face"
        return encs[0], None
    else:
        # Fallback: histogram-based (less accurate)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade = get_face_cascade()
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None, "No face detected"
        x, y, w, h = faces[0]
        face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        hist = cv2.calcHist([face], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        return hist, None

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return {}  # {face_id: [enc1, enc2, ...]}

def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def train_model():
    model = {}
    for item in os.listdir(FACE_DATA_DIR):
        face_dir = os.path.join(FACE_DATA_DIR, item)
        if os.path.isdir(face_dir):
            try:
                fid = int(item)
            except ValueError:
                continue
            encodings = []
            for fname in os.listdir(face_dir):
                if fname.endswith((".jpg", ".png")):
                    img = cv2.imread(os.path.join(face_dir, fname))
                    if img is None:
                        continue
                    enc, err = extract_face_encoding(img)
                    if enc is not None:
                        encodings.append(enc)
            if encodings:
                model[fid] = encodings
    save_model(model)
    total = sum(len(v) for v in model.values())
    print(f"[MODEL] Trained: {len(model)} users, {total} samples")
    return len(model), total

def recognize_face(img_bgr):
    """
    Returns (face_id, confidence, error_msg).
    confidence: 0–100 (higher=better match).
    """
    model = load_model()
    if not model:
        return None, 0, "Model not trained yet"

    enc, err = extract_face_encoding(img_bgr)
    if enc is None:
        return None, 0, err

    if FACE_LIB == "face_recognition":
        best_fid, best_conf = None, 0
        THRESHOLD = 0.50  # L2 distance threshold (lower=stricter)
        for fid, stored_encs in model.items():
            known = np.array(stored_encs)
            dists = face_recognition.face_distance(known, enc)
            min_d = float(np.min(dists))
            # Convert distance to confidence (0-100)
            conf  = max(0, (1 - min_d / 0.6)) * 100
            if min_d < THRESHOLD and conf > best_conf:
                best_conf = conf
                best_fid  = fid
        if best_fid is not None:
            return best_fid, round(best_conf, 1), None
        return None, 0, "Face not recognized"
    else:
        # OpenCV fallback: cosine similarity
        best_fid, best_sim = None, -1
        for fid, stored_encs in model.items():
            for s_enc in stored_encs:
                sim = float(np.dot(enc, s_enc) / (np.linalg.norm(enc) * np.linalg.norm(s_enc) + 1e-7))
                if sim > best_sim:
                    best_sim = sim
                    best_fid  = fid
        if best_sim > 0.92:
            return best_fid, round(best_sim * 100, 1), None
        return None, 0, "Face not recognized"

# ─────────────────────────────────────────────
# EMAIL OTP SENDER
# ─────────────────────────────────────────────
def send_otp_email(user, otp):
    """Send OTP to user's registered email via Gmail SMTP."""
    try:
        msg = Message(
            subject="🔐 SecureATM — Your OTP Code",
            recipients=[user["email"]],
            html=f"""
            <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;
                        background:#0a0f1e;color:#e0e6f0;border-radius:12px;padding:30px;">
              <h2 style="color:#00e5ff;margin-bottom:4px;">&#127970; SecureATM</h2>
              <p style="color:#8fa3c0;font-size:13px;margin-top:0;">Secure ATM Transaction System</p>
              <hr style="border-color:#1e2d45;margin:20px 0;"/>
              <p>Hello <strong>{user['name']}</strong>,</p>
              <p>Your One-Time Password (OTP) for ATM verification is:</p>
              <div style="background:#111827;border:2px solid #00e5ff;border-radius:10px;
                          text-align:center;padding:20px;margin:20px 0;">
                <span style="font-size:2.5rem;font-weight:700;letter-spacing:12px;
                             color:#00e5ff;">{otp}</span>
              </div>
              <p style="color:#ffa502;font-size:13px;">
                &#9200; This OTP is valid for <strong>5 minutes</strong> only.
              </p>
              <p style="color:#8fa3c0;font-size:12px;">
                If you did not attempt this transaction, please contact your bank immediately.
              </p>
              <hr style="border-color:#1e2d45;margin:20px 0;"/>
              <p style="color:#4a5568;font-size:11px;text-align:center;">
                SecureATM &mdash; Dual-Layer Authentication System<br/>
                Do NOT share this OTP with anyone.
              </p>
            </div>
            """
        )
        mail.send(msg)
        print(f"[MAIL] OTP sent to {user['email']}")
        return True
    except Exception as e:
        print(f"[MAIL ERROR] {e}")
        return False

# ─────────────────────────────────────────────
# CAMERA STREAM
# ─────────────────────────────────────────────
camera_ref = {"cap": None}

def get_camera():
    if camera_ref["cap"] is None or not camera_ref["cap"].isOpened():
        camera_ref["cap"] = cv2.VideoCapture(0)
    return camera_ref["cap"]

def release_camera():
    if camera_ref["cap"] and camera_ref["cap"].isOpened():
        camera_ref["cap"].release()
        camera_ref["cap"] = None

def gen_frames():
    cap = get_camera()
    cascade = get_face_cascade()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
            cv2.putText(frame, "Face Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ─────────────────────────────────────────────
# ─── USER ROUTES ─────────────────────────────
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

# --- Face Verify Page ---
@app.route("/face-verify/")
def face_verify_page():
    session.pop("face_verified_user_id", None)
    return render_template("face_verify.html")

@app.route("/face-verify-feed/")
def face_verify_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/face-verify-check/", methods=["POST"])
def face_verify_check():
    data     = request.get_json(silent=True) or {}
    img_b64  = data.get("image", "")
    if not img_b64:
        return jsonify(status="error", message="No image received")

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[-1])
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify(status="error", message="Invalid image data")

    face_id, confidence, err = recognize_face(frame)
    if face_id is None:
        return jsonify(status="fail", message=err or "Face not recognized", confidence=0)

    user = query_db("SELECT * FROM bank_user WHERE face_id=?", (face_id,), one=True)
    if not user:
        return jsonify(status="fail", message="No account linked to this face", confidence=0)

    if is_locked(user):
        lu = user["locked_until"]
        msg = f"Account locked until {lu}" if lu else "Account is permanently locked. Contact bank."
        return jsonify(status="locked", message=msg, confidence=0)

    # Reset face fail on success
    exec_db("UPDATE bank_user SET face_fail_total=0 WHERE id=?", (user["id"],))
    session["face_verified_user_id"] = user["id"]
    return jsonify(status="success", message=f"Face matched! Confidence: {confidence}%",
                   confidence=confidence, name=user["name"])

@app.route("/face-verify-done/")
def face_verify_done():
    uid = session.get("face_verified_user_id")
    if not uid:
        flash("Please complete face verification first.", "danger")
        return redirect(url_for("face_verify_page"))
    # Generate OTP — store created_at as UTC explicitly so expiry check is always consistent
    otp = gen_otp()
    from datetime import timezone
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    exec_db("UPDATE otp_log SET is_used=1 WHERE user_id=? AND is_used=0", (uid,))
    exec_db("INSERT INTO otp_log(user_id, otp_code, created_at) VALUES(?,?,?)", (uid, otp, now_utc))
    user = query_db("SELECT * FROM bank_user WHERE id=?", (uid,), one=True)
    session["otp_user_id"] = uid
    # Send OTP via Gmail
    email_sent = send_otp_email(user, otp)
    masked_email = user["email"][:3] + "****" + user["email"][user["email"].index("@"):]
    if email_sent:
        flash(f"OTP sent to {masked_email} — check your inbox. Valid for 5 minutes.", "success")
    else:
        # Fallback: show OTP on screen so app still works even if email fails
        flash(f"Email delivery failed. [FALLBACK] OTP: {otp}  (Check server logs)", "warning")
    return redirect(url_for("otp_page"))

@app.route("/face-verify-cancel/")
def face_verify_cancel():
    session.pop("face_verified_user_id", None)
    return redirect(url_for("index"))

@app.route("/face-verify-stop/", methods=["POST"])
def face_verify_stop():
    release_camera()
    return jsonify(status="ok")

# --- OTP ---
@app.route("/otp/", methods=["GET", "POST"])
def otp_page():
    uid = session.get("otp_user_id") or session.get("face_verified_user_id")
    if not uid:
        return redirect(url_for("index"))

    if request.method == "POST":
        entered = request.form.get("otp", "").strip()
        user    = query_db("SELECT * FROM bank_user WHERE id=?", (uid,), one=True)

        if is_locked(user):
            flash("Account is locked. Contact your bank.", "danger")
            session.clear()
            return redirect(url_for("index"))

        log = query_db(
            "SELECT * FROM otp_log WHERE user_id=? AND is_used=0 ORDER BY id DESC LIMIT 1",
            (uid,), one=True
        )
        if not log:
            flash("No active OTP found. Please restart authentication.", "danger")
            session.clear()
            return redirect(url_for("index"))

        # Compare in UTC — SQLite datetime('now') stores UTC
        from datetime import timezone
        created_utc  = datetime.fromisoformat(log["created_at"]).replace(tzinfo=timezone.utc)
        expire_at    = created_utc + timedelta(minutes=OTP_EXPIRE_MIN)
        now_utc      = datetime.now(timezone.utc)
        if now_utc > expire_at:
            exec_db("UPDATE otp_log SET is_used=1 WHERE id=?", (log["id"],))
            flash("OTP has expired. Please authenticate again.", "warning")
            session.clear()
            return redirect(url_for("index"))

        if entered != log["otp_code"]:
            attempts = log["attempts"] + 1
            exec_db("UPDATE otp_log SET attempts=? WHERE id=?", (attempts, log["id"]))
            exec_db("UPDATE bank_user SET otp_fail_total=otp_fail_total+1 WHERE id=?", (uid,))
            exec_db("INSERT INTO auth_fail_log(user_id,fail_type,message) VALUES(?,?,?)",
                    (uid, "OTP", f"Wrong OTP attempt {attempts}"))
            remaining = MAX_OTP_FAILS - attempts
            if remaining <= 0:
                lu = (datetime.now() + timedelta(minutes=LOCK_MINUTES)).isoformat()
                exec_db("UPDATE bank_user SET is_active=0, locked_until=? WHERE id=?", (lu, uid))
                flash("Too many wrong OTPs. Account locked for 10 minutes.", "danger")
                session.clear()
                return redirect(url_for("index"))
            flash(f"Wrong OTP. {remaining} attempt(s) remaining.", "danger")
            return render_template("otp.html")

        # OTP correct
        exec_db("UPDATE otp_log SET is_used=1 WHERE id=?", (log["id"],))
        exec_db("UPDATE bank_user SET otp_fail_total=0 WHERE id=?", (uid,))
        session["user_id"]   = uid
        session["user_name"] = user["name"]
        session.pop("otp_user_id", None)
        session.pop("face_verified_user_id", None)
        return redirect(url_for("account_dashboard"))

    return render_template("otp.html")

# --- Resend OTP ---
@app.route("/resend-otp/", methods=["POST"])
def resend_otp():
    uid = session.get("otp_user_id") or session.get("face_verified_user_id")
    if not uid:
        return redirect(url_for("index"))
    otp = gen_otp()
    from datetime import timezone
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    exec_db("UPDATE otp_log SET is_used=1 WHERE user_id=? AND is_used=0", (uid,))
    exec_db("INSERT INTO otp_log(user_id, otp_code, created_at) VALUES(?,?,?)", (uid, otp, now_utc))
    user = query_db("SELECT * FROM bank_user WHERE id=?", (uid,), one=True)
    email_sent = send_otp_email(user, otp)
    masked_email = user["email"][:3] + "****" + user["email"][user["email"].index("@"):]
    if email_sent:
        flash(f"New OTP sent to {masked_email}. Valid for 5 minutes.", "success")
    else:
        flash(f"Email delivery failed. [FALLBACK] OTP: {otp}", "warning")
    return redirect(url_for("otp_page"))

# --- Dashboard ---
@app.route("/account/")
@login_required
def account_dashboard():
    uid  = session["user_id"]
    user = query_db("SELECT * FROM bank_user WHERE id=?", (uid,), one=True)
    bal  = query_db("SELECT amount FROM balance WHERE user_id=?", (uid,), one=True)
    txns = query_db("SELECT * FROM txn_history WHERE user_id=? ORDER BY id DESC LIMIT 10", (uid,))
    balance = bal["amount"] if bal else 0
    return render_template("dashboard.html", user=user, balance=balance, txns=txns)

# --- Deposit Request (user submits, admin approves) ---
@app.route("/deposit/", methods=["GET", "POST"])
@login_required
def deposit():
    uid = session["user_id"]
    if request.method == "POST":
        try:
            amt = float(request.form["amount"])
            if amt <= 0:
                raise ValueError
        except (ValueError, KeyError):
            flash("Invalid amount.", "danger")
            return render_template("deposit.html")

        exec_db("INSERT INTO deposit_request(user_id, amount) VALUES(?,?)", (uid, amt))
        flash(f"Deposit request of ₹{amt:,.2f} submitted successfully. Waiting for admin approval.", "success")
        return redirect(url_for("account_dashboard"))

    # Show pending requests
    pending = query_db(
        "SELECT * FROM deposit_request WHERE user_id=? ORDER BY id DESC LIMIT 10", (uid,)
    )
    return render_template("deposit.html", pending=pending)


# --- Withdraw (PIN verification required) ---
@app.route("/withdraw/", methods=["GET", "POST"])
@login_required
def withdraw():
    uid = session["user_id"]
    if request.method == "POST":
        pin    = request.form.get("pin", "").strip()
        try:
            amt = float(request.form["amount"])
            if amt <= 0:
                raise ValueError
        except (ValueError, KeyError):
            flash("Invalid amount.", "danger")
            return render_template("withdraw.html")

        # PIN verification
        user = query_db("SELECT * FROM bank_user WHERE id=?", (uid,), one=True)
        if not check_pin(pin, user["pin_hash"]):
            flash("Incorrect PIN. Withdrawal denied.", "danger")
            return render_template("withdraw.html")

        bal = query_db("SELECT amount FROM balance WHERE user_id=?", (uid,), one=True)
        current = bal["amount"] if bal else 0
        if amt > current:
            flash("Insufficient balance.", "danger")
            return render_template("withdraw.html")

        new_bal = current - amt
        exec_db("UPDATE balance SET amount=? WHERE user_id=?", (new_bal, uid))
        exec_db("INSERT INTO txn_history(user_id,tx_type,amount,balance_after) VALUES(?,?,?,?)",
                (uid, "WITHDRAW", amt, new_bal))
        flash(f"₹{amt:,.2f} withdrawn successfully.", "success")
        return redirect(url_for("account_dashboard"))
    return render_template("withdraw.html")

# --- Logout ---
@app.route("/logout/")
def user_logout():
    session.clear()
    release_camera()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))

# ─────────────────────────────────────────────
# ─── ADMIN ROUTES ────────────────────────────
# ─────────────────────────────────────────────

@app.route("/admin-login/", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_logged_in"):
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        uname = request.form.get("username", "")
        pw    = request.form.get("password", "")
        adm   = query_db("SELECT * FROM admin_user WHERE username=?", (uname,), one=True)
        if adm and adm["password"] == hashlib.sha256(pw.encode()).hexdigest():
            session["admin_logged_in"] = True
            session["admin_username"]  = uname
            return redirect(url_for("admin_dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("admin_login.html")

@app.route("/admin-logout/")
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_username", None)
    return redirect(url_for("admin_login"))

@app.route("/admin-dashboard/")
@admin_required
def admin_dashboard():
    users = query_db("""
        SELECT u.*, b.amount as balance,
               (SELECT COUNT(*) FROM auth_fail_log WHERE user_id=u.id) as fail_count
        FROM bank_user u
        LEFT JOIN balance b ON b.user_id=u.id
        ORDER BY u.id DESC
    """)
    total_users = len(users)
    total_bal   = sum(u["balance"] or 0 for u in users)
    model_info  = {}
    if os.path.exists(MODEL_PATH):
        m = load_model()
        model_info = {"users": len(m), "samples": sum(len(v) for v in m.values())}
    pending_deposits = query_db("SELECT COUNT(*) as cnt FROM deposit_request WHERE status='PENDING'", one=True)
    pending_count = pending_deposits["cnt"] if pending_deposits else 0
    return render_template("admin_dashboard.html", users=users,
                           total_users=total_users, total_bal=total_bal,
                           model_info=model_info, pending_count=pending_count)

@app.route("/admin-add-user/", methods=["GET", "POST"])
@admin_required
def admin_add_user():
    if request.method == "POST":
        name    = request.form.get("name","").strip()
        dob     = request.form.get("dob","").strip()
        phone   = request.form.get("phone","").strip()
        email   = request.form.get("email","").strip()
        address = request.form.get("address","").strip()
        stype   = request.form.get("saving_type","SAVINGS")
        pin     = request.form.get("pin","").strip()

        if not all([name, dob, phone, email, pin]):
            flash("All fields required.", "danger")
            return render_template("admin_add_user.html")
        if len(pin) != 4 or not pin.isdigit():
            flash("PIN must be exactly 4 digits.", "danger")
            return render_template("admin_add_user.html")
        if query_db("SELECT id FROM bank_user WHERE name=?", (name,), one=True):
            flash("User already exists.", "danger")
            return render_template("admin_add_user.html")

        acc = gen_account()
        ph  = hash_pin(pin)
        # Assign next face_id
        last = query_db("SELECT MAX(face_id) as mx FROM bank_user", one=True)
        fid  = (last["mx"] or 0) + 1

        uid = exec_db("""INSERT INTO bank_user
            (name,dob,account_number,phone,email,address,saving_type,pin_hash,face_id)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (name, dob, acc, phone, email, address, stype, ph, fid))
        exec_db("INSERT INTO balance(user_id,amount) VALUES(?,0)", (uid,))
        session["new_user_id"]   = uid
        session["new_user_name"] = name
        flash(f"User '{name}' created. Account: {acc}. Now capture face.", "success")
        return redirect(url_for("admin_capture_face", user_id=uid))
    return render_template("admin_add_user.html")

@app.route("/admin-capture-face/<int:user_id>/")
@admin_required
def admin_capture_face(user_id):
    user = query_db("SELECT * FROM bank_user WHERE id=?", (user_id,), one=True)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin_dashboard"))
    face_dir = os.path.join(FACE_DATA_DIR, str(user["face_id"]))
    os.makedirs(face_dir, exist_ok=True)
    existing = len([f for f in os.listdir(face_dir) if f.endswith(".jpg")])
    return render_template("admin_capture_face.html", user=user, existing=existing)

@app.route("/admin-capture-face-save/", methods=["POST"])
@admin_required
def admin_capture_face_save():
    data    = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    imgs    = data.get("images", [])
    user    = query_db("SELECT * FROM bank_user WHERE id=?", (user_id,), one=True)
    if not user or not imgs:
        return jsonify(status="error", message="Missing data")

    face_dir = os.path.join(FACE_DATA_DIR, str(user["face_id"]))
    os.makedirs(face_dir, exist_ok=True)

    saved, errors = 0, []
    for i, b64 in enumerate(imgs):
        try:
            img_bytes = base64.b64decode(b64.split(",")[-1])
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            enc, err  = extract_face_encoding(frame)
            if enc is None:
                errors.append(f"Image {i+1}: {err}")
                continue
            # Save image
            existing = len(os.listdir(face_dir))
            fname    = os.path.join(face_dir, f"{existing+1}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        except Exception as e:
            errors.append(str(e))

    return jsonify(status="ok", saved=saved, errors=errors)

@app.route("/admin-capture-feed/")
@admin_required
def admin_capture_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/admin-train-model/", methods=["POST"])
@admin_required
def admin_train_model():
    users_count, samples = train_model()
    flash(f"Model trained: {users_count} users, {samples} face samples.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin-edit-user/<int:user_id>/", methods=["GET", "POST"])
@admin_required
def admin_edit_user(user_id):
    user = query_db("SELECT * FROM bank_user WHERE id=?", (user_id,), one=True)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        phone   = request.form.get("phone","").strip()
        email   = request.form.get("email","").strip()
        address = request.form.get("address","").strip()
        stype   = request.form.get("saving_type","SAVINGS")
        new_pin = request.form.get("new_pin","").strip()
        if new_pin:
            if not (new_pin.isdigit() and len(new_pin) == 4):
                flash("PIN must be 4 digits.", "danger")
                return render_template("admin_edit_user.html", user=user)
            exec_db("UPDATE bank_user SET pin_hash=? WHERE id=?", (hash_pin(new_pin), user_id))
        exec_db("UPDATE bank_user SET phone=?,email=?,address=?,saving_type=? WHERE id=?",
                (phone, email, address, stype, user_id))
        flash("User updated.", "success")
        return redirect(url_for("admin_user_details", user_id=user_id))
    return render_template("admin_edit_user.html", user=user)

@app.route("/admin-delete-user/<int:user_id>/", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    user = query_db("SELECT * FROM bank_user WHERE id=?", (user_id,), one=True)
    if user:
        face_dir = os.path.join(FACE_DATA_DIR, str(user["face_id"]))
        import shutil
        if os.path.exists(face_dir):
            shutil.rmtree(face_dir)
        exec_db("DELETE FROM bank_user WHERE id=?", (user_id,))
        flash("User deleted.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin-user/<int:user_id>/")
@admin_required
def admin_user_details(user_id):
    user  = query_db("SELECT * FROM bank_user WHERE id=?", (user_id,), one=True)
    bal   = query_db("SELECT amount FROM balance WHERE user_id=?", (user_id,), one=True)
    txns  = query_db("SELECT * FROM txn_history WHERE user_id=? ORDER BY id DESC", (user_id,))
    fails = query_db("SELECT * FROM auth_fail_log WHERE user_id=? ORDER BY id DESC LIMIT 20", (user_id,))
    face_dir   = os.path.join(FACE_DATA_DIR, str(user["face_id"])) if user else ""
    face_count = len([f for f in os.listdir(face_dir) if f.endswith(".jpg")]) if os.path.exists(face_dir) else 0
    return render_template("admin_user_details.html", user=user,
                           balance=bal["amount"] if bal else 0,
                           txns=txns, fails=fails, face_count=face_count)

@app.route("/admin-user/<int:user_id>/toggle/", methods=["POST"])
@admin_required
def admin_toggle_user(user_id):
    user = query_db("SELECT * FROM bank_user WHERE id=?", (user_id,), one=True)
    if user:
        new_active = 0 if user["is_active"] else 1
        lu = None if new_active else None
        exec_db("UPDATE bank_user SET is_active=?, locked_until=? WHERE id=?", (new_active, lu, user_id))
        flash("User status updated.", "success")
    return redirect(url_for("admin_user_details", user_id=user_id))

@app.route("/admin-user/<int:user_id>/clear-fails/", methods=["POST"])
@admin_required
def admin_clear_fails(user_id):
    exec_db("UPDATE bank_user SET face_fail_total=0, otp_fail_total=0, is_active=1, locked_until=NULL WHERE id=?", (user_id,))
    exec_db("DELETE FROM auth_fail_log WHERE user_id=?", (user_id,))
    flash("All fail logs cleared.", "success")
    return redirect(url_for("admin_user_details", user_id=user_id))

@app.route("/admin-deposit-requests/")
@admin_required
def admin_deposit_requests():
    requests_list = query_db("""
        SELECT dr.*, u.name, u.account_number
        FROM deposit_request dr
        JOIN bank_user u ON u.id = dr.user_id
        ORDER BY dr.status ASC, dr.id DESC
    """)
    pending_count = sum(1 for r in requests_list if r["status"] == "PENDING")
    return render_template("admin_deposit_requests.html",
                           requests_list=requests_list, pending_count=pending_count)

@app.route("/admin-deposit-approve/<int:req_id>/", methods=["POST"])
@admin_required
def admin_deposit_approve(req_id):
    req = query_db("SELECT * FROM deposit_request WHERE id=?", (req_id,), one=True)
    if not req or req["status"] != "PENDING":
        flash("Request not found or already processed.", "danger")
        return redirect(url_for("admin_deposit_requests"))

    uid = req["user_id"]
    amt = req["amount"]
    bal = query_db("SELECT amount FROM balance WHERE user_id=?", (uid,), one=True)
    new_bal = (bal["amount"] if bal else 0) + amt
    if bal:
        exec_db("UPDATE balance SET amount=? WHERE user_id=?", (new_bal, uid))
    else:
        exec_db("INSERT INTO balance(user_id,amount) VALUES(?,?)", (uid, new_bal))
    exec_db("INSERT INTO txn_history(user_id,tx_type,amount,balance_after) VALUES(?,?,?,?)",
            (uid, "DEPOSIT", amt, new_bal))
    from datetime import timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    exec_db("UPDATE deposit_request SET status='APPROVED', reviewed_at=? WHERE id=?", (now, req_id))
    flash(f"Deposit of ₹{amt:,.2f} approved and credited.", "success")
    return redirect(url_for("admin_deposit_requests"))

@app.route("/admin-deposit-reject/<int:req_id>/", methods=["POST"])
@admin_required
def admin_deposit_reject(req_id):
    req = query_db("SELECT * FROM deposit_request WHERE id=?", (req_id,), one=True)
    if not req or req["status"] != "PENDING":
        flash("Request not found or already processed.", "danger")
        return redirect(url_for("admin_deposit_requests"))
    from datetime import timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    exec_db("UPDATE deposit_request SET status='REJECTED', reviewed_at=? WHERE id=?", (now, req_id))
    flash("Deposit request rejected.", "warning")
    return redirect(url_for("admin_deposit_requests"))

# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True, port=5000)
