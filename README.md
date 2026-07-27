# SecureATM – Flask Application
## Additional Security in ATM Transactions Using Face Recognition + OTP Verification

---

## Project Structure
```
flask_atm/
├── app.py                  # Main Flask application (all routes + logic)
├── requirements.txt
├── instance/
│   └── atm.db              # SQLite3 database (auto-created)
├── face_data/
│   ├── face_model.pkl      # Trained model (auto-generated)
│   └── <face_id>/          # Face images per user
└── templates/
    ├── base.html
    ├── index.html
    ├── face_verify.html
    ├── otp.html
    ├── dashboard.html
    ├── deposit.html
    ├── withdraw.html
    ├── admin_login.html
    ├── admin_dashboard.html
    ├── admin_add_user.html
    ├── admin_capture_face.html
    ├── admin_user_details.html
    └── admin_edit_user.html
```

---

## Installation

### Step 1 — Install system dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y python3-pip cmake libopenblas-dev liblapack-dev libx11-dev
```

### Step 2 — Install Python packages
```bash
pip install -r requirements.txt
```

> **Note:** `face_recognition` uses `dlib` (128-d embeddings) for strong face verification.
> If dlib build fails, install prebuilt wheel:
> ```bash
> pip install dlib --find-links https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases
> pip install face_recognition
> ```

### Step 3 — Run the app
```bash
python app.py
```
Open: http://localhost:5000

---

## Usage

### Admin Panel
- URL: http://localhost:5000/admin-login/
- Default credentials: **admin / admin123**

#### Workflow to add a user:
1. Admin Login → Admin Dashboard → Add User
2. Fill in details (name, DOB, phone, email, 4-digit PIN)
3. Capture face (10 photos in different angles)
4. Model trains automatically after saving
5. User can now authenticate at ATM

### User Authentication (ATM Flow)
1. Visit http://localhost:5000/
2. Click "Start Authentication"
3. **Step 1**: Face is scanned by webcam — matched against trained model (dlib 128-d embedding)
4. **Step 2**: OTP sent to registered phone → enter 6-digit OTP
5. Access granted → Dashboard (Balance, Deposit, Withdraw, History)

---

## Security Features

| Feature | Description |
|---|---|
| **Face Recognition** | dlib 128-d face embeddings (L2 distance < 0.50 threshold) |
| **OTP Verification** | 6-digit OTP, 5-min expiry, 3 attempt limit |
| **Account Locking** | 10-min lockout after 3 failed OTP attempts |
| **PIN Hashing** | SHA-256 hashed storage |
| **Fail Logging** | All auth failures recorded in `auth_fail_log` table |
| **Multi-face detection** | Rejects if more than one face in frame |

---

## Database Schema (SQLite3)

- `bank_user` — User profiles, PIN hash, face_id, lock status
- `otp_log` — OTP records with expiry tracking
- `balance` — Account balances
- `transaction` — Deposit/Withdraw history
- `auth_fail_log` — Face & OTP failure logs
- `admin_user` — Admin credentials

---

## Face Recognition Details

Uses **face_recognition** library (dlib HOG + 128-d ResNet embeddings):
- Threshold: L2 distance < **0.50** (strict — rejects strangers)
- Confidence: `(1 - distance/0.6) × 100`
- Falls back to OpenCV Haar + histogram if face_recognition not installed
- 10 training photos per user recommended for best accuracy
