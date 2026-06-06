# PhishGuard v2 — PythonAnywhere Deployment Guide

## Prerequisites
- A PythonAnywhere account (free tier works for demo)
- Python 3.10 or 3.11

---

## Step 1 — Upload the project

**Option A: Upload zip via the Files tab**
1. Zip the `phishing-detector-v2/` folder on your machine.
2. Log in to PythonAnywhere → Files tab → Upload.
3. Open a Bash console and run:
   ```bash
   cd ~
   unzip phishing-detector-v2.zip
   ```

**Option B: Clone from GitHub**
```bash
git clone https://github.com/youruser/phishing-detector-v2.git
```

---

## Step 2 — Install dependencies

Open a **Bash** console on PythonAnywhere:

```bash
cd ~/phishing-detector-v2
pip3 install --user -r requirements.txt
```

> PythonAnywhere free accounts use a shared Python environment.  
> The `--user` flag installs packages to your home directory.

---

## Step 3 — Train the model

```bash
cd ~/phishing-detector-v2
python3 train_model.py
```

This creates `models/model.pkl`, `models/vectorizer.pkl`, `models/scaler.pkl`,  
and `models/model_info.json`. Expected runtime: ~30–60 seconds.

---

## Step 4 — Configure the Web app

1. Go to the **Web** tab → click **Add a new web app**.
2. Choose **Manual configuration** (not Flask wizard).
3. Choose **Python 3.10**.

### WSGI configuration
- Click the WSGI configuration file link (e.g. `/var/www/yourusername_pythonanywhere_com_wsgi.py`).
- Replace its entire content with the contents of `wsgi.py` from this project.
- Update the two lines at the top:
  ```python
  PROJECT_HOME = "/home/YourUsername/phishing-detector-v2"
  PYTHON_VERSION = "python3.10"
  ```

### Source code & working directory
| Setting | Value |
|---------|-------|
| Source code | `/home/YourUsername/phishing-detector-v2` |
| Working directory | `/home/YourUsername/phishing-detector-v2` |

### Static files
| URL | Directory |
|-----|-----------|
| `/static/` | `/home/YourUsername/phishing-detector-v2/static` |

---

## Step 5 — Set the SECRET_KEY environment variable

In the **Web** tab → **Environment variables** section, add:

| Key | Value |
|-----|-------|
| `SECRET_KEY` | `<a long random string — generate with: python3 -c "import secrets; print(secrets.token_hex(32))"` |

---

## Step 6 — Reload & test

Click **Reload** on the Web tab.  
Visit `https://yourusername.pythonanywhere.com` — you should see PhishGuard.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Re-run `pip3 install --user -r requirements.txt` in a Bash console |
| `FileNotFoundError: models/model.pkl` | Re-run `python3 train_model.py` |
| Static CSS not loading | Check the static files mapping in the Web tab |
| 500 errors | Check the error log in the Web tab → Log files |

---

## Keeping the app awake (free tier)

Free PythonAnywhere accounts sleep after inactivity.  
Use a free cron service (e.g. cron-job.org) to ping `/health` every 20 minutes:
```
https://yourusername.pythonanywhere.com/health
```

---

## Environment variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Recommended | Flask session signing key |
| `FLASK_DEBUG` | No | Set to `1` only during development |
