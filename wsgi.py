"""
wsgi.py
-------
WSGI entry point for PythonAnywhere deployment.

PythonAnywhere looks for a variable called `application` in this file.
Set the paths below to match your actual PythonAnywhere username and project folder.

Setup steps on PythonAnywhere:
  1. Upload the project to /home/<username>/phishing-detector-v2/
  2. In the Web tab → WSGI configuration file, paste or point to this file.
  3. Set the working directory to /home/<username>/phishing-detector-v2/
  4. Set SECRET_KEY in the Web tab → Environment Variables section.
  5. Run `python train_model.py` from a Bash console before starting the app.
"""

import sys
import os

# ── Adjust these two lines to match your PythonAnywhere setup ──
PROJECT_HOME = "/home/YourUsername/phishing-detector-v2"
PYTHON_VERSION = "python3.10"  # match your PythonAnywhere python version
# ───────────────────────────────────────────────────────────────

if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)

os.chdir(PROJECT_HOME)

# Optional: load a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_HOME, ".env"))
except ImportError:
    pass

from app import app as application  # noqa: E402 — must come after sys.path setup
