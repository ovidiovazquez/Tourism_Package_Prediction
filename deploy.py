
import base64
import os
import sys
from pathlib import Path

import requests


# ============================================================
# GITHUB CONFIGURATION
# ============================================================

REPO_OWNER = "ovidiovazquez"
REPO_NAME = "Tourism_Package_Prediction"
REPO_BRANCH = "main"

BASE_DIR = Path(__file__).resolve().parent

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN environment variable is not configured."
    )

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}


# ============================================================
# FILE UPLOAD FUNCTION
# ============================================================

def upload_file(local_path, github_path):

    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(local_path)

    url = (
        f"https://api.github.com/repos/"
        f"{REPO_OWNER}/{REPO_NAME}/contents/{github_path}"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        params={"ref": REPO_BRANCH},
        timeout=30
    )

    payload = {
        "message": f"Deploy {github_path}",
        "content": base64.b64encode(
            local_path.read_bytes()
        ).decode("utf-8"),
        "branch": REPO_BRANCH
    }

    if response.status_code == 200:
        payload["sha"] = response.json()["sha"]

    elif response.status_code != 404:
        response.raise_for_status()

    upload_response = requests.put(
        url,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    upload_response.raise_for_status()

    print(f"SUCCESS: {github_path}")


# ============================================================
# DEPLOYMENT FILES
# ============================================================

deployment_files = {
    "app.py": BASE_DIR / "app.py",
    "requirements.txt": BASE_DIR / "requirements.txt",
    "Dockerfile": BASE_DIR / "Dockerfile",

    "models/tourism_random_forest_pipeline.joblib":
        BASE_DIR / "models" /
        "tourism_random_forest_pipeline.joblib",

    "models/model_schema.json":
        BASE_DIR / "models" /
        "model_schema.json"
}


# ============================================================
# EXECUTE DEPLOYMENT
# ============================================================

if __name__ == "__main__":

    print("Starting deployment to GitHub...")

    for github_path, local_path in deployment_files.items():
        upload_file(local_path, github_path)

    print("\nAll deployment files uploaded successfully.")
