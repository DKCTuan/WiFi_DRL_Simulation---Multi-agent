import os
import sys
import zipfile
from datetime import datetime


def build_zip_name() -> str:
    label = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        return f"WiFi_DRL_{safe_label}_{timestamp}.zip"
    return f"WiFi_DRL_{timestamp}.zip"


zip_name = build_zip_name()
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk("."):
        # Source-only Kaggle archive: results/checkpoints are experiment outputs,
        # not reproducible code, and can make uploads unnecessarily large.
        dirs[:] = [d for d in dirs if d not in [
            "results", "results_data", "__pycache__", ".git",
        ]]

        for file in files:
            if file == "create_zip.py" or file.endswith(".zip"):
                continue

            filepath = os.path.join(root, file)
            arcname = filepath.replace(os.sep, "/")
            if arcname.startswith("./"):
                arcname = arcname[2:]
            if "tempCodeRunnerFile" not in arcname:
                zipf.write(filepath, arcname=arcname)

print(f"Created {zip_name} with forward slashes (Kaggle compatible)")
