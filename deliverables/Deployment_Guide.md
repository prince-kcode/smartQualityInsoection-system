# IndustrialGuard AI Local Deployment Guide

## Recommended mini-project deployment

Run the app on the development laptop. This keeps the trained checkpoints and dataset local and makes the NVIDIA RTX 4060 available for inference. The app currently has no authentication, so bind it to localhost and do not expose its Streamlit port to a public network.

## Start locally

Open PowerShell in `D:\ai project` and run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\streamlit.exe run app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. Select a category matching the object shown in the image. Keep the SQLite file and generated artifacts backed up if the inspection history matters.

## Files to keep private

- `data/raw/`: MVTec AD images and masks; follow the dataset license.
- `results/`: trained checkpoints and evaluation outputs.
- `artifacts/`: uploaded images, heatmaps, overlays, local database, and reports.
- `.env`: future secrets, if any.

The project `.gitignore` excludes these locations and local secrets. Do not put checkpoints or dataset files in a public GitHub repository. Keep `requirements-lock.txt` with the project for environment reproducibility.

## Hosting options

The recommended demonstration remains local. Hosted free tiers can change their resource limits and commonly do not provide a dedicated NVIDIA GPU. The two PatchCore checkpoints are roughly 218 MB and 224 MB, so hosting must account for model storage, cold-start time, CPU fallback latency, RAM, and build limits. A hosted CPU demo can be useful for a small number of requests, but performance must be measured on the chosen platform. Do not claim GPU acceleration without verifying the deployed device.

If a cloud demo is later needed, use a private source repository during development, place model files in a controlled model store, use environment variables for credentials, and keep database access restricted. A public app should add authentication, upload quotas, rate limits, safe temporary-file cleanup, database backups, and monitoring before it is exposed to users. Never add API keys to source code or commit them to Git.

## Release checklist

1. Confirm dataset license permits the intended academic use.
2. Rebuild the environment from the lock file on a clean machine.
3. Verify GPU or explicitly benchmark the CPU fallback.
4. Test normal, defective, oversized, malformed, and unsupported-format images.
5. Confirm the app cannot write outside its artifact directory from user-provided filenames.
6. Back up the SQLite database and required reports.
7. Publish source code only after confirming no dataset, weights, database, user image, or secret is included.
8. Describe the model as an academic prototype and show limitations with the reported metrics.

## Not deployed

No cloud deployment has been performed. The project is currently a local app. Public hosting would expose an unauthenticated upload interface and requires a separate security and resource review.
