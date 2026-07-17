# Built by Hugging Face Spaces (Docker SDK). deploy.py uploads the CONTENTS of the local
# webapp/ folder to the Space repo root (upload_folder(folder_path="webapp")), so this
# Dockerfile itself, requirements.txt, __init__.py, main.py, theme.py, data.py,
# components/, pages/ and data/ (incl. extract.duckdb) all land at the Space repo root,
# next to each other — the build context IS that root. COPY . /app/webapp therefore
# reconstitutes the exact local webapp/ package layout one level down, importable as
# `webapp` once /app is on PYTHONPATH.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/webapp
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["python", "-m", "webapp.main"]
