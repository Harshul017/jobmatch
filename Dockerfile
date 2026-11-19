FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a code change does not re-run the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jobmatch/ ./jobmatch/
COPY pyproject.toml companies.txt ./

# The slim index, built by `python -m jobmatch.prepare_deploy`. Baked in rather
# than mounted, because the free tiers of most hosts have no persistent disk.
# To run against the full local index instead, skip this and mount it:
#   docker run -v $(pwd)/data:/app/data -p 8000:8000 jobmatch
COPY data/deploy/ ./data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Hosts inject the port they want; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn jobmatch.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
