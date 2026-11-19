FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a code change does not re-run the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jobmatch/ ./jobmatch/
COPY pyproject.toml companies.txt ./

# The index is built from the public APIs rather than baked into the image:
# it goes stale, and it is far larger than the code.
#   docker run -v $(pwd)/data:/app/data -p 8000:8000 jobmatch
VOLUME /app/data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "jobmatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
