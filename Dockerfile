FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860


CMD ["python", "-m", "uvicorn", "main:app", \
"--host", "0", "--port", "7860", \
"--workers", "1", "--log-level", "info", \
"--timeout-keep-alive", "75"]
