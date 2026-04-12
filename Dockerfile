FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL=https://api.groq.com/openai/v1
ENV MODEL_NAME=llama-3.3-70b-versatile

CMD ["python", "-m", "uvicorn", "main:app", \
"--host", "127.0.0.1", "--port", "7860", \
"--workers", "1", "--log-level", "info", \
"--timeout-keep-alive", "75"]
