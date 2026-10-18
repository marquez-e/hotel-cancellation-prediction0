# 1. Python image
FROM python:3.9-slim

# 2. Set working directory in the container
WORKDIR /app

# Copy necessary files to container
COPY requirements.txt .
COPY main.py .
COPY final_model.pkl .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]