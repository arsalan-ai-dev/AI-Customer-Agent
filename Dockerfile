FROM python:3.10-slim

WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir streamlit

# Copy application files
COPY . .

# Expose FastAPI backend and Phoenix ports
EXPOSE 8000 6006

# Default command for the backend image
CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]