# Usa un'immagine base di Python leggera
FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /app

# Installa le dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia i file delle dipendenze
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copia il resto del codice
COPY . .

# Espone la porta utilizzata da FastAPI
EXPOSE 8000

# Comando per avviare l'API
CMD ["python", "api_gloss.py"]
