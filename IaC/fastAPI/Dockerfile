FROM python:3.11

WORKDIR /app

# Copia o requirements.txt que está em ../ (IaC/requirements.txt)
COPY ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
