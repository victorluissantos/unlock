FROM python:3.9-slim

WORKDIR /app

# Copia o mesmo requirements.txt
COPY ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
