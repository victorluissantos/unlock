FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências necessárias
RUN apt-get update && apt-get install -y \
    gnupg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Adiciona a chave GPG oficial do MongoDB
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-archive-keyring.gpg

# Adiciona o repositório do MongoDB
RUN echo "deb [ arch=amd64 signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Instala o MongoDB
RUN apt-get update && apt-get install -y mongodb-org

# Cria diretório para dados do MongoDB
RUN mkdir -p /data/db

# Copia scripts de inicialização
COPY init/ /docker-entrypoint-initdb.d/

# Expõe a porta padrão do MongoDB
EXPOSE 27017

# Comando padrão para iniciar o MongoDB
CMD ["mongod", "--bind_ip_all"]
