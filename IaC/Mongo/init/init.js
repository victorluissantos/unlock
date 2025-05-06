// Cria o usuário admin no banco "admin"
db = db.getSiblingDB('admin');
db.createUser({
  user: 'admin',
  pwd: 'adminpassword',
  roles: [{ role: 'root', db: 'admin' }]
});

// Cria o banco de aplicação e a collection de exemplo
db = db.getSiblingDB('meu_banco');
db.createCollection('exemplo');
