from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from routes.decripto import router as decripto_router
import os

app = FastAPI()
app.include_router(decripto_router)

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}

@app.get("/ping-db")
async def ping_db():
    try:
        server_status = await app.database.command("ping")
        return {"ping": server_status}
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup_db():
    mongo_url = os.getenv("MONGO_URL", "mongodb://admin:adminpassword@mongo:27017/admin")
    app.mongodb_client = AsyncIOMotorClient(mongo_url)
    app.database = app.mongodb_client["meu_banco"]

@app.on_event("shutdown")
async def shutdown_db():
    app.mongodb_client.close()

