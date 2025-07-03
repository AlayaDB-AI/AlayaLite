from fastapi import FastAPI

from app.routers import client

app = FastAPI(
    title="AlayaLite-Standalone",
    description="The standalone service of AlayaLite",
    version="1.0.0",
)

app.include_router(client.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "AlayaLite standalone service is ready! Please use /api to access the API."}
