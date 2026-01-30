from fastapi import FastAPI
from infra.db.database import engine, Base
from api.routers import items, chat

# Create tables (for simplicity in this example)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="GRACE Backend",
    description="Backend API structured with 4 layers: Infra, Domain, App, API",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items.router)
app.include_router(chat.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to GRACE Backend API"}
