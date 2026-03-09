import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app: FastAPI):
    """ configure CORS middleware based on environment"""

    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "production":
        origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    elif environment == "staging":
        origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    else:
        origins = ["*"] # allow all in development

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )