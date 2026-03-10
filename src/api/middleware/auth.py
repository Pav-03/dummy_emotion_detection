import os
import jwt
import time
import sys
from fastapi import Request, HTTPException
from fastapi.responses import  JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger("auth")

# configuration -read from environment variables
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_secret_key")
EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_TIME_MINUTES", "30"))
ALGORITHM = "HS256"

# Endpoints that required no authentication

PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/auth/login",
    "/favicon.ico",
}

# Users (for demo purposes, replace with DB in production)
USERS = {
    "pavan": "secure123",
    "api_service": "service-key-2026"
}

# Create token

def create_token(user_id: str) -> str:

    # Create JWT token with user_id and expiry
    payload = {
        "user_id" : user_id,
        "exp": time.time() + EXPIRY_MINUTES * 60,
        "iat" : time.time()
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"token created for user_id: {user_id}")
    return token

# Verify the token
def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        logger.warning("invalid token provided")
        raise HTTPException(status_code=401, detail="Invalid token")    
    
# authenticate the middleware
async def auth_guard(request: Request, call_next):

    # skip authentication for public paths
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    
    # skip authentication for OPTIONS method (CORS preflight)
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # get token from Authorization header
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        logger.warning(f"No auth header for {request.method} {request.url.path}")

        return JSONResponse(
            status_code=401,
            content={"detail": "Authorization header missing or invalid"}
        )
    
    # check formatet of the token
    token = auth_header.split(" ")
    if len(token) != 2 or token[0].lower() != "bearer":
        logger.warning(f"Invalid auth header format for {request.method} {request.url.path}")
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid authorization header format"}
        )
    token = token[1]

    # verify the token
    try:
        payload = verify_token(token)
        request.state.user_id = payload.get("user_id")
        logger.info(f"Authenticated user_id: {request.state.user_id} for {request.method} {request.url.path}")
    except HTTPException as e:
        logger.warning(f"Authentication failed for {request.method} {request.url.path}: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail}
        )
    
    # token is valid, proceed to the next middleware or route handler
    response = await call_next(request)
    return response

# Login helper function

def authenticate_user(username: str, pasword: str) -> str:

    """ verify user credentialss and return a JWT token if valid"""
    if username not in USERS:
        logger.warning(f"Authentication failed for username: {username} - user not found")
        raise HTTPException(
            status_code=401, 
            detail="Invalid username or password")
    if USERS[username] != pasword:
        logger.warning(f"Authentication failed for username: {username} - incorrect password")
        raise HTTPException(
            status_code=401, 
            detail="Invalid username or password")
    
    token = create_token(username)
    logger.info(f"User {username} authenticated successfully")
    return token