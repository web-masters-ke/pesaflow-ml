"""JWT Authentication Middleware."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger

from serving.app.settings import Settings, get_settings

security = HTTPBearer(auto_error=False)


async def verify_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Verify JWT token and return decoded payload."""
    if credentials is None:
        # Allow unauthenticated access in development
        if settings.ENVIRONMENT == "development":
            return {"sub": "dev-user", "role": "ML_ADMIN"}
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")


def require_role(*roles: str):
    """Dependency that checks JWT payload for required role."""

    async def role_checker(payload: dict = Depends(verify_jwt)) -> dict:
        user_role = payload.get("role", "")
        if user_role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {roles}",
            )
        return payload

    return role_checker
