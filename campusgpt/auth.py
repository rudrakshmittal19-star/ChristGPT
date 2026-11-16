"""SQLite-backed Argon2 authentication and JWT role checks."""

import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import db
from .config import ENV, JWT_SECRET

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify(username: str, password: str):
    user = db.get_user(username)
    if not user:
        return None
    try:
        return (
            user if _password_hasher.verify(user["password_hash"], password) else None
        )
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return None


def seed_development_users():
    if ENV != "dev":
        return
    for username, name, role in (
        ("student", "Demo Student", "student"),
        ("faculty", "Demo Faculty", "faculty"),
        ("admin", "Demo Admin", "admin"),
    ):
        if not db.get_user(username):
            db.create_user(username, name, role, hash_password("christ123"))


def create_token(username: str) -> str:
    user = db.get_user(username)
    if not user:
        raise HTTPException(401, "Unknown user")
    return jwt.encode(
        {
            "sub": user["username"],
            "role": user["role"],
            "name": user["name"],
            "exp": int(time.time()) + 8 * 3600,
        },
        JWT_SECRET,
        "HS256",
    )


def current_user(cred: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    try:
        return jwt.decode(cred.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")


def require_role(role: str):
    def dep(user=Depends(current_user)):
        if user["role"] != role:
            raise HTTPException(403, f"{role} access only")
        return user

    return dep
