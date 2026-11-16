"""Create CampusGPT login accounts without putting passwords in shell history."""

import argparse
from getpass import getpass

from . import auth, db


def main():
    parser = argparse.ArgumentParser(description="Manage CampusGPT accounts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create a user account")
    create.add_argument("username")
    create.add_argument("--name", required=True)
    create.add_argument(
        "--role", required=True, choices=("student", "faculty", "admin")
    )
    args = parser.parse_args()
    password = getpass("Password (minimum 12 characters): ")
    if len(password) < 12:
        parser.error("password must be at least 12 characters")
    try:
        db.create_user(
            args.username, args.name, args.role, auth.hash_password(password)
        )
    except Exception as exc:
        parser.error(f"could not create user: {exc}")
    print(f"Created {args.role} account: {args.username.strip().lower()}")


if __name__ == "__main__":
    main()
