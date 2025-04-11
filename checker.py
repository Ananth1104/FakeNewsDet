from scripts.db import SessionLocal
from scripts.models import User

session = SessionLocal()
users = session.query(User).all()
for user in users:
    print(user.id, user.username, user.email)
