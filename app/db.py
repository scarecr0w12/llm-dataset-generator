from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR / 'dataset_manager.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
