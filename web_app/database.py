from datetime import datetime
import hashlib
import hmac
import secrets
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker

# 数据库文件放在项目的 data 目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR / 'analysis.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    image_name = Column(String(255), nullable=False)
    image_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    yolo_result = Column(Text, nullable=True)
    vision_result = Column(Text, nullable=True)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

def hash_password(password: str) -> str:
    """使用 PBKDF2 对密码进行哈希，不保存明文密码。"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        600_000
    ).hex()

    return f"{salt}${password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """验证用户输入的密码是否匹配数据库中的哈希值。"""
    try:
        salt, expected_hash = stored_hash.split("$", 1)
        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            600_000
        ).hex()

        return hmac.compare_digest(actual_hash, expected_hash)
    except (ValueError, TypeError):
        return False

def init_db():
    Base.metadata.create_all(bind=engine)


def migrate_db():
    """为已有的分析记录表增加用户归属字段。"""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    columns = [
        column["name"]
        for column in inspector.get_columns("analysis_records")
    ]

    if "user_id" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE analysis_records "
                    "ADD COLUMN user_id INTEGER"
                )
            )



def save_analysis(image_name, image_id, yolo_result, vision_result, user_id):
    db = SessionLocal()
    try:
        record = AnalysisRecord(
            image_name=image_name,
            image_id=image_id,
            yolo_result=yolo_result,
            vision_result=vision_result,
            user_id=user_id,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()



def get_recent_analyses(limit=20, user_id=None):
    db = SessionLocal()
    try:
        query = db.query(AnalysisRecord)

        if user_id is not None:
            query = query.filter(
                AnalysisRecord.user_id == user_id
            )

        return (
            query
            .order_by(AnalysisRecord.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()




def update_analysis(
    record_id,
    user_id,
    yolo_result=None,
    vision_result=None,
):
    """更新指定分析记录的结果。"""
    db = SessionLocal()
    try:
        record = db.query(AnalysisRecord).filter(
    AnalysisRecord.id == record_id,
    AnalysisRecord.user_id == user_id,
).first()

        if record is None:
            return False

        if yolo_result is not None:
            record.yolo_result = yolo_result

        if vision_result is not None:
            record.vision_result = vision_result

        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()



def delete_analysis(record_id, user_id):
    """只删除当前用户自己的分析记录。"""
    db = SessionLocal()
    try:
        record = db.query(AnalysisRecord).filter(
            AnalysisRecord.id == record_id,
            AnalysisRecord.user_id == user_id,
        ).first()

        if record is None:
            return False

        db.delete(record)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
