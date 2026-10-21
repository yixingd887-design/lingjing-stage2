from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
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
    image_name = Column(String(255), nullable=False)
    image_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    yolo_result = Column(Text, nullable=True)
    vision_result = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def save_analysis(image_name, image_id, yolo_result, vision_result):
    db = SessionLocal()
    try:
        record = AnalysisRecord(
            image_name=image_name,
            image_id=image_id,
            yolo_result=yolo_result,
            vision_result=vision_result
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()


def get_recent_analyses(limit=20):
    db = SessionLocal()
    try:
        return (
            db.query(AnalysisRecord)
            .order_by(AnalysisRecord.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()
