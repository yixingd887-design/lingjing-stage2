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



def update_analysis(record_id, yolo_result=None, vision_result=None):
    """更新指定分析记录的结果。"""
    db = SessionLocal()
    try:
        record = db.query(AnalysisRecord).filter(
            AnalysisRecord.id == record_id
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


def delete_analysis(record_id):
    """删除指定分析记录。"""
    db = SessionLocal()
    try:
        record = db.query(AnalysisRecord).filter(
            AnalysisRecord.id == record_id
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
