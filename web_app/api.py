from PIL import Image
from io import BytesIO
import sys
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
import secrets

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from database import SessionLocal, User, hash_password, verify_password
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from database import (
    init_db,
    migrate_db,
    save_analysis,
    get_recent_analyses,
    update_analysis,
    delete_analysis,
)

# 项目根目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
UPLOAD_DIR = PROJECT_DIR / "web_app" / "uploads"
RESULT_DIR = PROJECT_DIR / "web_app" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 让 Python 能找到第二阶段的源码
sys.path.insert(0, str(SRC_DIR))

from yolo_detect import model, DEVICE
from vision_api import analyze_image
from ultralytics import SAM

SAM_MODEL_PATH = PROJECT_DIR / "models" / "sam_b.pt"

sam_model = SAM(str(SAM_MODEL_PATH))

app = FastAPI(title="灵境 AI 图像分析系统")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise RuntimeError("缺少 JWT_SECRET_KEY 环境变量")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": username,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_error = HTTPException(
        status_code=401,
        detail="登录已失效或令牌无效，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        username = payload.get("sub")
        if not isinstance(username, str) or not username:
            raise credentials_error
    except jwt.InvalidTokenError:
        raise credentials_error

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_error

    return user

@app.on_event("startup")
def startup_event():
    init_db()
    migrate_db()


ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def home():
    return {"message": "灵境 AI 图像分析后端运行正常"}

@app.post("/register", status_code=201)
def register(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    username = form_data.username.strip()
    password = form_data.password

    if len(username) < 3 or len(username) > 50:
        raise HTTPException(
            status_code=400,
            detail="用户名长度必须为 3 到 50 个字符",
        )

    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="密码长度至少为 8 个字符",
        )

    existing_user = db.query(User).filter(
        User.username == username
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="用户名已存在",
        )

    user = User(
        username=username,
        hashed_password=hash_password(password),
    )
    db.add(user)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="注册失败，请稍后重试",
        )

    return {"message": "注册成功", "username": username}


@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    username = form_data.username.strip()

    user = db.query(User).filter(
        User.username == username
    ).first()

    if user is None or not verify_password(
        form_data.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user.username)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="请上传 JPG、PNG 或 WebP 图片"
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="图片文件扩展名不正确"
        )

    # 使用随机文件名，避免覆盖其他上传文件
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="上传的图片为空")

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图片不能超过 10 MB")

        # 验证文件内容是否为有效图片
        try:
            with Image.open(BytesIO(content)) as img:
                img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="文件内容不是有效的图片"
            )

        image_path.write_bytes(content)


        # YOLO 检测
        def run_yolo():
            results = model(str(image_path), device=DEVICE, verbose=False)
            detections = []
            boxes_xyxy = []
            output_path = RESULT_DIR / f"{image_path.stem}_yolo.jpg"

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    detections.append({
                        "name": model.names[cls_id],
                        "confidence": round(confidence, 4)
                    })

                    boxes_xyxy.append(box.xyxy[0].tolist())

                result.save(filename=str(output_path))

            return detections, boxes_xyxy, output_path

        detections, boxes_xyxy, output_path = await run_in_threadpool(run_yolo)



        # SAM 目标分割
        def run_sam():
            sam_output_path = RESULT_DIR / f"{image_path.stem}_sam.jpg"

            if not boxes_xyxy:
                shutil.copyfile(image_path, sam_output_path)
                return sam_output_path

            results = sam_model(
                str(image_path),
                bboxes=boxes_xyxy,
                device=DEVICE,
                verbose=False
            )

            result = results[0]
            result.save(filename=str(sam_output_path))

            return sam_output_path

        sam_output_path = await run_in_threadpool(run_sam)


        # 多模态大模型分析
        description = await run_in_threadpool(analyze_image, str(image_path))


        # 保存分析记录到数据库
        await run_in_threadpool(
            save_analysis,
            file.filename or image_path.name,
            image_path.stem,
            str(detections),
            str(description),
            current_user.id
        )


        return {
            "filename": file.filename,
            "detections": detections,
            "description": description,
            "yolo_image_url": f"/result/{output_path.name}",
            "sam_image_url": f"/result/{sam_output_path.name}"
        }



    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"图像分析失败：{type(exc).__name__}: {exc}"
        )
    finally:
        await file.close()


@app.get("/history")
def get_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
):
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit 必须在 1 到 100 之间"
        )

    records = get_recent_analyses(limit, user_id=current_user.id)
    history = []

    for record in records:
        image_id = record.image_id

        # 查找原始上传图片
        original_url = None
        if image_id:
            for path in UPLOAD_DIR.glob(f"{image_id}.*"):
                if path.is_file():
                    original_url = f"/upload/{path.name}"
                    break

        # YOLO 检测图和 SAM 分割图
        yolo_path = RESULT_DIR / f"{image_id}_yolo.jpg" if image_id else None
        sam_path = RESULT_DIR / f"{image_id}_sam.jpg" if image_id else None

        history.append({
            "id": record.id,
            "image_name": record.image_name,
            "created_at": record.created_at.isoformat(),
            "yolo_result": record.yolo_result,
            "vision_result": record.vision_result,
            "original_image_url": original_url,
            "yolo_image_url": (
                f"/result/{yolo_path.name}"
                if yolo_path and yolo_path.is_file()
                else None
            ),
            "sam_image_url": (
                f"/result/{sam_path.name}"
                if sam_path and sam_path.is_file()
                else None
            )
        })

    return history


@app.put("/history/{record_id}")
def update_history(
    record_id: int,
    yolo_result: str | None = None,
    vision_result: str | None = None,
    current_user: User = Depends(get_current_user),
):


    success = update_analysis(
    record_id,
    user_id=current_user.id,
    yolo_result=yolo_result,
    vision_result=vision_result,
)


    if not success:
        raise HTTPException(
            status_code=404,
            detail="历史记录不存在",
        )

    return {"message": "历史记录更新成功", "id": record_id}



@app.delete("/history/{record_id}")
def delete_history(
    record_id: int,
    current_user: User = Depends(get_current_user),
):

    success = delete_analysis(
    record_id,
    user_id=current_user.id,
)

    if not success:
        raise HTTPException(
            status_code=404,
            detail="历史记录不存在",
        )

    return {"message": "历史记录删除成功", "id": record_id}


@app.get("/upload/{filename}")
def get_upload(filename: str):
    # 只允许访问文件名，防止路径穿越
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="文件名不合法"
        )

    path = UPLOAD_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="原始图片不存在"
        )

    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }

    media_type = media_types.get(path.suffix.lower())
    if media_type is None:
        raise HTTPException(
            status_code=400,
            detail="不支持的图片格式"
        )

    return FileResponse(path, media_type=media_type)

@app.get("/result/{filename}")
def get_result(filename: str):
    # 只允许访问结果目录下的文件名
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="文件名不合法")

    path = RESULT_DIR / filename

    if not path.is_file():
        raise HTTPException(status_code=404, detail="结果图片不存在")

    return FileResponse(path, media_type="image/jpeg")
