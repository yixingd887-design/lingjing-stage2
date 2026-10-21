from PIL import Image
from io import BytesIO
import sys
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from database import save_analysis, get_recent_analyses

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

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def home():
    return {"message": "灵境 AI 图像分析后端运行正常"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
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
            str(description)
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
def get_history(limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit 必须在 1 到 100 之间"
        )

    records = get_recent_analyses(limit)
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
