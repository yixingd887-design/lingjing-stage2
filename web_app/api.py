from PIL import Image
from io import BytesIO
import sys
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

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


@app.get("/result/{filename}")
def get_result(filename: str):
    # 只允许访问结果目录下的文件名
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="文件名不合法")

    path = RESULT_DIR / filename

    if not path.is_file():
        raise HTTPException(status_code=404, detail="结果图片不存在")

    return FileResponse(path, media_type="image/jpeg")
