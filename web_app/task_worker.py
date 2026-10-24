
import json
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Lock

from task_queue import redis_client

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
UPLOAD_DIR = PROJECT_DIR / "web_app" / "uploads"
RESULT_DIR = PROJECT_DIR / "web_app" / "results"
MODEL_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_DIR / "web_app"))

from yolo_detect import model, DEVICE
from vision_api import analyze_image
from ultralytics import SAM
from database import save_analysis

SAM_MODEL_PATH = MODEL_DIR / "sam_b.pt"
sam_model = SAM(str(SAM_MODEL_PATH))
model_lock = Lock()


def process_task(task: dict) -> None:
    task_id = str(task["task_id"])
    data = task.get("data", {})
    image_path = Path(data["image_path"])
    image_name = data.get("image_name") or image_path.name
    user_id = data.get("user_id")

    try:
        redis_client.hset(
            f"task:{task_id}",
            mapping={"status": "processing", "error": ""},
        )

        if not image_path.is_file():
            raise FileNotFoundError(f"找不到上传图片：{image_path}")

        image_id = image_path.stem
        yolo_path = RESULT_DIR / f"{image_id}_yolo.jpg"
        sam_path = RESULT_DIR / f"{image_id}_sam.jpg"

        # YOLO 检测
        with model_lock:
            results = model(
                str(image_path),
                device=DEVICE,
                verbose=False,
            )

        detections = []
        boxes_xyxy = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append({
                    "name": model.names[cls_id],
                    "confidence": round(confidence, 4),
                })
                boxes_xyxy.append(box.xyxy[0].tolist())

            result.save(filename=str(yolo_path))

        # 没有检测结果时，也生成 YOLO 结果图
        if not yolo_path.exists():
            from PIL import Image
            with Image.open(image_path) as img:
                img.convert("RGB").save(yolo_path, format="JPEG")

        # SAM 分割
        if not boxes_xyxy:
            import shutil
            shutil.copyfile(image_path, sam_path)
        else:
            with model_lock:
                sam_results = sam_model(
                    str(image_path),
                    bboxes=boxes_xyxy,
                    device=DEVICE,
                    verbose=False,
                )
            sam_results[0].save(filename=str(sam_path))

        # 多模态视觉分析
        description = analyze_image(str(image_path))

        # 保存到共享 SQLite 数据库
        record_id = save_analysis(
            image_name,
            image_id,
            json.dumps(detections, ensure_ascii=False),
            str(description),
            user_id,
        )

        result_data = {
            "record_id": record_id,
            "filename": image_name,
            "detections": detections,
            "description": description,
            "yolo_image_url": f"/result/{yolo_path.name}",
            "sam_image_url": f"/result/{sam_path.name}",
        }

        redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "completed",
                "result": json.dumps(result_data, ensure_ascii=False),
                "error": "",
            },
        )
        print(f"任务 {task_id} 处理成功，历史记录 ID：{record_id}", flush=True)

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "failed",
                "error": error_message,
            },
        )
        print(f"任务 {task_id} 失败：{error_message}", flush=True)
        traceback.print_exc()


def main() -> None:
    print("真实图像分析 Worker 已启动，运行设备：" + DEVICE, flush=True)

    while True:
        try:
            item = redis_client.brpop("analysis_tasks", timeout=5)
            if item is None:
                continue

            _, task_json = item
            try:
                task = json.loads(task_json)
                process_task(task)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"任务数据异常：{exc}", flush=True)

        except Exception as exc:
            print(f"Worker 循环异常：{exc}，3 秒后重试", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
