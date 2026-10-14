from pathlib import Path

from ultralytics import YOLO
import torch


# 项目路径
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "yolo11n.pt"


# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("运行设备:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# 加载模型
model = YOLO(MODEL_PATH)


def detect_image(image_path):

    """
    图片目标检测
    """

    results = model(
        image_path,
        device=DEVICE
    )


    for result in results:

        # 保存检测结果
        save_path = result.save()

        print("结果保存:", save_path)


        # 输出检测目标
        boxes = result.boxes

        print("\n检测结果:")

        for box in boxes:

            cls_id = int(box.cls[0])

            confidence = float(box.conf[0])

            name = model.names[cls_id]

            print(
                f"{name}: {confidence:.2%}"
            )


if __name__ == "__main__":

    image = input(
        "请输入图片路径:"
    )

    detect_image(image)