from yolo_detect import detect_image
from vision_api import analyze_image


print("======================")
print("灵境视觉智能分析系统")
print("======================")

print("""
请选择功能:

1. YOLO目标检测
2. 视觉大模型分析
""")


choice = input("请输入编号:")


image_path = input(
    "请输入图片路径:"
)


if choice == "1":

    print("\n开始YOLO检测...\n")

    detect_image(image_path)


elif choice == "2":

    print("\n开始视觉大模型分析...\n")

    result = analyze_image(
        image_path
    )

    print("\n分析结果:")
    print("----------------")
    print(result)


else:

    print("输入错误")