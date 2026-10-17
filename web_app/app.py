
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/analyze"
PROJECT_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="灵境 AI 图像分析系统",
    page_icon="🔍",
    layout="wide"
)

st.title("灵境 AI 图像分析系统")
st.write("上传一张图片，对比目标检测与多模态大模型的分析结果。")

uploaded_file = st.file_uploader(
    "选择一张图片",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="待分析图片",
        use_container_width=True
    )

    if st.button("开始 AI 分析", type="primary"):
        with st.spinner("正在调用 AI 模型，请稍候……"):
            try:
                uploaded_file.seek(0)

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=300
                )

                response.raise_for_status()
                data = response.json()

                left, right = st.columns(2)

                with left:
                    st.subheader("YOLO 目标检测")

                    result_url = (
                        "http://127.0.0.1:8000"
                        + data["yolo_image_url"]
                    )

                    st.image(
                        result_url,
                        caption="YOLO 标注结果",
                        use_container_width=True
                    )

                    detections = data.get("detections", [])

                    if detections:
                        st.write("识别到的目标：")
                        for item in detections:
                            st.write(
                                f"- {item['name']}："
                                f"{item['confidence']:.2%}"
                            )
                    else:
                        st.info("没有检测到目标。")

                with right:
                    st.subheader("多模态大模型分析")
                    st.write(data.get("description", "没有返回描述。"))

                st.success("分析完成！")

            except requests.RequestException as exc:
                st.error(f"后端连接失败或请求超时：{exc}")
            except (ValueError, KeyError) as exc:
                st.error(f"返回结果格式异常：{exc}")
