
from pathlib import Path

import requests
import streamlit as st

import os

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
IMAGE_BASE_URL = "http://localhost:8000"
API_URL = API_BASE_URL + "/analyze"
HISTORY_URL = API_BASE_URL + "/history"
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

                
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("YOLO 目标检测")

                    result_url = IMAGE_BASE_URL + data["yolo_image_url"]
                    st.write("YOLO 图片地址：", result_url)

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

                with col2:
                    st.subheader("SAM 目标分割")

                    sam_url = IMAGE_BASE_URL + data["sam_image_url"]
                    st.write("SAM 图片地址：", sam_url)

                    st.image(
                        sam_url,
                        caption="SAM 分割结果",
                        use_container_width=True
                    )

                with col3:
                    st.subheader("多模态大模型分析")
                    st.write(data.get("description", "没有返回描述。"))

                st.success("分析完成！")


            except requests.RequestException as exc:
                st.error(f"后端连接失败或请求超时：{exc}")
            except (ValueError, KeyError) as exc:
                st.error(f"返回结果格式异常：{exc}")
st.divider()
st.header("历史分析记录")

if st.button("刷新历史记录"):
    st.rerun()

try:
    response = requests.get(HISTORY_URL, timeout=10)
    response.raise_for_status()
    history_records = response.json()

    if history_records:
        st.write(f"共找到 {len(history_records)} 条历史记录")

        for record in history_records:
            title = (
                f"记录 #{record['id']} - "
                f"{record['image_name']} - "
                f"{record['created_at']}"
            )

            with st.expander(title):
                # 展示原图、YOLO 检测图和 SAM 分割图
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("原始图片")
                    original_url = record.get("original_image_url")
                    if original_url:
                        st.image(
                            IMAGE_BASE_URL + original_url,
                            use_container_width=True
                        )
                    else:
                        st.info("暂无原图地址，请检查是否为旧记录。")

                with col2:
                    st.subheader("YOLO 检测图")
                    yolo_url = record.get("yolo_image_url")
                    if yolo_url:
                        st.image(
                            IMAGE_BASE_URL + yolo_url,
                            use_container_width=True
                        )
                    else:
                        st.info("暂无 YOLO 图片。")

                with col3:
                    st.subheader("SAM 分割图")
                    sam_url = record.get("sam_image_url")
                    if sam_url:
                        st.image(
                            IMAGE_BASE_URL + sam_url,
                            use_container_width=True
                        )
                    else:
                        st.info("暂无 SAM 图片。")

                st.subheader("YOLO 文字检测结果")
                st.code(record.get("yolo_result") or "暂无结果")

                st.subheader("多模态大模型分析结果")
                st.write(record.get("vision_result") or "暂无结果")
    else:
        st.info("暂时没有历史分析记录。")

except requests.RequestException as exc:
    st.error(f"读取历史记录失败：{exc}")
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"历史记录格式异常：{exc}")
