
from pathlib import Path

import requests
import streamlit as st

import os
import time

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
IMAGE_BASE_URL = "http://localhost:8000"
API_URL = API_BASE_URL + "/analyze"
TASKS_URL = API_BASE_URL + "/tasks"
HISTORY_URL = API_BASE_URL + "/history"
PROJECT_DIR = Path(__file__).resolve().parent.parent

LOGIN_URL = API_BASE_URL + "/login"
REGISTER_URL = API_BASE_URL + "/register"



def login_user(username, password):
    """向后端发送登录请求，成功后保存 JWT。"""
    try:
        response = requests.post(
            LOGIN_URL,
            data={
                "username": username,
                "password": password,
            },
            timeout=15,
        )

        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.username = username
            return True, "登录成功"

        try:
            message = response.json().get("detail", "登录失败")
        except ValueError:
            message = "登录失败"

        return False, message

    except requests.RequestException:
        return False, "无法连接后端，请检查服务是否正常运行"


def register_user(username, password):
    """向后端发送注册请求。"""
    try:
        response = requests.post(
            REGISTER_URL,
            data={
                "username": username,
                "password": password,
            },
            timeout=15,
        )

        if response.status_code == 201:
            return True, "注册成功，请登录"

        try:
            message = response.json().get("detail", "注册失败")
        except ValueError:
            message = "注册失败"

        return False, message

    except requests.RequestException:
        return False, "无法连接后端，请检查服务是否正常运行"

st.set_page_config(
    page_title="灵境 AI 图像分析系统",
    page_icon="🔍",
    layout="wide"
)

# 登录状态
if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "username" not in st.session_state:
    st.session_state.username = None
# 未登录时显示登录和注册页面
if not st.session_state.access_token:
    st.title("灵境 AI 图像分析系统")
    st.subheader("用户登录")

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("用户名")
            login_password = st.text_input("密码", type="password")
            login_submit = st.form_submit_button("登录")

        if login_submit:
            success, message = login_user(
                login_username.strip(),
                login_password,
            )

            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with register_tab:
        with st.form("register_form"):
            register_username = st.text_input(
                "新用户名",
                key="register_username",
            )
            register_password = st.text_input(
                "新密码",
                type="password",
                key="register_password",
            )
            confirm_password = st.text_input(
                "确认密码",
                type="password",
                key="confirm_password",
            )

            register_submit = st.form_submit_button("注册")

        if register_submit:
            username = register_username.strip()

            if len(username) < 3 or len(username) > 50:
                st.error("用户名长度必须为 3 到 50 个字符")
            elif len(register_password) < 8:
                st.error("密码长度至少为 8 个字符")
            elif register_password != confirm_password:
                st.error("两次输入的密码不一致")
            else:
                success, message = register_user(
                    username,
                    register_password,
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.stop()


st.write("上传一张图片，对比目标检测与多模态大模型的分析结果。")

col_title, col_logout = st.columns([4, 1])

with col_title:
    st.title("灵境 AI 图像分析系统")
    st.caption(f"当前登录用户：{st.session_state.username}")

with col_logout:
    st.write("")
    if st.button("退出登录"):
        st.session_state.access_token = None
        st.session_state.username = None
        st.rerun()

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
        with st.spinner("正在提交分析任务……"):
            try:
                response = requests.post(
                    TASKS_URL,
                    headers={
                        "Authorization": f"Bearer {st.session_state.access_token}"
                    },
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=30
                )
                response.raise_for_status()
                task_data = response.json()
                task_id = str(task_data["task_id"])

                progress = st.progress(0)
                status_text = st.empty()
                deadline = time.time() + 600
                data = None

                while time.time() < deadline:
                    task_response = requests.get(
                        f"{TASKS_URL}/{task_id}",
                        headers={
                            "Authorization": f"Bearer {st.session_state.access_token}"
                        },
                        timeout=15
                    )
                    task_response.raise_for_status()
                    task_info = task_response.json()
                    status = task_info.get("status")

                    if status == "completed":
                        data = task_info.get("result")
                        if not isinstance(data, dict):
                            raise ValueError("任务已完成，但没有获取到有效的分析结果。")
                        progress.progress(100)
                        status_text.success("分析完成！")
                        break

                    if status == "failed":
                        raise RuntimeError(
                            task_info.get("error", "后台分析任务失败。")
                        )

                    if status == "pending":
                        status_text.info("任务正在排队，等待 Worker 处理……")
                        progress.progress(10)
                    elif status == "processing":
                        status_text.info("AI 正在分析图片，请耐心等待……")
                        progress.progress(50)
                    else:
                        status_text.info(f"任务状态：{status}")

                    time.sleep(2)
                else:
                    raise TimeoutError("等待分析结果超过 10 分钟，请刷新历史记录确认任务是否完成。")




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
            except (RuntimeError, TimeoutError) as exc:
                st.error(f"分析任务未能正常完成：{exc}")
st.divider()
st.header("历史分析记录")

if st.button("刷新历史记录"):
    st.rerun()

try:
    response = requests.get(
    HISTORY_URL,
    headers={
        "Authorization": f"Bearer {st.session_state.access_token}"
    },
    timeout=10,
)
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

                with st.expander("编辑分析结果", key=f"edit_area_{record['id']}"):
                    edited_yolo = st.text_area(
                        "YOLO 文字检测结果",
                        value=record.get("yolo_result") or "",
                        key=f"edit_yolo_{record['id']}",
                    )

                    edited_vision = st.text_area(
                        "多模态大模型分析结果",
                        value=record.get("vision_result") or "",
                        key=f"edit_vision_{record['id']}",
                    )

                    if st.button("保存修改", key=f"save_{record['id']}"):
                        try:
                            update_response = requests.put(
                                f"{API_BASE_URL}/history/{record['id']}",
                                headers={
                                    "Authorization": (
                                        f"Bearer {st.session_state.access_token}"
                                    )
                                },
                                params={
                                    "yolo_result": edited_yolo,
                                    "vision_result": edited_vision,
                                },
                                timeout=10,
                            )
                            update_response.raise_for_status()
                            st.success("历史记录修改成功")
                            st.rerun()
                        except requests.RequestException as exc:
                            st.error(f"修改历史记录失败：{exc}")

                if st.button("删除这条记录", key=f"delete_{record['id']}"):
                    try:
                        delete_response = requests.delete(
                            f"{API_BASE_URL}/history/{record['id']}",
                            headers={
                                "Authorization": (
                                    f"Bearer {st.session_state.access_token}"
                                )
                            },
                            timeout=10,
                        )
                        delete_response.raise_for_status()
                        st.success("历史记录删除成功")
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"删除历史记录失败：{exc}")
    else:
        st.info("暂时没有历史分析记录。")

except requests.RequestException as exc:
    st.error(f"读取历史记录失败：{exc}")
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"历史记录格式异常：{exc}")