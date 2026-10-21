# 灵境视觉智能分析系统

## 项目简介

本项目实现了一个结合目标检测、图像目标分割与视觉大语言模型的智能图片分析系统。

系统集成以下三种 AI 分析功能：

1. **YOLO 目标检测**：识别图片中的目标，输出目标类别、置信度，并生成检测结果图片。
2. **SAM 目标分割**：利用 YOLO 检测得到的目标边界框作为提示信息，引导 SAM 对目标进行分割，并生成分割结果图片。
3. **视觉大模型分析**：调用视觉大模型 API，对图片进行语义理解、场景描述和环境分析。

项目采用 Python、PyTorch、Ultralytics、FastAPI 和 Streamlit 等技术，实现从图片上传、模型推理到结果可视化展示的完整流程，并支持通过 Docker Compose 部署。

说明：当前 SAM 分割采用 YOLO 检测框作为提示，并非自动分割图片中的所有物体。


---

## 功能介绍


### 1. YOLO目标检测

采用 YOLO 模型进行本地视觉检测。

功能：

- 检测图片中的目标
- 输出目标类别
- 输出检测置信度
- 保存检测结果图片


示例：

输入：

images/bus.jpg


输出：

- bus
- person

并生成检测结果图片。
---

---

### 2. SAM 目标分割

使用 Segment Anything Model（SAM）进行图像目标分割。

功能：

- 接收 YOLO 检测得到的目标边界框。
- 根据目标框引导 SAM 生成目标分割结果。
- 保存并展示分割结果图片。
- 当 YOLO 未检测到目标时，当前实现返回原始图片，不执行基于检测框的分割。

模型权重文件：

`models/sam_b.pt`

---



### 3. 视觉大模型分析

调用视觉大模型 API，对图片进行多模态理解。

功能：

- 场景描述
- 目标理解
- 图片文字识别
- 环境分析


---

## 技术栈

- Python 3.11
- PyTorch
- TorchVision
- Ultralytics YOLO / SAM
- FastAPI
- Streamlit
- OpenAI SDK（用于视觉大模型接口调用，具体以实际代码为准）
- OpenCV
- Docker Compose

---

## 项目结构

```text
lingjing-stage2/
├── src/
│   ├── main.py
│   ├── yolo_detect.py
│   └── vision_api.py
├── web_app/
│   ├── api.py
│   ├── app.py
│   ├── uploads/
│   └── results/
├── images/
├── models/
│   ├── yolo11n.pt
│   └── sam_b.pt
├── results/
├── docs/
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

说明：以上为主要文件和目录，实际结构以项目仓库为准。

## 环境配置
创建虚拟环境：
python -m venv .venv

## 安装依赖
pip install -r requirements.txt

## API配置
在项目根目录创建：
.env
填写：
DASHSCOPE_API_KEY=your_api_key_here
注意：
.env 文件包含个人 API Key，不会上传到代码仓库。
项目已通过 .gitignore 忽略 .env 文件。
使用者需要自行申请视觉大模型 API Key。

## 运行方式
进入项目目录：
python src/main.py
选择：
1
使用 YOLO 目标检测。
选择：
2
使用视觉大模型分析。

## 实验环境
测试设备：
- Windows
- Python 3.11.9
- NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA


## 项目特点

本项目采用多模型协同分析的设计思路：

- **目标检测**：利用 YOLO 识别图片中的目标，并输出类别、置信度和检测结果图片。
- **目标分割**：利用 YOLO 检测框引导 SAM 对目标进行分割，生成分割结果图片。
- **语义理解**：调用视觉大模型 API，生成图片内容描述和场景分析。

系统通过 FastAPI 提供后端分析接口，并使用 Streamlit 构建网页，实现图片上传、模型调用和多种分析结果的可视化展示。同时支持通过 Docker Compose 部署，便于环境配置和演示。


## Docker 部署

### 1. 环境准备

安装并启动 Docker Desktop，确保 Docker 服务正常运行。

在项目根目录创建 `.env` 文件，填写自己的 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### SAM 模型文件准备

项目使用的 SAM 模型权重文件为 `models/sam_b.pt`，文件名必须保持一致。

由于该文件体积较大，GitHub 仓库未直接包含此文件。首次部署前，请按以下步骤准备模型：

1. 确保项目根目录下存在 `models/` 文件夹。
2. 从 Ultralytics 官方 SAM 模型权重来源获取对应的 SAM 权重文件，并确认使用的是与当前项目兼容的 `sam_b.pt`。
3. 将下载的文件放置在项目的 `models/` 目录下，最终路径为 `models/sam_b.pt`。
4. 确认文件存在后，重新构建并启动 Docker 服务：

   ```bash
   docker compose up -d --build
   ```

Docker 构建时会将 `models/` 目录复制到镜像中。如果模型文件缺失或路径不正确，API 服务可能无法正常启动。

如果本地已经存在 `models/sam_b.pt`，则无需重复下载。


### 2. 启动服务

在项目根目录执行：

```bash
docker compose up -d --build
```

查看容器运行状态：

```bash
docker compose ps
```

查看运行日志：

```bash
docker compose logs --tail=50
```

### 3. 访问服务

- 前端界面：http://localhost:8501
- 后端接口：http://localhost:8000/
- API 文档：http://localhost:8000/docs

### 4. 功能验证

1. 打开前端界面：http://localhost:8501。
2. 上传一张 JPG、PNG 或 WebP 格式的图片。
3. 点击「开始 AI 分析」。
4. 检查 YOLO 目标检测结果、目标类别和置信度。
5. 检查 SAM 目标分割结果图片。
6. 检查视觉大模型生成的图片描述和场景分析。
7. 如果图片未能正常显示或分析失败，查看前后端日志：

```bash
docker compose logs --tail=100 api
docker compose logs --tail=100 frontend
```

8. 确认三个功能均能正常运行后，完成部署验证。

### 5. 停止服务

```bash
docker compose down
```

说明：Docker 镜像使用 CPU 版 PyTorch，容器内推理默认使用 CPU。Windows 本地运行方式仍可使用原有 GPU 环境。