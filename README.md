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

1. 阅读 [Ultralytics SAM 官方文档](https://docs.ultralytics.com/models/sam/)，了解模型及其权重文件。
2. 使用 Ultralytics 提供的模型权重获取方式获取 `sam_b.pt`。如果通过程序自动下载，请确认最终文件位于项目的 `models/` 目录中。
3. 确认模型文件路径为 `models/sam_b.pt`。
4. 在项目根目录执行以下命令，重新构建并启动 Docker 服务：

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
---

## 第二阶段：Web 系统与后端服务

在原有 YOLO 目标检测、SAM 目标分割和视觉大模型分析功能的基础上，项目进一步实现了可交互的 Web 应用与后端服务。

### 1. 用户注册与登录

- 使用 Streamlit 构建登录、注册界面。
- 后端通过 FastAPI 提供注册和登录接口。
- 使用 JWT Bearer Token 进行身份认证。
- 用户登录后可以访问受保护的业务接口。

### 2. 异步图像分析

- 前端提交图像分析任务后，后端返回任务标识。
- 系统使用 Redis 作为任务队列，并由独立 Worker 处理分析任务。
- 前端可以轮询任务状态，并在任务完成后查看分析结果。
- 异步处理将任务提交与实际模型推理分开，改善耗时任务的交互体验。

### 3. 历史记录与数据库

- 使用 SQLAlchemy 管理数据库访问。
- 使用 SQLite 保存分析历史记录。
- 支持查询、修改和删除历史记录。
- 历史记录与用户身份关联，业务接口遵循用户数据隔离规则。
- Docker Compose 将宿主机的 `data/` 目录挂载到容器的 `/app/data`，用于保存数据库文件。

### 4. 前后端服务与反向代理

系统主要由以下服务组成：

| 服务 | 作用 |
| --- | --- |
| Streamlit | 提供用户交互界面 |
| FastAPI | 处理用户认证、图像分析、任务与历史记录接口 |
| Redis | 为异步任务提供队列服务 |
| Worker | 从队列中获取任务并执行图像分析 |
| Nginx | 统一入口、反向代理与请求转发 |
| SQLite | 保存用户及分析记录等数据 |

### 5. 主要 API 接口

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| POST | `/register` | 用户注册 |
| POST | `/login` | 用户登录 |
| POST | `/analyze` | 同步图像分析 |
| POST | `/tasks` | 提交异步分析任务 |
| GET | `/tasks/{task_id}` | 查询任务状态 |
| GET | `/history` | 查询历史记录 |
| PUT | `/history/{record_id}` | 修改历史记录 |
| DELETE | `/history/{record_id}` | 删除历史记录 |
| GET | `/upload/{filename}` | 获取上传图片 |
| GET | `/result/{filename}` | 获取分析结果图片 |

需要身份认证的接口应携带有效的 Bearer Token。具体参数、请求体和响应格式以当前 API 实现及 `/docs` 文档为准。

### 6. Docker Compose 部署与访问

在项目根目录准备好 `.env` 文件、所需模型权重和 Docker 环境后，启动服务：

启动服务：`docker compose up -d --build`

查看服务状态：`docker compose ps`

查看 API 和 Worker 日志：`docker compose logs --tail=50 api worker`

推荐通过 Nginx 统一入口访问系统：

- Web 前端：`http://localhost:8080`
- API 文档：`http://localhost:8080/docs`

也可以在本机直接访问：

- Streamlit 前端：`http://localhost:8501`
- FastAPI 文档：`http://localhost:8000/docs`

### 7. 第二阶段功能验证建议

1. 注册一个测试用户并登录。
2. 上传图片，提交分析任务。
3. 查看任务状态，确认任务完成。
4. 检查 YOLO、SAM 和视觉大模型分析结果。
5. 查看历史记录，确认记录能够正常展示。
6. 修改并删除测试记录，验证对应操作。
7. 使用未登录状态访问受保护接口，确认系统拒绝未授权请求。
8. 检查 Docker 服务状态与 API、Worker 日志。

