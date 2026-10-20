# 灵境视觉智能分析系统

## 项目简介

本项目实现了一个结合传统计算机视觉模型与视觉大语言模型的智能图片分析系统。

系统融合 YOLO 目标检测模型和视觉大模型 API，可以分别完成：

1. 图片目标检测
2. 图片语义理解与场景分析

项目基于 Python、PyTorch 和 CUDA GPU 开发。


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

### 2. 视觉大模型分析

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
- YOLO
- OpenAI SDK
- CUDA
- OpenCV
---

## 项目结构

```text
lingjing-stage2
│
├── src                 # 核心代码
│   ├── main.py         # 主程序入口
│   ├── yolo_detect.py  # YOLO检测模块
│   └── vision_api.py   # 视觉大模型接口
│
├── images              # 测试图片
│
├── models              # AI模型文件
│   └── yolo11n.pt
│
├── results             # 检测结果
│
├── docs                # 项目文档
│
├── requirements.txt    # 环境依赖
│
├── .gitignore          # Git忽略配置
│
└── README.md           # 项目说明
```

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
本项目采用“双视觉系统”设计：
YOLO负责快速目标检测与定位；
视觉大模型负责复杂场景理解。
二者结合，实现从目标检测到语义理解的完整视觉分析流程

## Docker 部署

### 1. 环境准备

安装并启动 Docker Desktop，确保 Docker 服务正常运行。

在项目根目录创建 `.env` 文件，填写自己的 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

请将示例值替换为自己的有效 API Key。不要将真实密钥提交到 GitHub。

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

1. 打开前端界面并上传一张图片。
2. 检查 YOLO 目标检测结果。
3. 检查视觉大模型分析结果。
4. 检查前后端日志是否存在错误。

### 5. 停止服务

```bash
docker compose down
```

说明：Docker 镜像使用 CPU 版 PyTorch，容器内推理默认使用 CPU。Windows 本地运行方式仍可使用原有 GPU 环境。