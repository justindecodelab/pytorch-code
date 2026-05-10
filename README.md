# Dive into CV — PyTorch 示例代码 / PyTorch Code Samples

[中文](#中文) · [English](#english)

---

## 中文

### 项目简介

本仓库收录配套《动手学计算机视觉（PyTorch 版）》学习路线的示例程序，覆盖 **PyTorch 基础**、**图像分类** 与 **目标检测入门**。代码按章节组织，便于按顺序对照教材运行与改写实验。

### 目录结构

```
pytorch_code/
├── chapter1/                 # 第 1 章：PyTorch 与深度学习基础
│   ├── 1_tensor.py           # Tensor 创建与基本属性
│   ├── 2_operations.py       # 张量运算、变形、拼接等
│   ├── 3_numpy.py            # Tensor 与 NumPy 互转、共享内存
│   ├── 4_autograd.py         # 自动求导与简单训练流程
│   ├── 5_linear_regression.py
│   └── 6_FC_MNIST_Classification.py
│   └── Readme.md             # 各脚本函数级说明（中文）
├── chapter2/                 # 第 2 章：图像分类
│   ├── README.md             # 本章总览（数据、经典 CNN、实战、调参）
│   ├── section_1/            # 数据读取、自定义 Dataset、ImageFolder、增强
│   ├── section_2/            # 经典 CNN：LeNet / AlexNet / VGG / GoogLeNet / ResNet / NiN（多为 CIFAR-10）
│   ├── section_3/            # CIFAR-10 完整分类流程（SimpleCNN）
│   ├── section_4/            # SVHN 多字符分类基线（预训练 ResNet18 + 多头）
│   └── dataset/              # 部分示例标注与样例（如 SVHN、tianchi）
└── chapter3/
    └── tiny_detector_demo/   # 小型检测 demo（Pascal VOC、VGG+SSD 风格）
```

### 环境依赖

运行前请安装 **Python 3** 及常用科学计算与深度学习库，例如：

- `torch`、`torchvision`（版本需互相兼容）
- `numpy`、`matplotlib`
- 部分脚本另需：`pandas`、`Pillow` 等

各章节脚本可能在首次运行时 **自动下载** MNIST、CIFAR-10、SVHN 等数据；检测章节需自备 **Pascal VOC** 数据目录（见 `chapter3/tiny_detector_demo/train.py` 中的 `data_folder` 配置）。

建议使用虚拟环境，且 **不要将本地 `venv` 目录提交到 Git**（可在仓库根目录使用 `.gitignore` 忽略）。

### 如何运行

- **第 1 章**：在 `chapter1/` 下直接运行对应 `python *.py` 脚本；顺序建议见 `chapter1/Readme.md`。
- **第 2 章**：按 `section_1` → `section_2` 各子目录中的入口脚本运行；`section_3` 的 `classifier.py`、`section_4` 的 `baseline.py` 为完整训练示例。详细说明见 `chapter2/README.md` 与各 `readme.md`。
- **第 3 章**：进入 `chapter3/tiny_detector_demo/`，按项目内 `train.py`、`eval.py`、`detect.py` 的说明准备 VOC 数据后执行。

> 若遇 `pretrained` / `weights` 等 API 与当前 `torchvision` 版本不一致，请按官方文档将加载预训练权重的写法改为本环境支持的参数（例如 `weights=...`）。

### 许可证与致谢

若本仓库为课程或开源书的配套代码，使用与分发时请遵循原书/原项目的版权与许可要求；对外分享时建议保留原作者与教材出处说明。

---

## English

### Overview

This repository contains **PyTorch sample code** for a hands-on computer vision learning path. It covers **PyTorch fundamentals**, **image classification**, and an **introductory object-detection demo**. Scripts are grouped by chapter to match a typical textbook or tutorial progression.

### Repository layout

- **`chapter1/`** — Tensors, basic ops, NumPy interchange, autograd, a tiny linear regression demo, and a **fully connected MNIST classifier**. See `chapter1/Readme.md` for function-level notes (Chinese).
- **`chapter2/`** — **Image classification**: loading & augmenting data, classic CNNs (**LeNet**, **AlexNet**, **VGG**, **GoogLeNet**, **ResNet**, **NiN** on CIFAR-10), an end-to-end **CIFAR-10** exercise, and an **SVHN multi-digit** baseline (shared backbone + multiple classification heads). Chapter overview: `chapter2/README.md`.
- **`chapter3/tiny_detector_demo/`** — A compact **detection** pipeline: **Pascal VOC**-style data, VGG-style backbone, SSD-like training with **MultiBox** loss (`train.py`, `eval.py`, `detect.py`).

### Requirements

Install **Python 3** and PyTorch stack libraries, e.g.:

- `torch`, `torchvision` (compatible versions)
- `numpy`, `matplotlib`
- Some scripts also need `pandas`, `Pillow`, etc.

Datasets such as MNIST / CIFAR-10 / SVHN may be **downloaded automatically** on first run. The detection demo expects a local **VOCdevkit** tree; set `data_folder` in `chapter3/tiny_detector_demo/train.py` accordingly.

Use a **virtual environment** and avoid committing local `venv` folders to Git.

### How to run

Run scripts from their directories with `python <script>.py`. Follow the suggested order in `chapter1/Readme.md` and `chapter2/README.md`. For detection, prepare VOC data, then run training/evaluation scripts under `chapter3/tiny_detector_demo/`.

> If `torchvision.models.*(pretrained=...)` fails on your version, update calls to the current API (e.g. `weights=ResNet18_Weights.DEFAULT` or `weights=None`).

### License & credits

Respect the license of the original course/book/project if this code is derived from official materials; keep attribution when redistributing.

---

**推荐阅读顺序 / Suggested order:** `chapter1` → `chapter2` (section 1 → 2 → 3 → 4) → `chapter3/tiny_detector_demo`.
