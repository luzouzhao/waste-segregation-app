# 部署到 Streamlit Community Cloud（免费、公开访问）

本文档把本项目发布到 **Streamlit Community Cloud**，部署完成后任何人打开链接即可使用，不需要你会写代码。

- 代码托管：GitHub（免费版要求仓库为 **Public**）
- 运行环境：Streamlit 官方免费服务器，Linux + CPU，内存约 2.7 GB
- 访问地址：`https://<你的应用名>.streamlit.app`

---

## 0. 已经为云端做好的改动

| 事项 | 说明 |
| --- | --- |
| `requirements.txt` | `torch` / `torchvision` 改为 PyTorch 官方 CPU 源（`+cpu`）；PyPI 的 Linux 版 torch 会连带下载 2.5 GB 以上的 CUDA 依赖，云端构建会超时或超磁盘 |
| `packages.txt` | 新增，让云端自动安装 `libgl1`、`libglib2.0-0t64`，否则 `import cv2` 会报 libGL 错误（注意：该文件只能一行一个包名，**不能写注释**） |
| `assets/fonts/NotoSansSC.ttf` | 内置中文字体（约 17 MB），云端生成的 PDF 报告中文不会乱码；同时保留了 Windows 字体作为回退 |
| `.gitignore` | 放行 3 个必需权重与 `reports/` 评估文件，其余大文件仍然忽略 |
| `src/utils/history_db.py` | 自动创建 `data/` 目录，避免全新克隆下来的仓库因缺目录而报错 |
| `src/detection/detect.py`、上传页 | 检测权重改为优先读项目内 `yolov8n.pt`，避免运行时联网下载 |

随仓库提交的模型文件：

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| `yolov8n.pt` | 6.5 MB | 目标检测（只用于画框） |
| `runs/classify/train-2/weights/best.pt` | 3.0 MB | 上线使用的 YOLOv8n-cls 分类模型 |
| `models/dl/best_model.pt` | 16.4 MB | 对照模型 EfficientNet-B0，用于模型对比页的 Grad-CAM |

---

## 1. 在 GitHub 上新建仓库

1. 打开 <https://github.com/new>
2. **Repository name** 填 `waste-segregation-app`
3. 可见性选 **Public**（免费部署必须公开）
4. **不要**勾选 “Add a README file”、“Add .gitignore”、“Choose a license”
5. 点击 **Create repository**

创建后会看到一个空仓库页面，记下形如
`https://github.com/你的用户名/waste-segregation-app.git` 的地址。

---

## 2. 把本地代码推上去

在项目根目录（`waste-segregation-app-main`）打开 PowerShell，把下面命令里的
`你的用户名` 换成自己的 GitHub 用户名，逐行执行：

```powershell
git remote add origin https://github.com/你的用户名/waste-segregation-app.git
git branch -M main
git push -u origin main
```

- 第一次 `push` 会弹出浏览器窗口要求登录 GitHub，按提示授权即可（登录一次以后不用再登）。
- 上传总量约 45 MB，网速慢时耐心等待。
- 如果提示 `remote origin already exists`，说明之前加过，改用：
  `git remote set-url origin https://github.com/你的用户名/waste-segregation-app.git`

推送完成后刷新 GitHub 页面，应该能看到 `app.py`、`pages/`、`assets/`、`models/`、`runs/` 等目录。

---

## 3. 在 Streamlit Cloud 部署

1. 打开 <https://share.streamlit.io>，点 **Continue with GitHub** 完成授权
2. 点右上角 **Create app**（或 **New app**）
3. 按下表填写：

| 字段 | 填写内容 |
| --- | --- |
| Repository | `你的用户名/waste-segregation-app` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL（可选） | 自定义子域名，如 `waste-segregation-demo` |

4. 展开 **Advanced settings**，把 **Python version** 选为 **3.12**
5. 点 **Deploy**

首次构建需要下载并安装 torch、ultralytics 等依赖，通常 **5～10 分钟**。
构建过程中页面会显示日志；看到 `You can now view your Streamlit app` 即部署成功，
把页面上的 `https://xxx.streamlit.app` 链接发给任何人都能打开。

---

## 4. 部署后是什么样

- 四个页面都可用：首页、上传与识别、模型对比、数据分析
- 推理在 CPU 上执行，单张图片约 1～3 秒（本地 GPU 环境会快很多）
- 应用 7×24 可访问；长时间无人访问会休眠，下一个人打开时自动唤醒（约 30 秒）

### 免费版的三条限制

1. **数据库是临时的**：识别记录写在云端的临时磁盘，应用重启或重新部署后会清空。
   如果只是想演示“每次识别 +1 条记录”，完全够用。
2. **内存约 2.7 GB**：当前模型很小（合计约 26 MB），够用；不要再往页面里加载更大的模型。
3. **仓库必须公开**：所有代码、权重、报告都会出现在 GitHub 上。

---

## 5. 常见问题排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 构建日志里 `No matching distribution found for torch==2.6.0+cpu` | Python 版本选错。到应用设置里把 Python 版本改为 3.12 后重新部署 |
| 构建日志出现大量 `nvidia-*` 下载 / 卡在 Installing dependencies | `requirements.txt` 顶部缺少 `--extra-index-url https://download.pytorch.org/whl/cpu`，会导致安装 2.5 GB 的 CUDA 依赖 |
| `ImportError: libGL.so.1: cannot open shared object file` | `packages.txt` 没生效。确认它和 `requirements.txt` 在同一层目录且已推送 |
| 页面报 `Classifier checkpoint not found` | `runs/classify/train-2/weights/best.pt` 没进仓库。本地执行 `git ls-files runs` 检查，补提交后重新部署 |
| 模型对比页空白 / 报找不到 npy | `reports/` 没进仓库，同样用 `git ls-files reports` 检查 |
| PDF 报告中文是方块或乱码 | `assets/fonts/NotoSansSC.ttf` 缺失，用 `git ls-files assets` 检查 |
| 应用一段时间不用后打不开 | 免费版休眠，重新访问会自动唤醒 |
| 数据分析里的历史记录变回 0 | 云端临时存储被重启清空，属正常现象 |

---

## 6. 以后更新代码

本地改完代码后：

```powershell
git add .
git commit -m "描述这次改了什么"
git push
```

Streamlit Cloud 检测到 GitHub 有新提交会自动重新部署，一般 1～2 分钟生效。
