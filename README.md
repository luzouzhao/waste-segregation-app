# AI 智能垃圾分类识别系统

基于 YOLOv8 与迁移学习的垃圾分类识别 Web 应用。上传一张垃圾图片，系统返回所属类别、可回收判定、处置建议与置信度，并提供模型评估、混淆矩阵、Grad-CAM 可解释性与识别历史统计。

## 支持的类别

| 类别 | 显示名 | 回收属性 |
|---|---|---|
| organic | 厨余垃圾 | 不可回收 |
| plastic | 塑料 | 可回收 |
| paper | 纸类 | 可回收 |
| glass | 玻璃 | 可回收 |
| metal | 金属 | 可回收 |
| electronic | 电子垃圾 | 可回收（需专门处理） |
| hazardous | 有害垃圾 | 不可回收（有害） |

## 功能页面

| 页面 | 说明 |
|---|---|
| 首页 `app.py` | 项目介绍与功能入口 |
| 上传与识别 `pages/1_Upload_and_Classify.py` | 上传图片 → 检测框标注 → 7 类分类 → Top-3 置信度 → 处置建议 → 导出 PDF 报告 |
| 模型对比 `pages/2_Model_Comparison.py` | 训练曲线、各类别 F1、双模型混淆矩阵并排对比、Grad-CAM 可解释性 |
| 数据分析 `pages/3_Analytics.py` | 识别历史统计、类别分布、可回收占比、置信度分布、CSV 导出 |

## 快速开始

```powershell
# 1. 创建环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt

# 若需要 GPU 版 PyTorch（requirements.txt 中的 +cu124 版本需指定官方源）
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 3. 启动（模型权重已随仓库提供）
streamlit run app.py
```

浏览器打开 http://localhost:8501 。端口冲突时加 `--server.port 8502`。

## 技术栈

| 层次 | 技术 |
|---|---|
| 界面 | Streamlit（多页应用） |
| 分类（上线） | ultralytics YOLOv8n-cls + 四视图 TTA |
| 分类（对照） | PyTorch + torchvision EfficientNet-B0 |
| 检测（仅可视化） | YOLOv8n（COCO 预训练） |
| 图像处理 | OpenCV、Pillow、NumPy |
| 可解释性 | 手写 Grad-CAM（无第三方依赖） |
| 报告导出 | fpdf2（中文字体嵌入） |
| 历史记录 | SQLite |

## 目录结构

```
waste-segregation-app/
├── app.py                         # 首页
├── pages/
│   ├── 1_Upload_and_Classify.py   # 上传与识别
│   ├── 2_Model_Comparison.py      # 模型对比与可解释性
│   └── 3_Analytics.py             # 数据分析
├── config/settings.py             # 路径、类别、超参数
├── src/
│   ├── preprocessing/             # 数据集下载、预处理、校验
│   ├── detection/                 # 检测与推理管线（pipeline.py 为核心）
│   ├── classification/            # 训练、评估、模型对比
│   ├── evaluation/                # 混淆矩阵、Grad-CAM
│   ├── recommendation/rules.py    # 回收规则与处置建议
│   └── utils/history_db.py        # SQLite 历史记录
├── data/
│   ├── raw/                       # 原始数据（7 类）
│   ├── processed/train|test/      # 预处理后训练集 / 测试集
│   └── waste_metadata.db          # 识别历史
├── models/dl/best_model.pt        # EfficientNet-B0 权重
├── runs/classify/train-2/weights/best.pt   # YOLOv8n-cls 权重（上线模型）
├── reports/                       # 评估报告、混淆矩阵、训练日志
└── requirements.txt
```

## 模型与实测指标

全部在完整测试集（3,479 张）上评测：

| 方案 | 准确率 | 单张耗时 | 说明 |
|---|---|---|---|
| YOLOv8n-cls 整图 + 4 视图 TTA | **94.62%** | 21.6 ms | 上线方案 |
| EfficientNet-B0（迁移学习） | 86.49% | 2.21 ms | 对照模型，macro-F1 0.8587 |
| 端到端（含检测框） | 93.60% | 62 ms | 抽样 140 张 |

上线模型分类别召回率：电子 99.3%、玻璃 97.7%、厨余 96.8%、纸类 95.0%、金属 94.8%、塑料 90.7%、有害 90.3%。

## 复现流程

```powershell
# 数据下载（需 Kaggle 凭据，放在 ~/.kaggle/kaggle.json）
python src/preprocessing/download_datasets.py

# 预处理：缩放 + 增强 + 分层划分
python src/preprocessing/preprocess.py

# 训练上线模型（YOLOv8n-cls，15 轮，256 输入）
yolo classify train model=yolov8n-cls.pt data=data/processed epochs=15 imgsz=256 batch=32 device=0

# 训练对照模型（EfficientNet-B0）
python src/classification/dl_train.py

# 评估与报告
python src/classification/dl_evaluate.py
python src/classification/dl_compare.py
python src/evaluation/confusion_matrix_viz.py
python src/evaluation/xai.py
```

## 设计要点

- **检测与分类解耦**：通用 COCO 检测器只负责画框定位，分类器始终看完整原图。实测若先按检测框裁剪再分类，准确率会从约 95% 掉到 53%（玻璃最容易被裁成电子垃圾）。
- **四视图 TTA**：原图 + 水平翻转 + 1.15× / 1.3× 中心缩放，四个视图概率平均。零训练成本换约 8 个百分点准确率。
- **通道一致性**：ultralytics 的 numpy 输入按 BGR 解析，若先转 RGB 再送模型，准确率会再掉约 11 个百分点。

## 已知局限

- 模型置信度高度饱和（平均 97.6%），「低置信 → 人工复核」机制实际很少触发，校准不足。
- 弱项为塑料（90.7%）与有害垃圾（90.3%），二者与纸类、金属混淆最多。
- 数据集以白底摆拍为主，真实复杂背景下准确率约 90.5%。
- 数据分析页记录的是本机识别历史，换机器或清库后为空。
- Grad-CAM 基于对照模型 EfficientNet-B0 实现（需反向传播），与上线模型的判断可能不一致。
