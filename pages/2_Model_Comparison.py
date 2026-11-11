import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image as PILImage

from src.recommendation.rules import get_category_label

st.set_page_config(page_title="模型对比", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
YOLO_TRAIN_LOG = ROOT / "runs" / "classify" / "train-2" / "results.csv"

CLASS_NAMES = [
    "organic", "plastic", "paper", "glass",
    "metal", "electronic", "hazardous",
]


def _inject_css() -> None:
    st.markdown("""
<style>
    .stButton button { border-radius: 8px; font-weight: 600; transition: transform 0.2s ease, box-shadow 0.2s ease !important; }
    .stButton button:hover { transform: translateY(-2px) scale(1.02) !important; box-shadow: 0 6px 20px rgba(46,125,50,0.3) !important; }
    .stButton button:active { transform: translateY(0) scale(0.98) !important; }
    div[data-testid="stContainer"] { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 16px; }
    h1, h2, h3 { color: #2E7D32; }
    img { border-radius: 12px; max-height: 500px; object-fit: contain; width: 100%; }
    .stProgress > div > div > div > div { background-color: #2E7D32 !important; height: 12px; border-radius: 6px; }
    .stProgress > div > div { background-color: #E8F0E3; border-radius: 6px; }
    .stInfo, .stWarning { border-radius: 8px; }

    :root {
        --text-sm: clamp(0.8rem,1.5vw,0.9rem);
        --text-base: clamp(1rem,2vw,1.1rem);
        --text-lg: clamp(1.2rem,2.5vw,1.4rem);
        --text-xl: clamp(1.5rem,3vw,2rem);
        --text-2xl: clamp(2rem,5vw,3.5rem);
        --text-3xl: clamp(2.5rem,7vw,5rem);
    }

    input:focus, textarea:focus, select:focus { outline: none !important; box-shadow: 0 0 0 3px rgba(46,125,50,0.35) !important; transition: box-shadow 0.2s ease; }

    section[data-testid="stSidebar"] a { transition: all 0.2s ease; border-radius: 8px; }
    section[data-testid="stSidebar"] a:hover { background: rgba(46,125,50,0.1); padding-left: 8px; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def _load_confusion_matrix(filename: str) -> np.ndarray | None:
    path = REPORT_DIR / filename
    if not path.exists():
        return None
    return np.load(path)


@st.cache_data
def _load_yolo_train_log() -> pd.DataFrame | None:
    if not YOLO_TRAIN_LOG.exists():
        return None
    df = pd.read_csv(YOLO_TRAIN_LOG)
    df.columns = [c.strip() for c in df.columns]
    return df


def _cm_metrics(cm: np.ndarray) -> dict:
    """由混淆矩阵反算总体准确率与各类别精确率、召回率、F1。"""
    cm = cm.astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0
    )
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for name, p, r, f in zip(CLASS_NAMES, precision, recall, f1)
        },
    }


def _plot_curves(
    df: pd.DataFrame,
    series: list[tuple[str, str, str]],
    title: str,
    y_title: str,
    percent: bool = False,
) -> None:
    fig = go.Figure()
    for name, col, color in series:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col],
            mode="lines+markers", name=name,
            line=dict(color=color),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="轮次", yaxis_title=y_title,
        height=320, legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    if percent:
        fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def _style_accuracy(val) -> str:
    try:
        num = float(val)
        if num >= 0.85:
            return "color: #2E7D32; font-weight: bold"
        if num >= 0.7:
            return "color: #F9A825; font-weight: bold"
        if num >= 0.5:
            return "color: #C77B23;"
        return "color: #C62828;"
    except (ValueError, TypeError):
        return ""


def _style_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if numeric_cols:
        return df.style.map(_style_accuracy, subset=numeric_cols)
    return df


def _render_section(
    header_text: str,
    data: pd.DataFrame,
    caption: str = "",
    formats: dict[str, str] | None = None,
) -> None:
    if header_text:
        st.subheader(header_text)
    styled = _style_table(data)
    if formats:
        styled = styled.format(formats)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    if caption:
        st.caption(caption)


def main() -> None:
    _inject_css()
    with st.sidebar:
        st.markdown("### 导航")
        st.page_link("app.py", label="首页", use_container_width=True)
        st.page_link("pages/1_Upload_and_Classify.py", label="上传与识别", use_container_width=True)
        st.page_link("pages/2_Model_Comparison.py", label="模型对比", use_container_width=True)
        st.page_link("pages/3_Analytics.py", label="数据分析", use_container_width=True)
    st.title("模型对比")

    cm_yolo = _load_confusion_matrix("confusion_matrix_yolo.npy")
    cm_dl = _load_confusion_matrix("confusion_matrix_dl.npy")
    yolo_metrics = _cm_metrics(cm_yolo) if cm_yolo is not None else None
    dl_metrics = _cm_metrics(cm_dl) if cm_dl is not None else None

    st.header("深度学习模型对比")

    if yolo_metrics and dl_metrics:
        summary_rows = [
            {
                "模型": "YOLOv8n-cls（+4 视图 TTA）",
                "准确率": yolo_metrics["accuracy"],
                "各类 F1（宏平均）": yolo_metrics["macro_f1"],
                "推理耗时（毫秒/张）": 21.6,
            },
            {
                "模型": "EfficientNet-B0（迁移学习）",
                "准确率": dl_metrics["accuracy"],
                "各类 F1（宏平均）": dl_metrics["macro_f1"],
                "推理耗时（毫秒/张）": 2.21,
            },
        ]
        _render_section(
            "汇总",
            pd.DataFrame(summary_rows),
            caption="两个模型均在完整 test 集（3,479 张）上评测。"
            "准确率与 F1 由混淆矩阵换算，推理耗时为单张平均（YOLOv8n-cls 含 4 视图 TTA 的四次前向）。",
            formats={
                "准确率": "{:.1%}",
                "各类 F1（宏平均）": "{:.1%}",
                "推理耗时（毫秒/张）": "{:g}",
            },
        )

        # ── Training History ──────────────────────────────────────────────
        st.subheader("训练过程")
        tab_yolo, tab_dl = st.tabs(["YOLOv8n-cls", "EfficientNet-B0"])

        with tab_yolo:
            yolo_log = _load_yolo_train_log()
            if yolo_log is None:
                st.info("未找到 YOLOv8n-cls 训练日志（runs/classify/train-2/results.csv）。")
            else:
                col_l, col_a = st.columns(2)
                with col_l:
                    _plot_curves(
                        yolo_log,
                        [("训练损失", "train/loss", "#1E88E5"),
                         ("验证损失", "val/loss", "#E53935")],
                        "损失曲线",
                        "损失",
                    )
                with col_a:
                    _plot_curves(
                        yolo_log,
                        [("验证 Top-1 准确率", "metrics/accuracy_top1", "#43A047")],
                        "验证准确率曲线",
                        "准确率",
                        percent=True,
                    )
                st.caption("YOLOv8n-cls 的训练日志只记录验证集 Top-1 准确率，没有训练集准确率曲线。")

        with tab_dl:
            dl_log_path = REPORT_DIR / "dl_training_log.csv"
            if not dl_log_path.exists():
                st.info("未找到 EfficientNet-B0 训练日志。")
            else:
                log_df = pd.read_csv(dl_log_path)
                col_l, col_a = st.columns(2)
                with col_l:
                    _plot_curves(
                        log_df,
                        [("训练损失", "train_loss", "#1E88E5"),
                         ("验证损失", "val_loss", "#E53935")],
                        "损失曲线",
                        "损失",
                    )
                with col_a:
                    _plot_curves(
                        log_df,
                        [("训练准确率", "train_acc", "#1E88E5"),
                         ("验证准确率", "val_acc", "#E53935")],
                        "准确率曲线",
                        "准确率",
                        percent=True,
                    )

        # ── Per-class F1 ──────────────────────────────────────────────────
        st.subheader("各类别 F1 明细")
        per_class_rows = []
        for model_name, metrics in (
            ("YOLOv8n-cls", yolo_metrics),
            ("EfficientNet-B0", dl_metrics),
        ):
            for class_name, values in metrics["per_class"].items():
                per_class_rows.append({
                    "模型": model_name,
                    "类别": get_category_label(class_name),
                    "F1 分数": values["f1"],
                })
        pc_df = pd.DataFrame(per_class_rows)
        fig = px.bar(
            pc_df, x="类别", y="F1 分数", color="模型",
            barmode="group", text_auto=".1%",
            title="各类别 F1 分数对比",
            color_discrete_sequence=["#1E88E5", "#43A047"],
        )
        fig.update_layout(
            yaxis_tickformat=".0%", height=400,
            legend_title_text="", xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_y, col_d = st.columns(2)
        for column, title, metrics in (
            (col_y, "YOLOv8n-cls", yolo_metrics),
            (col_d, "EfficientNet-B0", dl_metrics),
        ):
            with column:
                st.markdown(f"**{title}**")
                detail = pd.DataFrame([
                    {
                        "类别": get_category_label(name),
                        "精确率": values["precision"],
                        "召回率": values["recall"],
                        "F1 分数": values["f1"],
                    }
                    for name, values in metrics["per_class"].items()
                ])
                st.dataframe(
                    detail.style.format({
                        "精确率": "{:.1%}",
                        "召回率": "{:.1%}",
                        "F1 分数": "{:.1%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.error("未找到混淆矩阵文件，请先运行 `python src/evaluation/confusion_matrix_viz.py`。")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.divider()
    st.header("混淆矩阵")
    st.caption(
        "行为真实标签，列为预测标签。对角线为预测正确的样本。"
        "单元格显示原始数量与行内召回率。左侧为实际部署的 YOLOv8n-cls 管线，"
        "右侧为对照模型 EfficientNet-B0，两者均在完整 test 集上评测。"
    )

    CLASS_LABELS = [get_category_label(c) for c in CLASS_NAMES]

    def _plotly_cm(cm: np.ndarray, title: str):
        n = len(CLASS_NAMES)
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        annotations = [
            [f"{int(cm[i,j])}<br>({cm_norm[i,j]:.0%})" for j in range(n)]
            for i in range(n)
        ]

        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=CLASS_LABELS,
            y=CLASS_LABELS,
            colorscale="Greens",
            zmin=0, zmax=1,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="真实: %{y}<br>预测: %{x}<br>占比: %{z:.0%}<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="预测标签",
            yaxis_title="真实标签",
            xaxis=dict(tickangle=40),
            yaxis=dict(tickangle=0),
            height=440,
            margin=dict(l=80, r=40, t=60, b=120),
        )
        return fig

    cm_dl_png = REPORT_DIR / "confusion_matrix_dl.png"
    cm_yolo_png = REPORT_DIR / "confusion_matrix_yolo.png"

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("**上线模型 — YOLOv8n-cls + 4 视图 TTA**")
        if cm_yolo is not None:
            st.plotly_chart(
                _plotly_cm(cm_yolo, f"YOLOv8n-cls · 准确率 {yolo_metrics['accuracy']:.1%}"),
                use_container_width=True,
            )
        elif cm_yolo_png.exists():
            st.image(str(cm_yolo_png), use_container_width=True)
        else:
            st.info("运行 `python src/evaluation/confusion_matrix_viz.py` 生成。")

    with col_cm2:
        st.markdown("**对照模型 — EfficientNet-B0（迁移学习）**")
        if cm_dl is not None:
            st.plotly_chart(
                _plotly_cm(cm_dl, f"EfficientNet-B0 · 准确率 {dl_metrics['accuracy']:.1%}"),
                use_container_width=True,
            )
        elif cm_dl_png.exists():
            st.image(str(cm_dl_png), use_container_width=True)
        else:
            st.info("运行 `python src/evaluation/confusion_matrix_viz.py` 生成。")

    # ── XAI Section ───────────────────────────────────────────────────────────
    st.divider()
    st.header("可解释性分析（Grad-CAM）")

    st.markdown(
        "**Grad-CAM** 展示模型做预测时「关注」了图像的哪些区域。"
        "颜色越暖（红/黄）表示激活程度越高。"
    )
    st.caption(
        "说明：Grad-CAM 需要反向传播，本页基于对照模型 EfficientNet-B0 实现；"
        "上线模型 YOLOv8n-cls 的分类结果请以「上传与识别」页为准，两者判断可能不同。"
    )

    xai_uploaded = st.file_uploader(
        "上传一张垃圾图片，生成 Grad-CAM 解释",
        type=["jpg", "jpeg", "png"],
        key="xai_upload",
    )
    if xai_uploaded:
        with tempfile.NamedTemporaryFile(suffix=Path(xai_uploaded.name).suffix, delete=False) as tmp:
            tmp.write(xai_uploaded.read())
            tmp_path = Path(tmp.name)

        with st.spinner("正在生成 Grad-CAM…"):
            try:
                from src.evaluation.xai import gradcam_to_image
                overlay_rgb, pred_label = gradcam_to_image(tmp_path)

                orig_img = PILImage.open(tmp_path).convert("RGB")
                orig_arr = np.array(orig_img)

                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.image(orig_arr, caption="原图", use_container_width=True)
                with gcol2:
                    st.image(
                        overlay_rgb,
                        caption=f"Grad-CAM 叠加图（EfficientNet-B0 预测：{get_category_label(pred_label.lower())}）",
                        use_container_width=True,
                    )

                st.success(
                    f"EfficientNet-B0 预测：**{get_category_label(pred_label.lower())}**"
                    " — 红色/黄色区域对该分类结果贡献最大。"
                )
            except Exception as exc:
                st.error(f"Grad-CAM 生成失败：{exc}")
    else:
        existing = sorted(REPORT_DIR.glob("gradcam_*.png"))
        if existing:
            st.markdown("**预生成的 Grad-CAM 示例：**")
            cols = st.columns(min(len(existing), 3))
            for idx, img_p in enumerate(existing[:6]):
                with cols[idx % 3]:
                    st.image(
                        str(img_p),
                        caption=get_category_label(
                            img_p.stem.replace("gradcam_", "").lower()
                        ),
                        use_container_width=True,
                    )
        else:
            st.info("在上方上传图片即可查看实时 Grad-CAM，或运行 `python src/evaluation/xai.py` 生成示例。")


if __name__ == "__main__":
    main()
