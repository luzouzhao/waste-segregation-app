import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
from fpdf import FPDF
from PIL import Image
from ultralytics import YOLO

from src.detection.pipeline import classify_waste_image
from src.recommendation.rules import (
    HAZARD_LABELS,
    get_category_label,
    get_guidance,
)
from src.utils.history_db import log_disposal

st.set_page_config(page_title="上传与识别", layout="wide")

CATEGORY_COLORS: dict[str, str] = {
    "organic": "#6D4C29",
    "plastic": "#1976D2",
    "paper": "#8D6E63",
    "glass": "#00897B",
    "metal": "#78909C",
    "electronic": "#5E35B1",
    "hazardous": "#C62828",
}


@st.cache_resource
def _load_models():
    root = Path(__file__).resolve().parent.parent
    det_path = root / "yolov8n.pt"
    det = YOLO(str(det_path) if det_path.exists() else "yolov8n.pt")
    ckpt = root / "runs" / "classify" / "train-2" / "weights" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found at {ckpt}")
    cls = YOLO(str(ckpt))
    return det, cls


def _inject_css() -> None:
    st.markdown("""
<style>
    .stButton button, .stDownloadButton button { border-radius: 8px; font-weight: 600; transition: transform 0.2s ease, box-shadow 0.2s ease !important; }
    .stButton button:hover, .stDownloadButton button:hover { transform: translateY(-2px) scale(1.02) !important; box-shadow: 0 6px 20px rgba(46,125,50,0.3) !important; }
    .stButton button:active, .stDownloadButton button:active { transform: translateY(0) scale(0.98) !important; }
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

    @keyframes fill-gauge { from { stroke-dashoffset: 339.292; } }
    .gauge-container { display: flex; flex-direction: column; align-items: center; margin: 12px 0; animation: fade-in-up 0.6s ease forwards; }
    @keyframes fade-in-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)


def _category_badge_html(category: str) -> str:
    color = CATEGORY_COLORS.get(category, "#6B6B6B")
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:4px 14px;border-radius:12px;font-weight:bold;"
        f"font-size:14px;'>{get_category_label(category)}</span>"
    )


def _draw_bbox(pil_img: Image.Image, bbox: list[int]) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def _result_card(label: str, value: str, color: str = "#2E7D32") -> str:
    return (
        f"<div style='background:#E8F0E3;border-radius:8px;padding:8px 12px;"
        f"margin:4px 0;'><span style='color:#6B6B6B;font-size:12px;'>"
        f"{label}</span><br><span style='color:{color};font-weight:bold;"
        f"font-size:16px;'>{value}</span></div>"
    )


def _sanitize(text: str) -> str:
    return (
        text.replace("\u2014", "--")
            .replace("\u2013", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2026", "...")
            .replace("\u2022", "-")
            .replace("\u00a0", " ")
    )


_ASSETS_FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"

_CJK_FONT_CANDIDATES = [
    _ASSETS_FONTS / "NotoSansSC.ttf",   # 随项目打包，本地与云端通用
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
    Path(r"C:\Windows\Fonts\simkai.ttf"),
]


def _register_cjk_font(pdf: FPDF) -> str:
    """注册中文字体，找不到则退回内置 Helvetica。"""
    for path in _CJK_FONT_CANDIDATES:
        if path.exists():
            pdf.add_font("CJK", "", str(path))
            pdf.add_font("CJK", "B", str(path))
            pdf.add_font("CJK", "I", str(path))
            return "CJK"
    return "Helvetica"


def _pdf_report(
    category: str, confidence: float, top3: list[tuple[str, float]],
    recyclable: bool, hazard: str, disposal: str, image_filename: str,
) -> bytes:
    pdf = FPDF()
    font = _register_cjk_font(pdf)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.set_font(font, "B", 18)
    pdf.set_text_color(46, 125, 50)
    pdf.cell(0, 14, "垃圾分类识别报告", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font(font, "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  |  图片：{image_filename}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_draw_color(46, 125, 50)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font(font, "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "识别结果", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font(font, "", 11)
    category_colors = {
        "organic": (109, 76, 41), "plastic": (25, 118, 210), "paper": (141, 110, 99),
        "glass": (0, 137, 123), "metal": (120, 144, 156), "electronic": (94, 53, 177),
        "hazardous": (198, 40, 40),
    }
    cat_color = category_colors.get(category, (107, 107, 107))
    pdf.set_text_color(*cat_color)
    pdf.set_font(font, "B", 14)
    pdf.cell(0, 7, f"类别：{get_category_label(category)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font, "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, f"置信度：{confidence:.1%}", new_x="LMARGIN", new_y="NEXT")
    if top3:
        top3_str = " | ".join(
            f"{get_category_label(c)} ({p:.0%})" for c, p in top3
        )
        pdf.cell(0, 6, f"Top-3 预测：{top3_str}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font(font, "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "可回收性与危害等级", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(font, "", 11)
    pdf.set_fill_color(232, 240, 227)
    rc_text = "可回收" if recyclable else "不可回收"
    rc_color = (46, 125, 50) if recyclable else (198, 40, 40)
    pdf.set_text_color(*rc_color)
    pdf.cell(0, 6, f"是否可回收：{rc_text}", new_x="LMARGIN", new_y="NEXT")
    hz_colors = {"low": (46, 125, 50), "moderate": (249, 168, 37), "high": (198, 40, 40)}
    pdf.set_text_color(*hz_colors.get(hazard, (100, 100, 100)))
    pdf.cell(0, 6, f"危害等级：{HAZARD_LABELS.get(hazard, hazard)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_text_color(30, 30, 30)
    pdf.set_font(font, "B", 13)
    pdf.cell(0, 8, "处置建议", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(font, "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, _sanitize(disposal))
    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font(font, "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "本报告由 AI 生成，实际处置请以当地市政环卫部门的规定为准。", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def main() -> None:
    _inject_css()
    with st.sidebar:
        st.markdown("### 导航")
        st.page_link("app.py", label="首页", use_container_width=True)
        st.page_link("pages/1_Upload_and_Classify.py", label="上传与识别", use_container_width=True)
        st.page_link("pages/2_Model_Comparison.py", label="模型对比", use_container_width=True)
        st.page_link("pages/3_Analytics.py", label="数据分析", use_container_width=True)
    st.title("上传与识别")
    st.markdown("拍摄或上传一张垃圾物品的照片，即可开始识别。")

    for key in ("camera_on", "camera_capture"):
        if key not in st.session_state:
            st.session_state[key] = False if key == "camera_on" else None

    uploaded = st.file_uploader(
        "将图片拖放到此处，或点击浏览",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        st.session_state.camera_on = False

    camera_active = st.session_state.camera_on
    if st.button("打开摄像头", type="secondary", use_container_width=True):
        camera_active = True

    camera = None
    if camera_active:
        st.session_state.camera_on = True
        cam = st.camera_input("拍照")
        if cam is not None:
            st.session_state.camera_capture = cam
            st.session_state.camera_on = False

    if not camera and st.session_state.camera_capture is not None:
        camera = st.session_state.camera_capture

    image_source = uploaded or camera
    if image_source is None:
        if st.session_state.camera_on:
            return
        st.info("请上传或拍摄一张图片开始识别。")
        return

    img = Image.open(image_source)
    st.image(img, caption="输入图片", use_container_width=True)

    if not st.button("开始识别", type="primary"):
        return

    with st.spinner("正在分析垃圾物品..."):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp, format="PNG")
                tmp_path = tmp.name

            det_model, cls_model = _load_models()
            result = classify_waste_image(tmp_path, classifier=cls_model, det_model=det_model)
            Path(tmp_path).unlink(missing_ok=True)
            st.toast("识别完成！", icon=None)
        except FileNotFoundError as e:
            st.error(f"模型文件缺失：{e}。请先训练分类模型。")
            return
        except Exception as e:
            st.error(f"识别失败：{e}")
            return

    category = result["category"]
    confidence = result["confidence"]
    top3 = result.get("top3", [])
    bbox = result["bbox"]

    try:
        guidance = get_guidance(category)
    except ValueError:
        st.error(f"未知类别「{category}」。")
        return

    recyclable = guidance["recyclable"]
    hazard = guidance["hazard_level"]
    disposal = guidance["disposal_method"]
    recyclable_text = "可回收" if recyclable else "不可回收"

    result["recyclable"] = recyclable
    result["hazard_level"] = hazard
    result["image_filename"] = getattr(image_source, "name", "capture.png")

    try:
        log_disposal(result)
    except Exception as exc:
        st.warning(f"识别记录写入失败：{exc}")

    st.divider()
    st.markdown("### 识别结果")

    col_img, col_res = st.columns([1.2, 1])

    with col_img:
        if bbox is not None:
            img_with_box = _draw_bbox(img, bbox)
            st.image(img_with_box, caption="检测到的物体", use_container_width=True)
        else:
            st.image(img, caption="输入图片", use_container_width=True)

    with col_res:
        with st.container(border=True):
            st.markdown("**预测类别**")

            UNCERTAIN_THRESHOLD = 0.6
            if confidence < UNCERTAIN_THRESHOLD:
                st.warning(
                    f"置信度较低（{confidence:.1%}），"
                    "结果可能不准确，请人工核实。"
                )
                st.markdown("**类别：** 不确定")
            else:
                badge = _category_badge_html(category)
                st.markdown(
                    f"**类别：** {badge}",
                    unsafe_allow_html=True,
                )

            conf_pct = float(confidence)
            circumference = 339.292
            offset = circumference * (1 - conf_pct)
            st.markdown(
                f'<div class="gauge-container">'
                f'<svg width="120" height="120" viewBox="0 0 120 120">'
                f'<circle cx="60" cy="60" r="54" fill="none" stroke="#E8F0E3" stroke-width="8"/>'
                f'<circle cx="60" cy="60" r="54" fill="none" stroke="#2E7D32" stroke-width="8" '
                f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" '
                f'stroke-linecap="round" transform="rotate(-90 60 60)" '
                f'style="transition: stroke-dashoffset 1s ease-out;"/>'
                f'<text x="60" y="52" text-anchor="middle" dominant-baseline="central" '
                f'font-size="26" font-weight="bold" fill="#2E7D32">{conf_pct:.0%}</text>'
                f'<text x="60" y="74" text-anchor="middle" dominant-baseline="central" '
                f'font-size="10" fill="#8B8B8B">置信度</text>'
                f'</svg></div>',
                unsafe_allow_html=True,
            )

            if top3:
                top3_lines = " | ".join(
                    f"{get_category_label(c)} ({p:.0%})" for c, p in top3
                )
                st.caption(f"Top-3 预测：{top3_lines}")

            if confidence >= UNCERTAIN_THRESHOLD:
                rc_color = "#2E7D32" if recyclable else "#C62828"
                hz_color = {"low": "#2E7D32", "moderate": "#F9A825", "high": "#C62828"}.get(hazard, "#6B6B6B")
                st.markdown(
                    _result_card("是否可回收", recyclable_text, rc_color)
                    + _result_card("危害等级", HAZARD_LABELS.get(hazard, hazard), hz_color),
                    unsafe_allow_html=True,
                )

    with st.container(border=True):
        st.markdown("#### 处置建议")
        st.markdown(disposal)

    if hazard in ("high", "moderate"):
        st.warning(
            "投放前请先确认当地市政的垃圾处置规定。"
        )

    pdf_bytes = _pdf_report(
        category, confidence, top3,
        recyclable, hazard, disposal,
        getattr(image_source, "name", "capture.png"),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("再识别一张", type="secondary", use_container_width=True):
            for key in ("camera_on", "camera_capture"):
                st.session_state[key] = False if key == "camera_on" else None
    with col_b:
        st.download_button(
            label="下载 PDF 报告",
            data=pdf_bytes,
            file_name=f"recycling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            type="secondary",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
