import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="垃圾分类智能识别",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

with st.sidebar:
    st.markdown("### 导航")
    st.page_link("app.py", label="首页", use_container_width=True)
    st.page_link("pages/1_Upload_and_Classify.py", label="上传与识别", use_container_width=True)
    st.page_link("pages/2_Model_Comparison.py", label="模型对比", use_container_width=True)
    st.page_link("pages/3_Analytics.py", label="数据分析", use_container_width=True)

st.markdown(
    "<h1 style='text-align: center; color: #2E7D32;'>AI 智能垃圾分类</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #6B6B6B; font-size: 18px;'>"
    "上传一张垃圾图片，即可获得即时分类结果和处置建议。"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 上传与识别")
        st.markdown(
            "上传一张垃圾物品的照片，AI 会自动检测并把它归入 7 个垃圾类别之一，"
            "同时给出处置方法说明。"
        )
        if st.button("前往识别", type="primary", use_container_width=True):
            st.switch_page("pages/1_Upload_and_Classify.py")

with col2:
    with st.container(border=True):
        st.markdown("### 模型对比")
        st.markdown(
            "对比上线模型 YOLOv8n-cls 与对照模型 EfficientNet-B0 在完整测试集上的表现，"
            "查看准确率、各类别 F1、混淆矩阵与推理速度。"
        )
        if st.button("查看对比", type="secondary", use_container_width=True):
            st.switch_page("pages/2_Model_Comparison.py")

with col3:
    with st.container(border=True):
        st.markdown("### 数据分析")
        st.markdown(
            "追踪你的垃圾处置历史，查看类别分布以及可回收与不可回收垃圾的占比变化，"
            "一眼掌握自己的分类习惯和环保贡献。"
        )
        if st.button("查看数据", type="secondary", use_container_width=True):
            st.switch_page("pages/3_Analytics.py")

st.divider()

st.caption(
    "本系统为演示用途，实际处置请以当地市政环卫部门的规定为准。"
)
