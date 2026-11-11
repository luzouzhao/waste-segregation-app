import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.history_db import clear_history, get_history
from src.recommendation.rules import get_category_label

st.set_page_config(page_title="数据分析", layout="wide")

CATEGORY_COLORS: dict[str, str] = {
    "organic": "#6D4C29",
    "plastic": "#1976D2",
    "paper": "#8D6E63",
    "glass": "#00897B",
    "metal": "#78909C",
    "electronic": "#5E35B1",
    "hazardous": "#C62828",
}


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


def _plot_category_distribution(series: pd.Series):
    counts = series.value_counts().reset_index()
    counts.columns = ["category", "count"]
    counts = counts[counts["category"].isin(CATEGORY_COLORS)]
    counts = counts[counts["count"] > 0]
    counts["label"] = counts["category"].apply(get_category_label)
    fig = px.bar(
        counts, x="label", y="count", text="count",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        title="垃圾类别分布",
        labels={"label": "", "count": "数量"},
    )
    fig.update_traces(textposition="outside", texttemplate="%{text}")
    fig.update_layout(
        showlegend=False,
        title_font_size=16,
        height=400,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def _plot_recyclable_pie(series: pd.Series):
    total = len(series)
    recyclable = int(series.astype(bool).sum())
    non_recyclable = total - recyclable
    df = pd.DataFrame({
        "label": ["可回收", "不可回收"],
        "count": [recyclable, non_recyclable],
    })
    fig = px.pie(
        df, values="count", names="label",
        title="可回收与不可回收",
    )
    colors = ["#2E7D32", "#C62828"]
    fig.update_traces(textinfo="label+percent", marker=dict(colors=colors))
    fig.update_layout(
        title_font_size=16,
        height=400,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def _plot_confidence_hist(series: pd.Series):
    data = pd.DataFrame({"confidence": series.astype(float)})
    mean = float(data["confidence"].mean())
    fig = px.histogram(
        data, x="confidence", nbins=20,
        title="置信度分布", labels={"confidence": "置信度"},
    )
    fig.update_traces(marker_color="#2E7D32")
    fig.add_vline(
        x=mean, line_dash="dash", line_color="#F9A825",
        annotation_text=f"均值 {mean:.1%}", annotation_position="top",
    )
    fig.update_layout(
        showlegend=False,
        title_font_size=16,
        height=400,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis_range=[0, 1],
        xaxis_tickformat=".0%",
        bargap=0.05,
    )
    return fig


def _plot_confidence_by_category(df: pd.DataFrame):
    stats = (
        df.groupby("category")["confidence"]
        .agg(["mean", "count"])
        .reset_index()
    )
    stats = stats[stats["category"].isin(CATEGORY_COLORS)]
    stats["label"] = stats["category"].apply(get_category_label)
    stats = stats.sort_values("mean")
    fig = px.bar(
        stats, x="mean", y="label", orientation="h",
        text=stats["mean"].map(lambda v: f"{v:.1%}"),
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        title="各类别平均置信度",
        labels={"mean": "平均置信度", "label": ""},
        hover_data={"count": True, "mean": ":.3f", "category": False},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        title_font_size=16,
        height=400,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis_range=[0, 1.1],
        xaxis_tickformat=".0%",
    )
    return fig


def _format_confidence(val) -> str:
    if pd.notna(val):
        try:
            return f"{float(val):.1%}"
        except (ValueError, TypeError):
            return str(val)
    return ""


def main() -> None:
    _inject_css()
    with st.sidebar:
        st.markdown("### 导航")
        st.page_link("app.py", label="首页", use_container_width=True)
        st.page_link("pages/1_Upload_and_Classify.py", label="上传与识别", use_container_width=True)
        st.page_link("pages/2_Model_Comparison.py", label="模型对比", use_container_width=True)
        st.page_link("pages/3_Analytics.py", label="数据分析", use_container_width=True)
    col_title, col_action = st.columns([0.72, 0.28], vertical_alignment="center")
    with col_title:
        st.title("数据分析")
    with col_action:
        if st.button("清除历史数据", use_container_width=True):
            st.session_state["confirm_clear"] = True

    flash = st.session_state.pop("clear_flash", None)
    if flash:
        st.success(flash)

    if st.session_state.get("confirm_clear"):
        st.warning("确定清除全部历史记录吗？清除后统计将从 0 重新开始，且无法恢复。")
        c_yes, c_no = st.columns([1, 1])
        if c_yes.button("确认清除", type="primary", use_container_width=True):
            try:
                deleted = clear_history()
            except Exception as exc:
                st.error(f"清除失败：{exc}")
            else:
                st.session_state["confirm_clear"] = False
                st.session_state["clear_flash"] = f"已清除 {deleted} 条历史记录。"
                st.rerun()
        if c_no.button("取消", use_container_width=True):
            st.session_state["confirm_clear"] = False
            st.rerun()

    try:
        history = get_history(limit=None)
    except Exception as e:
        st.error(f"读取历史记录失败：{e}")
        return

    if not history:
        st.info(
            "暂无处置记录。在上传与识别页面完成几次分类后，"
            "记录就会显示在这里。"
        )
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp", ascending=False)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    low_conf = float((df["confidence"] < 0.6).mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("识别总数", f"{len(df)}")
    m2.metric("平均置信度", f"{df['confidence'].mean():.1%}")
    m3.metric("低置信占比", f"{low_conf:.1%}", help="置信度低于 60% 的记录占比，可视为需人工复核的比例")
    m4.metric("覆盖类别", f"{df['category'].nunique()} / 7")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = _plot_category_distribution(df["category"])
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = _plot_recyclable_pie(df["recyclable"])
        st.plotly_chart(fig_pie, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(_plot_confidence_hist(df["confidence"]), use_container_width=True)

    with col4:
        st.plotly_chart(_plot_confidence_by_category(df), use_container_width=True)

    st.divider()
    st.subheader("最近处置记录")

    display = df.head(200)[["timestamp", "category", "confidence"]].copy()
    display["confidence"] = display["confidence"].apply(_format_confidence)
    display["category"] = display["category"].apply(get_category_label)
    display.columns = ["时间", "类别", "置信度"]

    col_filter, col_export = st.columns([3, 1])
    with col_filter:
        category_filter = st.multiselect(
            "按类别筛选",
            options=sorted(df["category"].unique()),
            format_func=get_category_label,
            default=[],
            placeholder="全部类别",
        )
    with col_export:
        st.write("")
        st.write("")
        csv = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="导出 CSV",
            data=csv,
            file_name="disposal_history.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True,
        )

    filtered = display
    if category_filter:
        filtered = filtered[filtered["类别"].isin([get_category_label(c) for c in category_filter])]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "时间": st.column_config.DatetimeColumn("时间"),
            "类别": st.column_config.TextColumn("类别"),
            "置信度": st.column_config.TextColumn("置信度"),
        },
    )

    total = len(df)
    recyclable_count = int(df["recyclable"].sum())
    st.caption(
        f"识别总数：{total} | "
        f"可回收：{recyclable_count} | "
        f"不可回收：{total - recyclable_count}"
    )


if __name__ == "__main__":
    main()
