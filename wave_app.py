import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np
from propagate import run_propagate_simulation
from jiezhi import run_media_propagation_simulation

# ---------- 全局强制中文（关键！） ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 加载本地字体（云端必须带这个）
font_path = Path(__file__).parent / "fonts" / "simsunb.ttf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    plt.rcParams['font.sans-serif'] = ['SimSun']
    font_prop = fm.FontProperties(fname=str(font_path))
else:
    font_prop = None

# ---------- 页面 ----------
st.set_page_config(page_title="电磁波传播仿真", layout="wide")
st.title("📡 电磁波传播仿真")

if 'mode' not in st.session_state:
    st.session_state.mode = "vacuum_to_media"
if 'run_trigger' not in st.session_state:
    st.session_state.run_trigger = False

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 通用参数")
    amplitude = st.slider("振幅 E₀ (V/m)", 0.2, 3.0, 1.0, 0.1)
    interface_pos = st.slider("分界面位置 (m)", 2.0, 8.0, 5.0, 0.5)

    st.markdown("---")
    st.subheader("选择仿真模式")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("真空→介质", type="primary" if st.session_state.mode == "vacuum_to_media" else "secondary"):
            st.session_state.mode = "vacuum_to_media"
            st.session_state.run_trigger = False
    with col2:
        if st.button("介质1→介质2", type="primary" if st.session_state.mode == "media_to_media" else "secondary"):
            st.session_state.mode = "media_to_media"
            st.session_state.run_trigger = False

    st.markdown("---")
    if st.session_state.mode == "vacuum_to_media":
        st.subheader("真空→介质参数")
        lambda0 = st.slider("真空波长 λ₀ (m)", 0.5, 5.0, 2.0, 0.1)
        er2 = st.slider("相对介电常数 εᵣ", 1.0, 10.0, 4.0, 0.1)
        params = {"lambda0": lambda0, "er2": er2}
    else:
        st.subheader("介质1→介质2参数")
        lambda1 = st.slider("介质1波长 λ₁ (m)", 0.5, 5.0, 2.0, 0.1)
        er1 = st.slider("介质1相对介电常数 εᵣ₁", 1.0, 10.0, 2.0, 0.1)
        ur1 = st.slider("介质1相对磁导率 μᵣ₁", 1.0, 5.0, 1.0, 0.1)
        er2 = st.slider("介质2相对介电常数 εᵣ₂", 1.0, 10.0, 4.0, 0.1)
        ur2 = st.slider("介质2相对磁导率 μᵣ₂", 1.0, 5.0, 1.0, 0.1)
        params = {"lambda1": lambda1, "er1": er1, "ur1": ur1, "er2": er2, "ur2": ur2}

    st.markdown("---")
    if st.button("▶️ 开始仿真", type="primary", use_container_width=True):
        st.session_state.run_trigger = True

# ---------- 运行仿真 ----------
if st.session_state.run_trigger:
    with st.spinner("正在仿真中..."):
        if st.session_state.mode == "vacuum_to_media":
            fig, info = run_propagate_simulation(
                amplitude, interface_pos,
                params["lambda0"], params["er2"]
            )
        else:
            fig, info = run_media_propagation_simulation(
                amplitude, interface_pos,
                params["lambda1"], params["er1"], params["ur1"],
                params["er2"], params["ur2"]
            )
        st.pyplot(fig)
        st.success("✅ 仿真完成!")
        with st.expander("📊 传播参数", expanded=True):
            for key, value in info.items():
                st.write(f"**{key}**: {value}")
else:
    st.info("请在左侧设置参数并点击「开始仿真」按钮")
