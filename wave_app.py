# wave_app.py
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np

# ----------------------
# 注册字体
# ----------------------
font_path = Path(__file__).parent / "fonts" / "simhei.ttf"
if not font_path.exists():
    font_path = Path(__file__).parent / "fonts" / "SimsunExtG.ttf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    actual_font_name = font_prop.get_name()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [actual_font_name, 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(page_title="电磁波传播仿真", layout="wide")

# 标题
st.title("📡 电磁波传播仿真")

# 导入子模块
from propagate import run_propagate_simulation
from jiezhi import run_media_propagation_simulation
from reflectionandtransmission import run_reflection_simulation

# ========== 初始化会话状态 ==========
if 'mode' not in st.session_state:
    st.session_state.mode = "vacuum_to_media"
if 'run_trigger' not in st.session_state:
    st.session_state.run_trigger = False

# ========== 侧边栏参数 ==========
with st.sidebar:
    st.header("⚙️ 通用参数")

    # 模式选择
    st.subheader("📌 选择仿真模式")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.session_state.mode == "vacuum_to_media":
            btn1 = st.button("📡 真空→介质", use_container_width=True, type="primary", key="btn1_active")
        else:
            btn1 = st.button("📡 真空→介质", use_container_width=True, type="secondary", key="btn1_inactive")

    with col_btn2:
        if st.session_state.mode == "media_to_media":
            btn2 = st.button("🔄 介质1→介质2", use_container_width=True, type="primary", key="btn2_active")
        else:
            btn2 = st.button("🔄 介质1→介质2", use_container_width=True, type="secondary", key="btn2_inactive")

    with col_btn3:
        if st.session_state.mode == "reflection":
            btn3 = st.button("🔍 反射与折射", use_container_width=True, type="primary", key="btn3_active")
        else:
            btn3 = st.button("🔍 反射与折射", use_container_width=True, type="secondary", key="btn3_inactive")

    if btn1:
        st.session_state.mode = "vacuum_to_media"
        st.session_state.run_trigger = False
        st.rerun()

    if btn2:
        st.session_state.mode = "media_to_media"
        st.session_state.run_trigger = False
        st.rerun()

    if btn3:
        st.session_state.mode = "reflection"
        st.session_state.run_trigger = False
        st.rerun()

    st.markdown("---")

    # 根据模式显示不同参数
    if st.session_state.mode == "vacuum_to_media":
        st.subheader("📡 真空→介质参数")
        amplitude = st.slider("振幅 E₀ (V/m)", 0.2, 3.0, 1.0, 0.1)
        interface_pos = st.slider("分界面位置 (m)", 2.0, 8.0, 5.0, 0.5)
        wavelength = st.slider("真空波长 λ₀ (m)", 0.5, 5.0, 2.0, 0.1, key="vac_wavelength")
        er = st.slider("相对介电常数 εᵣ", 1.0, 10.0, 4.0, 0.1, key="vac_er")
        ur = st.slider("相对磁导率 μᵣ", 1.0, 5.0, 1.0, 0.1, key="vac_ur")
        run_btn = st.button("🚀 开始传播仿真", use_container_width=True, type="primary")
        frequency = None
        wavelength_media = None
        er1 = ur1 = er2 = ur2 = None
        theta_i = n2_refl = None

    elif st.session_state.mode == "media_to_media":
        st.subheader("🔄 介质1→介质2参数")
        amplitude = st.slider("振幅 E₀ (V/m)", 0.2, 3.0, 1.0, 0.1)
        interface_pos = st.slider("分界面位置 (m)", 2.0, 8.0, 5.0, 0.5)
        freq_or_lam = st.radio("输入方式", ["频率", "真空波长"], index=0, horizontal=True)

        if freq_or_lam == "频率":
            frequency = st.slider("频率 f (MHz)", 30.0, 300.0, 100.0, 10.0, key="media_freq")
            wavelength_media = None
        else:
            frequency = None
            wavelength_media = st.slider("真空波长 λ₀ (m)", 1.0, 10.0, 3.0, 0.5, key="media_wavelength")

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**介质1**")
            er1 = st.slider("εᵣ₁", 1.0, 6.0, 1.0, 0.1, key="er1")
            ur1 = st.slider("μᵣ₁", 1.0, 4.0, 1.0, 0.1, key="ur1")
        with col_m2:
            st.markdown("**介质2**")
            er2 = st.slider("εᵣ₂", 1.0, 10.0, 4.0, 0.1, key="er2")
            ur2 = st.slider("μᵣ₂", 1.0, 6.0, 2.0, 0.1, key="ur2")

        run_btn = st.button("🚀 开始传播仿真", use_container_width=True, type="primary")
        wavelength = None
        er = None
        theta_i = n2_refl = None

    else:  # reflection
        st.subheader("🔍 反射与折射参数")
        theta_i = st.slider("入射角 θi (度)", 0.0, 80.0, 30.0, 1.0, key="theta_i")
        n2_refl = st.slider("介质2折射率 n₂", 0.5, 2.5, 1.5, 0.01, key="n2_refl")
        run_btn = st.button("🚀 开始反射仿真", use_container_width=True, type="primary")
        amplitude = interface_pos = wavelength = er = None
        frequency = wavelength_media = None
        er1 = ur1 = er2 = ur2 = None

col_main, col_info = st.columns([3, 1])

if run_btn:
    st.session_state.run_trigger = True

if st.session_state.run_trigger:
    if st.session_state.mode == "vacuum_to_media":
        if wavelength is not None and er is not None:
            with st.spinner("正在计算电磁波传播..."):
                success = run_propagate_simulation(
                    wavelength=wavelength,
                    amplitude=amplitude,
                    er=er,
                    ur=ur,
                    interface_pos=interface_pos,
                    col_main=col_main,
                    col_info=col_info
                )
                if success:
                    st.toast("✅ 真空→介质传播仿真完成！", icon="🎉")
        else:
            with col_main:
                st.error("❌ 请设置真空→介质参数")

    elif st.session_state.mode == "media_to_media":
        if er1 is not None and ur1 is not None and er2 is not None and ur2 is not None:
            with st.spinner("正在计算介质间传播..."):
                freq_hz = frequency * 1e6 if frequency is not None else None
                success = run_media_propagation_simulation(
                    frequency=freq_hz,
                    wavelength=wavelength_media,
                    amplitude=amplitude,
                    er1=er1,
                    ur1=ur1,
                    er2=er2,
                    ur2=ur2,
                    interface_pos=interface_pos,
                    col_main=col_main,
                    col_info=col_info
                )
                if success:
                    st.toast("✅ 介质1→介质2传播仿真完成！", icon="🎉")
        else:
            with col_main:
                st.error("❌ 请设置介质1和介质2参数")

    else:  # reflection
        if theta_i is not None and n2_refl is not None:
            with st.spinner("正在计算反射与折射..."):
                success = run_reflection_simulation(
                    theta_i_deg=theta_i,
                    n2=n2_refl,
                    col_main=col_main,
                    col_info=col_info
                )
                if success:
                    st.toast("✅ 反射与折射仿真完成！", icon="🎉")
        else:
            with col_main:
                st.error("❌ 请设置反射与折射参数")

else:
    with col_main:
        if st.session_state.mode == "vacuum_to_media":
            st.info("👈 选择「真空→介质」模式，设置参数后点击「开始传播仿真」")
            st.markdown("""
            ### 📡 真空→介质传播

            电磁波从真空进入介质时的变化：
            - **波长变短**：λ = λ₀/√εᵣ
            - **速度减慢**：v = c/√εᵣ
            - **波阻抗减小**：η = η₀/√εᵣ
            """)
        elif st.session_state.mode == "media_to_media":
            st.info("👈 选择「介质1→介质2」模式，设置参数后点击「开始传播仿真」")
            st.markdown("""
            ### 🔄 介质1→介质2传播

            电磁波从一种介质进入另一种介质时：
            - **部分反射，部分透射**
            - **反射系数**：Γ = (η₂-η₁)/(η₂+η₁)
            - **透射系数**：τ = 2η₂/(η₂+η₁)
            """)
        else:
            st.info("👈 选择「反射与折射」模式，设置参数后点击「开始反射仿真」")
            st.markdown("""
            ### 🔍 反射与折射

            电磁波在两种介质界面的反射与折射：
            - **斯涅耳定律**：n₁sinθi = n₂sinθt
            - **菲涅耳公式**：计算反射/透射系数
            - **全反射**：当入射角大于临界角时发生
            """)

    with col_info:
        st.info("📌 设置参数后点击「开始仿真」")