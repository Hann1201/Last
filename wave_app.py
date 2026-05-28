# wave_app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from propagate import run_propagate_simulation
from jiezhi import run_media_propagation_simulation

# 页面配置
st.set_page_config(page_title="电磁波传播仿真", layout="wide")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.title("📡 电磁波传播仿真")

# ========== 初始化会话状态 ==========
if 'mode' not in st.session_state:
    st.session_state.mode = "vacuum_to_media"  # 默认模式
if 'run_trigger' not in st.session_state:
    st.session_state.run_trigger = False

# ========== 侧边栏参数 ==========
with st.sidebar:
    st.header("⚙️ 通用参数")
    amplitude = st.slider("振幅 E₀ (V/m)", 0.2, 3.0, 1.0, 0.1)
    interface_pos = st.slider("分界面位置 (m)", 2.0, 8.0, 5.0, 0.5)

    st.markdown("---")

    # ========== 模式选择按钮（带高亮效果）==========
    st.subheader("📌 选择仿真模式")

    col_btn1, col_btn2 = st.columns(2)

    # 按钮1：真空→介质
    with col_btn1:
        if st.session_state.mode == "vacuum_to_media":
            # 当前选中的按钮显示为红色（高亮）
            btn1 = st.button("📡 真空→介质", use_container_width=True,
                             type="primary", key="btn1_active")
        else:
            btn1 = st.button("📡 真空→介质", use_container_width=True,
                             type="secondary", key="btn1_inactive")

    # 按钮2：介质1→介质2
    with col_btn2:
        if st.session_state.mode == "media_to_media":
            # 当前选中的按钮显示为红色（高亮）
            btn2 = st.button("🔄 介质1→介质2", use_container_width=True,
                             type="primary", key="btn2_active")
        else:
            btn2 = st.button("🔄 介质1→介质2", use_container_width=True,
                             type="secondary", key="btn2_inactive")

    # 处理按钮点击：切换模式
    if btn1:
        st.session_state.mode = "vacuum_to_media"
        st.session_state.run_trigger = False
        st.rerun()

    if btn2:
        st.session_state.mode = "media_to_media"
        st.session_state.run_trigger = False
        st.rerun()

    st.markdown("---")

    # ========== 根据当前模式显示不同的参数面板 ==========

    if st.session_state.mode == "vacuum_to_media":
        st.subheader("📡 真空→介质参数")
        wavelength = st.slider("真空波长 λ₀ (m)", 0.5, 5.0, 2.0, 0.1, key="vac_wavelength")
        er = st.slider("相对介电常数 εᵣ", 1.0, 10.0, 4.0, 0.1, key="vac_er")

        # 运行按钮
        run_btn = st.button("🚀 开始传播仿真", use_container_width=True, type="primary")

        # 设置介质参数为None
        frequency = None
        wavelength_media = None
        er1 = ur1 = er2 = ur2 = None

    else:  # media_to_media
        st.subheader("🔄 介质1→介质2参数")

        # 输入方式选择
        freq_or_lam = st.radio("输入方式", ["频率", "真空波长"], index=0, horizontal=True)

        if freq_or_lam == "频率":
            frequency = st.slider("频率 f (MHz)", 30.0, 300.0, 100.0, 10.0, key="media_freq")
            wavelength_media = None
        else:
            frequency = None
            wavelength_media = st.slider("真空波长 λ₀ (m)", 1.0, 10.0, 3.0, 0.5, key="media_wavelength")

        st.markdown("---")

        # 介质1和介质2参数（两列布局）
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("**介质1**")
            er1 = st.slider("εᵣ₁", 1.0, 6.0, 1.0, 0.1, key="er1")
            ur1 = st.slider("μᵣ₁", 1.0, 4.0, 1.0, 0.1, key="ur1")

        with col_m2:
            st.markdown("**介质2**")
            er2 = st.slider("εᵣ₂", 1.0, 10.0, 4.0, 0.1, key="er2")
            ur2 = st.slider("μᵣ₂", 1.0, 6.0, 2.0, 0.1, key="ur2")

        # 运行按钮
        run_btn = st.button("🚀 开始传播仿真", use_container_width=True, type="primary")

        # 设置真空参数为None
        wavelength = None
        er = None

# 主显示区域
col_main, col_info = st.columns([3, 1])

# ========== 运行仿真 ==========
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
                    interface_pos=interface_pos,
                    col_main=col_main,
                    col_info=col_info
                )
                if success:
                    st.toast("✅ 真空→介质传播仿真完成！", icon="🎉")
        else:
            with col_main:
                st.error("❌ 请设置真空→介质参数")

    else:  # media_to_media
        if er1 is not None and ur1 is not None and er2 is not None and ur2 is not None:
            with st.spinner("正在计算介质间传播..."):
                # 转换频率单位
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

else:
    # 显示当前模式的说明
    with col_main:
        if st.session_state.mode == "vacuum_to_media":
            st.info("👈 选择「真空→介质」模式，设置参数后点击「开始传播仿真」")
            st.markdown("""
            ### 📡 真空→介质传播

            电磁波从真空进入介质时的变化：
            - **波长变短**：λ = λ₀/√εᵣ
            - **速度减慢**：v = c/√εᵣ
            - **波阻抗减小**：η = η₀/√εᵣ

            **左侧3D图**：展示电场在空间中的传播
            **右侧波形图**：展示电场E和磁场H的变化
            """)
        else:
            st.info("👈 选择「介质1→介质2」模式，设置参数后点击「开始传播仿真」")
            st.markdown("""
            ### 🔄 介质1→介质2传播

            电磁波从一种介质进入另一种介质时：
            - **部分反射，部分透射**
            - **反射系数**：Γ = (η₂-η₁)/(η₂+η₁)
            - **透射系数**：τ = 2η₂/(η₂+η₁)

            **左侧3D图**：展示电场在两种介质中的传播
            **右侧波形图**：展示波长变化和界面连续性
            """)

    with col_info:
        st.info("📌 设置参数后点击「开始传播仿真」")