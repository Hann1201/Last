
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

    # ========== 动画选项（所有模式通用）==========
    st.subheader("🎬 动画选项")
    enable_animation = st.toggle("启用GIF动画", value=True,
                                  help="生成动态GIF展示波的传播过程，播放更连贯流畅")

    if enable_animation:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            num_frames = st.number_input("帧数", min_value=12, max_value=60, value=36, step=6, key="gif_frames",
                                          help="帧数越多动画越流畅，但生成时间越长")
        with col_f2:
            gif_fps = st.number_input("帧率 (fps)", min_value=5, max_value=30, value=10, step=1, key="gif_fps",
                                       help="每秒播放的帧数")

        gif_periods = st.number_input("播放周期数", min_value=1, max_value=4, value=2, step=1, key="gif_periods",
                                       help="GIF循环播放几个周期")

        st.info(f"📊 预计GIF时长: {gif_periods * num_frames / gif_fps:.1f}秒")
    else:
        num_frames = 36
        gif_fps = 10
        gif_periods = 2

    st.markdown("---")

    # 根据模式显示不同参数
    if st.session_state.mode == "vacuum_to_media":
        st.subheader("📡 真空→介质参数")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            amplitude = st.number_input("振幅 E₀ (V/m)", min_value=0.2, max_value=3.0, value=1.0, step=0.1, key="vac_amp")
        with col_a2:
            interface_pos = st.number_input("分界面位置 (m)", min_value=2.0, max_value=8.0, value=5.0, step=0.5, key="vac_pos")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            wavelength = st.number_input("真空波长 λ₀ (m)", min_value=0.5, max_value=5.0, value=2.0, step=0.1, key="vac_wavelength")
        with col_b2:
            er = st.number_input("相对介电常数 εᵣ", min_value=1.0, max_value=10.0, value=4.0, step=0.1, key="vac_er")
        ur = st.number_input("相对磁导率 μᵣ", min_value=1.0, max_value=5.0, value=1.0, step=0.1, key="vac_ur")

        run_btn = st.button("🚀 开始传播仿真", use_container_width=True, type="primary")
        frequency = None
        wavelength_media = None
        er1 = ur1 = er2 = ur2 = None
        theta_i = n2_refl = None
        lam0_refl = amp_refl = None
        tand1 = tand2 = None

    elif st.session_state.mode == "media_to_media":
        st.subheader("🔄 介质1→介质2参数")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            amplitude = st.number_input("振幅 E₀ (V/m)", min_value=0.2, max_value=3.0, value=1.0, step=0.1, key="media_amp")
        with col_a2:
            interface_pos = st.number_input("分界面位置 (m)", min_value=2.0, max_value=8.0, value=5.0, step=0.5, key="media_pos")
        freq_or_lam = st.radio("输入方式", ["频率", "真空波长"], index=0, horizontal=True)

        if freq_or_lam == "频率":
            frequency = st.number_input("频率 f (MHz)", min_value=30.0, max_value=300.0, value=100.0, step=10.0, key="media_freq")
            wavelength_media = None
        else:
            frequency = None
            wavelength_media = st.number_input("真空波长 λ₀ (m)", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="media_wavelength")

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**介质1**")
            er1 = st.number_input("εᵣ₁", min_value=1.0, max_value=6.0, value=1.0, step=0.1, key="er1")
            ur1 = st.number_input("μᵣ₁", min_value=1.0, max_value=4.0, value=1.0, step=0.1, key="ur1")
        with col_m2:
            st.markdown("**介质2**")
            er2 = st.number_input("εᵣ₂", min_value=1.0, max_value=10.0, value=4.0, step=0.1, key="er2")
            ur2 = st.number_input("μᵣ₂", min_value=1.0, max_value=6.0, value=2.0, step=0.1, key="ur2")

        run_btn = st.button("🚀 开始传播仿真", use_container_width=True, type="primary")
        wavelength = None
        er = None
        theta_i = n2_refl = None
        lam0_refl = amp_refl = None
        tand1 = tand2 = None

    else:  # reflection
        st.subheader("🔍 反射与折射参数")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            theta_i = st.number_input("入射角 θi (度)", min_value=0.0, max_value=80.0, value=30.0, step=1.0, key="theta_i")
        with col_r2:
            n2_refl = st.number_input("介质2折射率 n₂", min_value=0.5, max_value=2.5, value=1.5, step=0.01, key="n2_refl")

        st.markdown("---")
        st.markdown("**波形参数**")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            lam0_refl = st.number_input("真空波长 λ₀ (m)", min_value=0.5, max_value=5.0, value=2.0, step=0.1, key="refl_lam0")
        with col_w2:
            amp_refl = st.number_input("振幅 E₀ (V/m)", min_value=0.2, max_value=3.0, value=1.0, step=0.1, key="refl_amp")

        st.markdown("---")
        st.markdown("**介质1参数**")
        col_m1a, col_m1b, col_m1c = st.columns(3)
        with col_m1a:
            er1 = st.number_input("εᵣ₁", min_value=1.0, max_value=10.0, value=1.0, step=0.1, key="refl_er1")
        with col_m1b:
            mur1 = st.number_input("μᵣ₁", min_value=1.0, max_value=5.0, value=1.0, step=0.1, key="refl_mur1")
        with col_m1c:
            tand1 = st.number_input("tanδ₁", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="refl_tand1")

        st.markdown("**介质2参数**")
        col_m2a, col_m2b, col_m2c = st.columns(3)
        with col_m2a:
            er2 = st.number_input("εᵣ₂", min_value=1.0, max_value=10.0, value=2.25, step=0.1, key="refl_er2")
        with col_m2b:
            mur2 = st.number_input("μᵣ₂", min_value=1.0, max_value=5.0, value=1.0, step=0.1, key="refl_mur2")
        with col_m2c:
            tand2 = st.number_input("tanδ₂", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="refl_tand2")

        run_btn = st.button("🚀 开始反射仿真", use_container_width=True, type="primary")
        amplitude = interface_pos = wavelength = er = None
        frequency = wavelength_media = None
        ur = None

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
                    col_info=col_info,
                    generate_animation=enable_animation,
                    num_frames=num_frames,
                    gif_fps=gif_fps,
                    gif_periods=gif_periods
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
                    col_info=col_info,
                    generate_animation=enable_animation,
                    num_frames=num_frames,
                    gif_fps=gif_fps,
                    gif_periods=gif_periods
                )
                if success:
                    st.toast("✅ 介质1→介质2传播仿真完成！", icon="🎉")
        else:
            with col_main:
                st.error("❌ 请设置介质1和介质2参数")

    else:  # reflection
        if theta_i is not None and n2_refl is not None and lam0_refl is not None and amp_refl is not None:
            with st.spinner("正在计算反射与折射..."):
                success = run_reflection_simulation(
                    theta_i_deg=theta_i,
                    n2=n2_refl,
                    lam0=lam0_refl,
                    amp=amp_refl,
                    er1=er1,
                    mur1=mur1,
                    tand1=tand1,
                    er2=er2,
                    mur2=mur2,
                    tand2=tand2,
                    col_main=col_main,
                    col_info=col_info,
                    generate_animation=enable_animation,
                    num_frames=num_frames,
                    gif_fps=gif_fps,
                    gif_periods=gif_periods
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
