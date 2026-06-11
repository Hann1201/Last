
# 介质1到介质2的电磁波传播仿真模块

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import matplotlib.font_manager as fm
from pathlib import Path
from io import BytesIO
from PIL import Image

# ----------------------
# 注册字体
# ----------------------
font_path = Path(__file__).parent / "fonts" / "simhei.ttf"
if not font_path.exists():
    font_path = Path(__file__).parent / "fonts" / "SimsunExtG.ttf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
else:
    font_prop = fm.FontProperties()

# 物理常量
c = 3e8


def get_wave_at_time(t, x, amplitude, k1, k2, eta1, eta2, interface_pos, lam1, lam2):
    """计算给定时刻 t 的电场和磁场分布（动态）"""
    omega = 2 * np.pi * c / lam1 if lam1 > 0 else 1e9
    E = np.zeros_like(x)
    H = np.zeros_like(x)

    mask1 = x < interface_pos
    mask2 = x >= interface_pos

    # 介质1中的波（入射波）
    E[mask1] = amplitude * np.sin(k1 * x[mask1] - omega * t)
    H[mask1] = (amplitude / eta1) * np.cos(k1 * x[mask1] - omega * t)

    # 介质2中的波（透射波，相位连续）
    phase_at_boundary = k1 * interface_pos - omega * t
    E[mask2] = amplitude * np.sin(k2 * (x[mask2] - interface_pos) + phase_at_boundary)
    H[mask2] = (amplitude / eta2) * np.cos(k2 * (x[mask2] - interface_pos) + phase_at_boundary)

    return E, H


def create_single_frame(t, x, amplitude, k1, k2, eta1, eta2, interface_pos, 
                        lam1, lam2, v1, v2, er1, ur1, er2, ur2, f, 
                        figsize=(12, 9)):
    """创建单帧图像"""
    E_1d, H_1d = get_wave_at_time(t, x, amplitude, k1, k2, eta1, eta2, interface_pos, lam1, lam2)

    T1 = lam1 / v1 if v1 > 0 else 1e-9

    fig = plt.figure(figsize=figsize)

    # ========== 上图：2D波形图 ==========
    ax2 = fig.add_subplot(2, 1, 1)

    ax2_e = ax2
    line_e, = ax2_e.plot(x, E_1d, 'b-', lw=2.5, label='电场 E(x,t)', alpha=0.8)
    ax2_e.set_xlabel('传播方向 z (m)', fontsize=11, fontproperties=font_prop)
    ax2_e.set_ylabel('电场强度 E (V/m)', fontsize=11, fontproperties=font_prop, color='blue')
    ax2_e.tick_params(axis='y', labelcolor='blue')
    ax2_e.set_ylim(-amplitude * 1.5, amplitude * 1.5)

    ax2_h = ax2_e.twinx()
    line_h, = ax2_h.plot(x, H_1d, color='orange', lw=2, label='磁场 H(x,t)', alpha=0.8)
    ax2_h.set_ylabel('磁场强度 H (A/m)', fontsize=11, fontproperties=font_prop, color='orange')
    ax2_h.tick_params(axis='y', labelcolor='orange')
    h_max = max(np.max(np.abs(H_1d)), 1e-10)
    ax2_h.set_ylim(-h_max * 2, h_max * 2)

    ax2_e.axvline(x=interface_pos, color='r', linestyle='--', lw=2, label='介质分界面')

    ylim_e = ax2_e.get_ylim()
    y_pos = ylim_e[1] * 0.85
    ax2_e.text(interface_pos / 2, y_pos, f'区域1 (εr={er1:.1f})',
             ha='center', fontsize=10, fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax2_e.text(interface_pos + (x.max() - interface_pos) / 2, y_pos,
             f'区域2 (εr={er2:.1f})', ha='center', fontsize=10,
             fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    lines = [line_e, line_h]
    labels = [l.get_label() for l in lines]
    ax2_e.legend(lines, labels, loc='upper right', prop=font_prop, fontsize=9)
    ax2_e.grid(True, alpha=0.3)
    ax2_e.set_xlim([0, 10])
    ax2_e.set_title(f'介质1→介质2传播波形 (t = {t*1e9:.2f} ns, T₁ = {T1*1e9:.2f} ns, f={f/1e6:.1f}MHz)',
                   fontsize=11, fontproperties=font_prop)

    # ========== 下图：3D传播图 ==========
    ax3d = fig.add_subplot(2, 1, 2, projection='3d')

    ax3d.plot(x, np.zeros_like(x), E_1d, lw=2.5, c='#1f77b4', label='电场 E')
    ax3d.plot(x, H_1d, np.zeros_like(x), lw=2.5, c='#ff6b35', label='磁场 H')

    H_range = [-h_max * 2, h_max * 2]
    E_range = [-amplitude * 1.5, amplitude * 1.5]
    ax3d.plot([interface_pos, interface_pos], H_range, E_range, 'r--', lw=2, label='介质分界面')

    ax3d.set_title(f'电磁波3D传播 (t = {t*1e9:.2f} ns)', fontsize=11, fontproperties=font_prop)
    ax3d.set_xlabel('z (m)', fontsize=9, fontproperties=font_prop)
    ax3d.set_ylabel('磁场 H', fontsize=9, fontproperties=font_prop)
    ax3d.set_zlabel('电场 E', fontsize=9, fontproperties=font_prop)
    ax3d.legend(loc='upper right', prop=font_prop, fontsize=8)
    ax3d.view_init(elev=25, azim=-60)
    ax3d.set_xlim([0, 10])
    ax3d.set_ylim(H_range)
    ax3d.set_zlim(E_range)

    plt.tight_layout()
    return fig


def generate_gif(frequency, wavelength, amplitude, er1, ur1, er2, ur2, interface_pos,
                 num_frames=36, periods=2, fps=10):
    """生成GIF动画"""
    # 计算参数
    if frequency is not None and frequency > 0:
        f = frequency
        lam0 = c / f
    else:
        lam0 = wavelength
        f = c / lam0

    v1 = c / np.sqrt(er1 * ur1)
    lam1 = v1 / f
    k1 = 2 * np.pi / lam1
    eta1 = 377 * np.sqrt(ur1 / er1)

    v2 = c / np.sqrt(er2 * ur2)
    lam2 = v2 / f
    k2 = 2 * np.pi / lam2
    eta2 = 377 * np.sqrt(ur2 / er2)

    x = np.linspace(0, 10, 600)
    T1 = lam1 / v1 if v1 > 0 else 1e-9

    frames = []
    times = np.linspace(0, periods * T1, num_frames, endpoint=False)

    progress_bar = st.progress(0, text="🎬 正在生成介质传播动画...")

    for i, t in enumerate(times):
        fig = create_single_frame(t, x, amplitude, k1, k2, eta1, eta2, interface_pos,
                                   lam1, lam2, v1, v2, er1, ur1, er2, ur2, f,
                                   figsize=(12, 9))
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        progress_bar.progress((i + 1) / num_frames, 
                               text=f"🎬 生成帧 {i+1}/{num_frames}...")

    progress_bar.empty()

    gif_buf = BytesIO()
    duration_ms = int(1000 / fps)
    frames[0].save(gif_buf, format='GIF', save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=0, optimize=True)
    gif_buf.seek(0)

    gamma = (eta2 - eta1) / (eta2 + eta1)
    tau = 2 * eta2 / (eta2 + eta1)

    return gif_buf, f, lam0, lam1, lam2, v1, v2, eta1, eta2, gamma, tau, T1


def run_media_propagation_simulation(frequency, wavelength, amplitude, er1, ur1, er2, ur2,
                                     interface_pos, col_main, col_info, 
                                     generate_animation=False, num_frames=36, 
                                     gif_fps=10, gif_periods=2):
    """运行介质间传播仿真"""
    try:
        if frequency is None and wavelength is None:
            raise ValueError("请设置频率或波长")

        if generate_animation:
            gif_buf, f, lam0, lam1, lam2, v1, v2, eta1, eta2, gamma, tau, T1 = generate_gif(
                frequency, wavelength, amplitude, er1, ur1, er2, ur2, interface_pos,
                num_frames=num_frames, periods=gif_periods, fps=gif_fps
            )

            with col_main:
                st.image(gif_buf, caption=f"介质1→介质2传播动画 ({gif_periods}个周期循环, {num_frames}帧, {gif_fps}fps)",
                        use_container_width=True)
                st.download_button(label="⬇️ 下载GIF动画", data=gif_buf.getvalue(),
                                   file_name="media_wave_propagation.gif", mime="image/gif",
                                   use_container_width=True)
        else:
            # 静态单帧
            if frequency is not None and frequency > 0:
                f = frequency
                lam0 = c / f
            else:
                lam0 = wavelength
                f = c / lam0

            v1 = c / np.sqrt(er1 * ur1)
            lam1 = v1 / f
            k1 = 2 * np.pi / lam1
            eta1 = 377 * np.sqrt(ur1 / er1)
            v2 = c / np.sqrt(er2 * ur2)
            lam2 = v2 / f
            k2 = 2 * np.pi / lam2
            eta2 = 377 * np.sqrt(ur2 / er2)
            x = np.linspace(0, 10, 600)
            T1 = lam1 / v1

            fig = create_single_frame(0, x, amplitude, k1, k2, eta1, eta2, interface_pos,
                                       lam1, lam2, v1, v2, er1, ur1, er2, ur2, f)
            with col_main:
                st.pyplot(fig)
                plt.close(fig)

            gamma = (eta2 - eta1) / (eta2 + eta1)
            tau = 2 * eta2 / (eta2 + eta1)

        with col_info:
            st.success("✅ 介质传播仿真完成！")
            st.info(f"""
📊 **传播参数**

**介质1**
• 波速: {v1:.2e} m/s
• 波长: {lam1:.4f} m
• 波阻抗: {eta1:.2f} Ω
• εr={er1:.2f}, μr={ur1:.2f}

**介质2**
• 波速: {v2:.2e} m/s
• 波长: {lam2:.4f} m
• 波阻抗: {eta2:.2f} Ω
• εr={er2:.2f}, μr={ur2:.2f}

**界面特性**
• 反射系数 Γ: {gamma:.4f}
• 透射系数 τ: {tau:.4f}
• 频率: {f / 1e6:.2f} MHz
            """)

        return True
    except Exception as e:
        with col_main:
            st.error(f"❌ 介质传播仿真失败: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

