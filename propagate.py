
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

c = 3e8  # 光速

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


def calculate_propagate(wavelength, amplitude, er, ur, interface_pos):
    """计算电磁波传播参数"""
    k0 = 2 * np.pi / wavelength
    k = k0 * np.sqrt(er * ur)
    eta0 = 377
    eta = eta0 * np.sqrt(ur / er)
    lam_med = wavelength / np.sqrt(er * ur)
    n = np.sqrt(er * ur)
    beta = k
    vp = c / np.sqrt(er * ur)
    return k0, k, eta0, eta, lam_med, n, beta, vp


def get_wave_at_time(t, x, amplitude, wavelength, lam_med, interface_pos, eta0, eta, n, k0):
    """计算给定时刻 t 的电场和磁场分布"""
    omega = 2 * np.pi * c / wavelength
    E = np.zeros_like(x)
    H = np.zeros_like(x)

    mask_left = x < interface_pos
    mask_right = x >= interface_pos

    # 左侧真空：入射波
    E[mask_left] = amplitude * np.sin(k0 * x[mask_left] - omega * t)
    H[mask_left] = (amplitude / eta0) * np.cos(k0 * x[mask_left] - omega * t)

    # 右侧介质：透射波，保持相位连续
    k_med = 2 * np.pi / lam_med
    phi0 = k0 * interface_pos
    E[mask_right] = amplitude * np.sin(k_med * (x[mask_right] - interface_pos) - omega * t + phi0)
    H[mask_right] = (amplitude / eta) * np.cos(k_med * (x[mask_right] - interface_pos) - omega * t + phi0)

    return E, H


def create_single_frame(t, x, z_3d, amplitude, wavelength, lam_med, interface_pos,
                        eta0, eta, n, k0, er, figsize=(14, 10)):
    """创建单帧图像，返回fig对象"""
    E, H = get_wave_at_time(t, x, amplitude, wavelength, lam_med, interface_pos, eta0, eta, n, k0)
    E_3d, H_3d = get_wave_at_time(t, z_3d, amplitude, wavelength, lam_med, interface_pos, eta0, eta, n, k0)

    T = wavelength / c

    fig = plt.figure(figsize=figsize)

    # ========== 上图：2D波形图 ==========
    ax2 = fig.add_subplot(2, 1, 1)

    ax2_e = ax2
    line_e, = ax2_e.plot(x, E, 'b-', linewidth=2.5, label='电场强度 E(z,t)', alpha=0.8)
    ax2_e.set_xlabel('传播方向 z (m)', fontsize=12, fontproperties=font_prop)
    ax2_e.set_ylabel('电场强度 E (V/m)', fontsize=12, fontproperties=font_prop, color='blue')
    ax2_e.tick_params(axis='y', labelcolor='blue')
    ax2_e.set_ylim(-amplitude * 1.5, amplitude * 1.5)

    ax2_h = ax2_e.twinx()
    line_h, = ax2_h.plot(x, H, 'r-', linewidth=2, label='磁场强度 H(z,t)', alpha=0.8)
    ax2_h.set_ylabel('磁场强度 H (A/m)', fontsize=12, fontproperties=font_prop, color='red')
    ax2_h.tick_params(axis='y', labelcolor='red')
    h_max = amplitude / eta0
    ax2_h.set_ylim(-h_max * 2, h_max * 2)

    ax2_e.axvline(interface_pos, color='k', linestyle='--', linewidth=2, label='介质分界面')

    ylim_e = ax2_e.get_ylim()
    y_pos = ylim_e[1] * 0.85
    ax2_e.text(interface_pos / 2, y_pos, '区域1 (真空)',
             ha='center', fontsize=11, fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax2_e.text(interface_pos + (x.max() - interface_pos) / 2, y_pos,
             f'区域2 (介质 εr={er})', ha='center', fontsize=11,
             fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    lines = [line_e, line_h]
    labels = [l.get_label() for l in lines]
    ax2_e.legend(lines, labels, loc='upper right', fontsize=10, prop=font_prop)
    ax2_e.grid(True, alpha=0.3, linestyle='--')
    ax2_e.set_title(f'电磁波传播波形 (t = {t*1e9:.2f} ns, T₀ = {T*1e9:.2f} ns, εr = {er}, n = {n:.2f})',
                   fontsize=12, fontproperties=font_prop)

    # ========== 下图：3D传播图 ==========
    ax3d = fig.add_subplot(2, 1, 2, projection='3d')

    ax3d.plot(z_3d, E_3d, np.zeros_like(z_3d), 'b-', linewidth=2, label='电场 E')
    ax3d.plot(z_3d, np.zeros_like(z_3d), H_3d, color='orange', linewidth=2, label='磁场 H')

    y_plane = np.linspace(-amplitude*1.5, amplitude*1.5, 10)
    z_plane = np.linspace(-h_max*2, h_max*2, 10)
    Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
    X_plane = np.full_like(Y_plane, interface_pos)
    ax3d.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.15, color='red')
    ax3d.plot([interface_pos, interface_pos], [0, 0], [0, 0], 'r--', linewidth=2, label='分界面')

    ax3d.set_xlabel('传播方向 z (m)', fontproperties=font_prop)
    ax3d.set_ylabel('电场 E (V/m)', fontproperties=font_prop)
    ax3d.set_zlabel('磁场 H (A/m)', fontproperties=font_prop)
    ax3d.set_title(f'电磁波3D传播 (t = {t*1e9:.2f} ns)', fontsize=12, fontproperties=font_prop)
    ax3d.legend(loc='upper right', fontsize=10, prop=font_prop)
    ax3d.set_xlim(0, 10)
    ax3d.set_ylim(-amplitude*1.5, amplitude*1.5)
    ax3d.set_zlim(-h_max*2, h_max*2)

    plt.tight_layout()
    return fig


def generate_gif(wavelength, amplitude, er, ur, interface_pos,
                 num_frames=36, periods=2, fps=10):
    """
    生成GIF动画，返回GIF字节流
    使用PIL合成GIF，无需额外依赖
    """
    from PIL import Image

    k0, k, eta0, eta, lam_med, n, beta, vp = calculate_propagate(
        wavelength, amplitude, er, ur, interface_pos
    )

    x = np.linspace(0, 10, 600)
    z_3d = np.linspace(0, 10, 200)
    T = wavelength / c

    frames = []
    times = np.linspace(0, periods * T, num_frames, endpoint=False)

    progress_bar = st.progress(0, text="🎬 正在生成动画帧...")

    for i, t in enumerate(times):
        fig = create_single_frame(t, x, z_3d, amplitude, wavelength, lam_med,
                                   interface_pos, eta0, eta, n, k0, er,
                                   figsize=(12, 9))
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        progress_bar.progress((i + 1) / num_frames, text=f"🎬 生成帧 {i+1}/{num_frames}...")

    progress_bar.empty()

    # 合成GIF
    gif_buf = BytesIO()
    duration_ms = int(1000 / fps)

    frames[0].save(
        gif_buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )

    gif_buf.seek(0)
    return gif_buf, T, beta, vp, eta, n, lam_med, k0


def run_propagate_simulation(wavelength, amplitude, er, ur, interface_pos,
                              col_main, col_info, generate_animation=False,
                              num_frames=36, gif_fps=10, gif_periods=2):
    """
    运行传播仿真
    generate_animation: 是否生成GIF动画
    """
    try:
        if generate_animation:
            gif_buf, T, beta, vp, eta, n, lam_med, k0 = generate_gif(
                wavelength, amplitude, er, ur, interface_pos,
                num_frames=num_frames, periods=gif_periods, fps=gif_fps
            )

            with col_main:
                st.image(gif_buf, caption=f"电磁波传播动画 ({gif_periods}个周期循环, {num_frames}帧, {gif_fps}fps)",
                        use_container_width=True)

                # 提供下载按钮
                st.download_button(
                    label="⬇️ 下载GIF动画",
                    data=gif_buf.getvalue(),
                    file_name="wave_propagation.gif",
                    mime="image/gif",
                    use_container_width=True
                )
        else:
            # 静态单帧
            k0, k, eta0, eta, lam_med, n, beta, vp = calculate_propagate(
                wavelength, amplitude, er, ur, interface_pos
            )
            x = np.linspace(0, 10, 600)
            z_3d = np.linspace(0, 10, 200)
            T = wavelength / c

            fig = create_single_frame(0, x, z_3d, amplitude, wavelength, lam_med,
                                       interface_pos, eta0, eta, n, k0, er)

            with col_main:
                st.pyplot(fig)
                plt.close(fig)

        with col_info:
            st.success("✅ 仿真完成！")
            st.info(f"""
📊 **传播参数**

**波参数**
• 相位常数 β: {beta:.4f} rad/m
• 介质波长 λ: {lam_med:.4f} m
• 真空波数 k₀: {k0:.4f} m⁻¹

**传播特性**
• 相速度 vp: {vp:.2e} m/s
• 波阻抗 η: {eta:.2f} Ω
• 折射率 n: {n:.3f}

**时间参数**
• 真空周期 T₀: {T*1e9:.4f} ns

**介质参数**
• 相对介电常数 εᵣ: {er}
• 相对磁导率 μᵣ: {ur}
• 真空波长 λ₀: {wavelength:.4f} m
            """)

        return True

    except Exception as e:
        with col_main:
            st.error(f"❌ 仿真失败: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False
