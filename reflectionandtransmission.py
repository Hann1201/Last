
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from pathlib import Path
import streamlit as st
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

# 物理常数
c = 3e8
eta0 = 377.0
n1 = 1.0  # 介质1折射率

# ========== 颜色定义（全局统一）==========
COLOR_INCIDENT = '#e74c3c'   # 红色 - 入射波
COLOR_REFLECTED = '#f39c12'  # 橙色 - 反射波
COLOR_TRANSMITTED = '#27ae60'  # 绿色 - 透射波
COLOR_TOTAL = '#3498db'      # 蓝色 - 总电场（备用）
COLOR_BOUNDARY = '#2c3e50'   # 深灰 - 分界面


def compute_angles(theta_i_deg, n2):
    """计算反射角、折射角和反射/透射系数 (TE波)"""
    theta_i = np.radians(theta_i_deg)
    theta_r_deg = theta_i_deg
    sin_theta_t = (n1 / n2) * np.sin(theta_i)

    if abs(sin_theta_t) >= 1.0:
        theta_t_deg = 90.0
        total_reflection = True
    else:
        theta_t_deg = np.degrees(np.arcsin(sin_theta_t))
        total_reflection = False

    cos_i = np.cos(theta_i)

    if not total_reflection:
        cos_t = np.cos(np.radians(theta_t_deg))
        denom = n1 * cos_i + n2 * cos_t
        if denom != 0:
            r = (n1 * cos_i - n2 * cos_t) / denom
            t = 2 * n1 * cos_i / denom
        else:
            r = 1.0
            t = 0.0
    else:
        r = 1.0
        t = 0.0

    return theta_r_deg, theta_t_deg, r, t, total_reflection


def get_three_waves_time(t, lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_deg, bound=5.0):
    """
    计算三种独立波（入射波、反射波、透射波）在时刻 t 的分布
    返回: x, Ei, Er, Et, E_total_left, E_total_right
    """
    freq = c / lam0
    omega = 2 * np.pi * freq
    k0 = 2 * np.pi / lam0
    theta = np.radians(theta_deg)
    x = np.linspace(0, 10, 600)

    er1_c = er1 * (1 - 1j * tand1)
    er2_c = er2 * (1 - 1j * tand2)

    n1_c = np.sqrt(er1_c * mur1)
    eta1 = eta0 * np.sqrt(mur1 / er1_c)
    k1 = k0 * n1_c

    n2_c = np.sqrt(er2_c * mur2)
    eta2 = eta0 * np.sqrt(mur2 / er2_c)
    k2 = k0 * n2_c

    sin_t = (n1_c / n2_c) * np.sin(theta)
    sin_t = np.clip(np.real(sin_t), -1, 1)
    theta_t = np.arcsin(sin_t)
    cos1 = np.cos(theta)
    cos2 = np.cos(theta_t)

    r = (eta2 * cos1 - eta1 * cos2) / (eta2 * cos1 + eta1 * cos2)
    t_coeff = 2 * eta2 * cos1 / (eta2 * cos1 + eta1 * cos2)

    k1x = k1 * cos1
    k2x = k2 * cos2

    # 时间相位因子
    phase_t = np.exp(-1j * omega * t)

    # 入射波（只在介质1，向右传播）
    Ei = np.zeros_like(x, dtype=complex)
    # 反射波（只在介质1，向左传播）
    Er = np.zeros_like(x, dtype=complex)
    # 透射波（只在介质2，向右传播）
    Et = np.zeros_like(x, dtype=complex)

    idx_left = x < bound
    idx_right = x >= bound

    # 入射波
    Ei[idx_left] = amp * np.exp(1j * k1x * x[idx_left]) * phase_t
    # 反射波
    Er[idx_left] = r * amp * np.exp(-1j * k1x * x[idx_left]) * phase_t
    # 透射波
    Et[idx_right] = t_coeff * amp * np.exp(1j * k2x * (x[idx_right] - bound)) * phase_t

    # 总电场
    E_total_left = np.real(Ei + Er)
    E_total_right = np.real(Et)

    lam1 = np.real(lam0 / n1_c)
    lam2 = np.real(lam0 / n2_c)

    return (x, np.real(Ei), np.real(Er), np.real(Et),
            E_total_left, E_total_right, freq, lam1, lam2,
            theta_t, r, t_coeff, np.real(n1_c), np.real(n2_c))


def create_2d_frame_three_waves(t, lam0, amp, er1, mur1, tand1, er2, mur2, tand2,
                                 theta_deg, theta_r_deg, theta_t_deg, total_ref,
                                 figsize=(14, 6)):
    """创建2D波形单帧 - 显示三种独立波"""
    (x, Ei, Er, Et, E_total_left, E_total_right, freq,
     lam1, lam2, theta_t, r, t_coeff, n1_r, n2_r) = get_three_waves_time(
        t, lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_deg
    )

    T = 1.0 / freq if freq > 0 else 1e-9
    bound = 5.0

    fig, ax = plt.subplots(figsize=figsize)

    # 介质1区域：入射波 + 反射波 + 总电场
    mask_left = x < bound
    mask_right = x >= bound

    # 入射波（红色，虚线）
    ax.plot(x[mask_left], Ei[mask_left], color=COLOR_INCIDENT, lw=2,
            linestyle='--', alpha=0.7, label='入射波')
    # 反射波（橙色，虚线）
    ax.plot(x[mask_left], Er[mask_left], color=COLOR_REFLECTED, lw=2,
            linestyle='--', alpha=0.7, label='反射波')
    # 总电场（介质1，红色实线）
    ax.plot(x[mask_left], E_total_left[mask_left], color=COLOR_INCIDENT, lw=2.5,
            alpha=0.9, label='介质1总电场')

    # 透射波（绿色，虚线）
    ax.plot(x[mask_right], Et[mask_right], color=COLOR_TRANSMITTED, lw=2,
            linestyle='--', alpha=0.7, label='透射波')
    # 总电场（介质2，绿色实线）
    ax.plot(x[mask_right], E_total_right[mask_right], color=COLOR_TRANSMITTED, lw=2.5,
            alpha=0.9, label='介质2总电场')

    # 分界面
    ax.axvline(x=bound, color=COLOR_BOUNDARY, ls='--', lw=2.5, label='分界面')

    ax.set_xlabel("传播方向 z (m)", fontproperties=font_prop, fontsize=12)
    ax.set_ylabel("电场 E (V/m)", fontproperties=font_prop, fontsize=12)
    ax.set_title(f"入射/反射/透射波动态 (t = {t*1e9:.2f} ns, T = {T*1e9:.2f} ns, θi={theta_deg:.0f}°)",
                 fontproperties=font_prop, fontsize=13)
    ax.legend(prop=font_prop, fontsize=9, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(-amp * 2.8, amp * 2.8)

    # 区域标注
    ylim = ax.get_ylim()
    y_pos = ylim[1] * 0.88
    ax.text(bound / 2, y_pos, f'介质1 (n={n1_r:.2f})',
            ha='center', fontsize=11, fontproperties=font_prop,
            bbox=dict(boxstyle='round', facecolor='#fdebd0', alpha=0.8, edgecolor=COLOR_INCIDENT))
    ax.text(bound + (10 - bound) / 2, y_pos,
            f'介质2 (n={n2_r:.2f})', ha='center', fontsize=11,
            fontproperties=font_prop,
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8, edgecolor=COLOR_TRANSMITTED))

    plt.tight_layout()
    return fig


def create_3d_frame(t, theta_i_deg, n2, lam0, amp, er1, mur1, tand1, er2, mur2, tand2,
                    theta_r_deg, theta_t_deg, total_ref, figsize=(10, 7)):
    """创建3D箭头单帧 - 颜色与2D图对应"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X', fontproperties=font_prop)
    ax.set_ylabel('Y', fontproperties=font_prop)
    ax.set_zlabel('Z', fontproperties=font_prop)
    ax.set_title(
        f'3D箭头图 (n1=1.0, n2={n2:.2f})\\n'
        f'θi={theta_i_deg:.1f}°, θr={theta_r_deg:.1f}°, θt={theta_t_deg:.1f}°',
        fontproperties=font_prop
    )

    # 分界面
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.2, 1.2, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.25, color='lightblue', edgecolor='none')

    # 法线
    ax.plot([0, 0], [0, 0], [-0.9, 0.9], 'gray', linestyle='--', linewidth=1, alpha=0.5)

    theta_i_rad = np.radians(theta_i_deg)
    arrow_len = 0.9

    # 入射波 - 红色
    inc_dir = np.array([np.sin(theta_i_rad), 0, -np.cos(theta_i_rad)])
    start_inc = -inc_dir * arrow_len
    ax.quiver(start_inc[0], start_inc[1], start_inc[2],
              inc_dir[0], inc_dir[1], inc_dir[2],
              color=COLOR_INCIDENT, arrow_length_ratio=0.18, linewidth=2.5, label='入射波')

    # 反射波 - 橙色
    ref_dir = np.array([np.sin(theta_i_rad), 0, np.cos(theta_i_rad)])
    ax.quiver(0, 0, 0, ref_dir[0], ref_dir[1], ref_dir[2],
              color=COLOR_REFLECTED, arrow_length_ratio=0.18, linewidth=2.5, label='反射波')

    # 透射波 - 绿色
    if not total_ref:
        theta_t_rad = np.radians(theta_t_deg)
        trans_dir = np.array([np.sin(theta_t_rad), 0, -np.cos(theta_t_rad)])
        ax.quiver(0, 0, 0, trans_dir[0], trans_dir[1], trans_dir[2],
                  color=COLOR_TRANSMITTED, arrow_length_ratio=0.18, linewidth=2.5, label='透射波')

    # 角度标注
    arc_radius = 0.45
    angles_i = np.linspace(0, theta_i_rad, 30)
    arc_x_i = arc_radius * np.sin(angles_i)
    arc_z_i = arc_radius * np.cos(angles_i)
    ax.plot(arc_x_i, np.zeros_like(arc_x_i), arc_z_i, 'k-', linewidth=1)
    mid_i = theta_i_rad / 2
    ax.text(arc_radius * 1.1 * np.sin(mid_i), 0, arc_radius * 1.1 * np.cos(mid_i),
            f'{theta_i_deg:.0f}°', fontsize=9, ha='center', va='center', fontproperties=font_prop)

    angles_r = np.linspace(0, theta_i_rad, 30)
    arc_x_r = arc_radius * np.sin(angles_r)
    arc_z_r = arc_radius * np.cos(angles_r)
    ax.plot(-arc_x_r, np.zeros_like(arc_x_r), arc_z_r, 'k-', linewidth=1)
    ax.text(-arc_radius * 1.1 * np.sin(mid_i), 0, arc_radius * 1.1 * np.cos(mid_i),
            f'{theta_r_deg:.0f}°', fontsize=9, ha='center', va='center', fontproperties=font_prop)

    if not total_ref and theta_t_deg > 0:
        theta_t_rad = np.radians(theta_t_deg)
        angles_t = np.linspace(0, theta_t_rad, 30)
        arc_x_t = arc_radius * np.sin(angles_t)
        arc_z_t = -arc_radius * np.cos(angles_t)
        ax.plot(arc_x_t, np.zeros_like(arc_x_t), arc_z_t, 'k-', linewidth=1)
        mid_t = theta_t_rad / 2
        ax.text(arc_radius * 1.1 * np.sin(mid_t), 0, -arc_radius * 1.1 * np.cos(mid_t),
                f'{theta_t_deg:.0f}°', fontsize=9, ha='center', va='center', fontproperties=font_prop)

    # 介质标注
    ax.text(0, 1.0, 0.85, '介质1 (n1=1.0)', ha='center', fontsize=10,
            color=COLOR_INCIDENT, fontproperties=font_prop)
    ax.text(0, 1.0, -0.85, f'介质2 (n2={n2:.2f})', ha='center', fontsize=10,
            color=COLOR_TRANSMITTED, fontproperties=font_prop)

    ax.legend(loc='upper left', fontsize=8, prop=font_prop)
    ax.set_box_aspect([1.5, 1, 1])
    plt.tight_layout()
    return fig


def generate_gif(theta_i_deg, n2, lam0, amp, er1, mur1, tand1, er2, mur2, tand2,
                num_frames=36, periods=2, fps=10):
    """生成反射与折射GIF动画（2D三种波动态）"""
    theta_r_deg, theta_t_deg, r, t_coeff, total_ref = compute_angles(theta_i_deg, n2)

    freq = c / lam0
    T = 1.0 / freq if freq > 0 else 1e-9

    frames = []
    times = np.linspace(0, periods * T, num_frames, endpoint=False)

    progress_bar = st.progress(0, text="🎬 正在生成反射折射动画...")

    for i, t in enumerate(times):
        fig_2d = create_2d_frame_three_waves(
            t, lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_i_deg,
            theta_r_deg, theta_t_deg, total_ref, figsize=(14, 6)
        )

        buf = BytesIO()
        fig_2d.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig_2d)

        progress_bar.progress((i + 1) / num_frames,
                               text=f"🎬 生成帧 {i+1}/{num_frames}...")

    progress_bar.empty()

    gif_buf = BytesIO()
    duration_ms = int(1000 / fps)
    frames[0].save(gif_buf, format='GIF', save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=0, optimize=True)
    gif_buf.seek(0)

    return gif_buf, freq, theta_r_deg, theta_t_deg, r, t_coeff, total_ref


def run_reflection_simulation(theta_i_deg, n2, lam0, amp, er1, mur1, tand1, er2, mur2, tand2,
                              col_main, col_info, generate_animation=False,
                              num_frames=36, gif_fps=10, gif_periods=2):
    """运行反射与折射仿真"""
    try:
        theta_r_deg, theta_t_deg, r, t_coeff, total_ref = compute_angles(theta_i_deg, n2)

        if generate_animation:
            gif_buf, freq, theta_r_deg, theta_t_deg, r, t_coeff, total_ref = generate_gif(
                theta_i_deg, n2, lam0, amp, er1, mur1, tand1, er2, mur2, tand2,
                num_frames=num_frames, periods=gif_periods, fps=gif_fps
            )

            with col_main:
                st.image(gif_buf, caption=f"入射/反射/透射波动态 ({gif_periods}个周期循环, {num_frames}帧, {gif_fps}fps)",
                        use_container_width=True)
                st.download_button(label="⬇️ 下载GIF动画", data=gif_buf.getvalue(),
                                   file_name="reflection_three_waves.gif", mime="image/gif",
                                   use_container_width=True)

                # 同时显示静态3D箭头图
                fig_3d = create_3d_frame(0, theta_i_deg, n2, lam0, amp, er1, mur1, tand1,
                                          er2, mur2, tand2, theta_r_deg, theta_t_deg, total_ref)
                st.pyplot(fig_3d)
                plt.close(fig_3d)
        else:
            # 静态模式 - 显示三种波
            (x, Ei, Er, Et, E_total_left, E_total_right, freq,
             lam1, lam2, theta_t, r, t_coeff, n1_r, n2_r) = get_three_waves_time(
                0, lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_i_deg
            )

            fig_2d, ax = plt.subplots(figsize=(14, 6))
            bound = 5.0
            mask_left = x < bound
            mask_right = x >= bound

            ax.plot(x[mask_left], Ei[mask_left], color=COLOR_INCIDENT, lw=2,
                    linestyle='--', alpha=0.7, label='入射波')
            ax.plot(x[mask_left], Er[mask_left], color=COLOR_REFLECTED, lw=2,
                    linestyle='--', alpha=0.7, label='反射波')
            ax.plot(x[mask_left], E_total_left[mask_left], color=COLOR_INCIDENT, lw=2.5,
                    alpha=0.9, label='介质1总电场')
            ax.plot(x[mask_right], Et[mask_right], color=COLOR_TRANSMITTED, lw=2,
                    linestyle='--', alpha=0.7, label='透射波')
            ax.plot(x[mask_right], E_total_right[mask_right], color=COLOR_TRANSMITTED, lw=2.5,
                    alpha=0.9, label='介质2总电场')
            ax.axvline(x=bound, color=COLOR_BOUNDARY, ls='--', lw=2.5, label='分界面')

            ax.set_xlabel("传播方向 z (m)", fontproperties=font_prop, fontsize=12)
            ax.set_ylabel("电场 E (V/m)", fontproperties=font_prop, fontsize=12)
            ax.set_title(f"入射/反射/透射波 (θi={theta_i_deg:.0f}°)",
                         fontproperties=font_prop, fontsize=13)
            ax.legend(prop=font_prop, fontsize=9, loc='upper right', ncol=2)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 10)
            ax.set_ylim(-amp * 2.8, amp * 2.8)

            ylim = ax.get_ylim()
            y_pos = ylim[1] * 0.88
            ax.text(bound / 2, y_pos, f'介质1 (n={n1_r:.2f})',
                    ha='center', fontsize=11, fontproperties=font_prop,
                    bbox=dict(boxstyle='round', facecolor='#fdebd0', alpha=0.8, edgecolor=COLOR_INCIDENT))
            ax.text(bound + (10 - bound) / 2, y_pos,
                    f'介质2 (n={n2_r:.2f})', ha='center', fontsize=11,
                    fontproperties=font_prop,
                    bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8, edgecolor=COLOR_TRANSMITTED))

            plt.tight_layout()

            fig_3d = create_3d_frame(0, theta_i_deg, n2, lam0, amp, er1, mur1, tand1,
                                      er2, mur2, tand2, theta_r_deg, theta_t_deg, total_ref)

            with col_main:
                st.pyplot(fig_2d)
                plt.close(fig_2d)
                st.pyplot(fig_3d)
                plt.close(fig_3d)

        with col_info:
            st.success("✅ 反射与折射仿真完成！")
            st.info(f"""
📊 **计算结果**

**角度**
• 入射角 θi: {theta_i_deg:.1f}°
• 反射角 θr: {theta_r_deg:.1f}°
• 折射角 θt: {theta_t_deg:.1f}°

**菲涅耳系数 (TE波)**
• 反射系数 r: {np.real(r):.4f}
• 透射系数 t: {np.real(t_coeff):.4f}
• 反射率 R: {abs(r)**2:.4f}
• 透射率 T: {abs(t_coeff)**2 * (n2/n1):.4f}

**波参数**
• 频率: {freq/1e6:.1f} MHz
• λ1: {lam1:.2f} m
• λ2: {lam2:.2f} m

**介质参数**
• 介质1: n1={n1_r:.2f}, εr={er1}, μr={mur1}, tanδ={tand1}
• 介质2: n2={n2_r:.2f}, εr={er2}, μr={mur2}, tanδ={tand2}

**颜色说明**
🔴 红色 = 入射波 / 介质1总电场
🟠 橙色 = 反射波
🟢 绿色 = 透射波 / 介质2总电场
            """)

        return True
    except Exception as e:
        with col_main:
            st.error(f"❌ 反射与折射仿真失败: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False
