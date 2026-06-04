from pathlib import Path


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from pathlib import Path
import streamlit as st

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


def get_wave_2d(lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_deg):
    """计算2D波形数据"""
    freq = c / lam0
    k0 = 2 * np.pi / lam0
    theta = np.radians(theta_deg)
    bound = 5.0
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
    t = 2 * eta2 * cos1 / (eta2 * cos1 + eta1 * cos2)

    k1x = k1 * cos1
    k2x = k2 * cos2

    Ei = np.zeros_like(x, dtype=complex)
    Er = np.zeros_like(x, dtype=complex)
    Et = np.zeros_like(x, dtype=complex)

    idx_left = x < bound
    idx_right = x >= bound

    Ei[idx_left] = amp * np.exp(1j * k1x * x[idx_left])
    Er[idx_left] = r * amp * np.exp(-1j * k1x * x[idx_left])
    Et[idx_right] = t * amp * np.exp(1j * k2x * (x[idx_right] - bound))

    lam1 = np.real(lam0 / n1_c)
    lam2 = np.real(lam0 / n2_c)
    return x, np.real(Ei), np.real(Er), np.real(Et), freq, lam1, lam2, theta_t, r, t, np.real(n1_c), np.real(n2_c)


def create_2d_plot(lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_deg):
    """创建2D波形图"""
    x, Ei, Er, Et, freq, lam1, lam2, theta_t, r, t, n1_r, n2_r = get_wave_2d(
        lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_deg
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, Ei, 'b', lw=1.2, label='入射波')
    ax.plot(x, Er, 'r', lw=1.2, label='反射波')
    ax.plot(x, Et, 'g', lw=1.2, label='透射波')
    ax.axvline(x=5.0, c='k', ls='--', lw=2, label='分界面')
    ax.set_xlabel("X (m)", fontproperties=font_prop)
    ax.set_ylabel("电场 E", fontproperties=font_prop)
    ax.set_title(f"2D波形图   入射角 {theta_deg:.0f}°", fontproperties=font_prop)
    ax.legend(prop=font_prop)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, freq, lam1, lam2, theta_t, r, t, n1_r, n2_r


def create_3d_plot(theta_i_deg, n2):
    """创建3D箭头图"""
    theta_r_deg, theta_t_deg, r, t, total_ref = compute_angles(theta_i_deg, n2)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X', fontproperties=font_prop)
    ax.set_ylabel('Y', fontproperties=font_prop)
    ax.set_zlabel('Z', fontproperties=font_prop)
    ax.set_title(
        f'3D箭头图 (n1=1.0, n2={n2:.2f})\n'
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

    # 入射波
    inc_dir = np.array([np.sin(theta_i_rad), 0, -np.cos(theta_i_rad)])
    start_inc = -inc_dir * arrow_len
    ax.quiver(start_inc[0], start_inc[1], start_inc[2],
              inc_dir[0], inc_dir[1], inc_dir[2],
              color='red', arrow_length_ratio=0.18, linewidth=2, label='入射波')

    # 反射波
    ref_dir = np.array([np.sin(theta_i_rad), 0, np.cos(theta_i_rad)])
    ax.quiver(0, 0, 0, ref_dir[0], ref_dir[1], ref_dir[2],
              color='orange', arrow_length_ratio=0.18, linewidth=2, label='反射波')

    # 透射波
    if not total_ref:
        theta_t_rad = np.radians(theta_t_deg)
        trans_dir = np.array([np.sin(theta_t_rad), 0, -np.cos(theta_t_rad)])
        ax.quiver(0, 0, 0, trans_dir[0], trans_dir[1], trans_dir[2],
                  color='green', arrow_length_ratio=0.18, linewidth=2, label='透射波')

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
    ax.text(0, 1.0, 0.85, '介质1 (n1=1.0)', ha='center', fontsize=10, color='darkblue', fontproperties=font_prop)
    ax.text(0, 1.0, -0.85, f'介质2 (n2={n2:.2f})', ha='center', fontsize=10, color='darkgreen', fontproperties=font_prop)

    ax.legend(loc='upper left', fontsize=8, prop=font_prop)
    ax.set_box_aspect([1.5, 1, 1])
    plt.tight_layout()
    return fig


def run_reflection_simulation(theta_i_deg, n2, lam0, amp, er1, mur1, tand1, er2, mur2, tand2,
                              col_main, col_info):
    """运行反射与折射仿真（2D波形 + 3D箭头）"""
    try:
        # 2D波形图
        fig_2d, freq, lam1, lam2, theta_t, r, t, n1_r, n2_r = create_2d_plot(
            lam0, amp, er1, mur1, tand1, er2, mur2, tand2, theta_i_deg
        )

        # 3D箭头图
        fig_3d = create_3d_plot(theta_i_deg, n2)

        with col_main:
            # 2D波形图在上，3D箭头图在下
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
• 反射角 θr: {theta_i_deg:.1f}°
• 折射角 θt: {np.degrees(theta_t):.1f}°

**菲涅耳系数 (TE波)**
• 反射系数 r: {np.real(r):.4f}
• 透射系数 t: {np.real(t):.4f}
• 反射率 R: {abs(r)**2:.4f}
• 透射率 T: {abs(t)**2 * (n2/n1):.4f}

**波参数**
• 频率: {freq/1e6:.1f} MHz
• λ1: {lam1:.2f} m
• λ2: {lam2:.2f} m

**介质参数**
• 介质1: n1={n1_r:.2f}, εr={er1}, μr={mur1}, tanδ={tand1}
• 介质2: n2={n2_r:.2f}, εr={er2}, μr={mur2}, tanδ={tand2}
            """)

        return True
    except Exception as e:
        with col_main:
            st.error(f"❌ 反射与折射仿真失败: {str(e)}")
        return False
