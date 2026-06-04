import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from pathlib import Path
import streamlit as st

# ----------------------
# 注册字体（与主文件一致）
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
n1 = 1.0  # 介质1折射率


def compute_angles(theta_i_deg, n2):
    """计算反射角、折射角和反射/透射系数 (TE波)"""
    theta_i = np.radians(theta_i_deg)
    theta_r_deg = theta_i_deg
    sin_theta_t = (n1 / n2) * np.sin(theta_i)

    if abs(sin_theta_t) >= 1.0:
        theta_t_deg = 90.0 if sin_theta_t > 0 else -90.0
        total_reflection = True
    else:
        theta_t_deg = np.degrees(np.arcsin(sin_theta_t))
        total_reflection = False

    cos_i = np.cos(theta_i)
    cos_t = np.cos(np.radians(theta_t_deg))
    denom = n1 * cos_i + n2 * cos_t

    if denom != 0 and not total_reflection:
        r = (n1 * cos_i - n2 * cos_t) / denom
        t = 2 * n1 * cos_i / denom
    else:
        r = 1.0
        t = 0.0

    return theta_r_deg, theta_t_deg, r, t, total_reflection


def create_reflection_plot(theta_i_deg, n2):
    """创建反射与折射3D图"""
    theta_r_deg, theta_t_deg, r, t, total_ref = compute_angles(theta_i_deg, n2)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.2, 1.5])
    ax.set_xlabel('X', fontproperties=font_prop)
    ax.set_ylabel('Y', fontproperties=font_prop)
    ax.set_zlabel('Z', fontproperties=font_prop)
    ax.set_title(
        f'电磁波反射与折射 (n1=1.0, n2={n2:.2f})\n'
        f'θi={theta_i_deg:.1f}°, θr={theta_r_deg:.1f}°, θt={theta_t_deg:.1f}°\n'
        f'r={r:.3f}, t={t:.3f}',
        fontproperties=font_prop
    )

    # 绘制分界面 (z=0平面)
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.3, color='lightblue', edgecolor='none')

    # 绘制法线
    ax.quiver(0, 0, -0.8, 0, 0, 1.6, color='gray', alpha=0.5, arrow_length_ratio=0.1, linestyle='dashed')

    # 计算方向向量
    theta_i_rad = np.radians(theta_i_deg)
    inc_dir = np.array([-np.sin(theta_i_rad), 0, -np.cos(theta_i_rad)])
    ref_dir = np.array([np.sin(theta_i_rad), 0, np.cos(theta_i_rad)])

    if not total_ref:
        theta_t_rad = np.radians(theta_t_deg)
        trans_dir = np.array([np.sin(theta_t_rad), 0, -np.cos(theta_t_rad)])
    else:
        trans_dir = np.array([0, 0, 0])

    # 箭头长度
    arrow_len = 0.8

    # 入射波
    start_inc = -inc_dir * arrow_len
    ax.quiver(start_inc[0], start_inc[1], start_inc[2],
              inc_dir[0], inc_dir[1], inc_dir[2],
              color='red', arrow_length_ratio=0.2, linewidth=2, label='入射波')

    # 反射波
    ax.quiver(0, 0, 0, ref_dir[0], ref_dir[1], ref_dir[2],
              color='orange', arrow_length_ratio=0.2, linewidth=2, label='反射波')

    # 透射波
    if not total_ref and np.linalg.norm(trans_dir) > 0:
        ax.quiver(0, 0, 0, trans_dir[0], trans_dir[1], trans_dir[2],
                  color='green', arrow_length_ratio=0.2, linewidth=2, label='透射波')

    # 标注角度 (圆弧)
    arc_radius = 0.4

    # 入射角圆弧
    angles_i = np.linspace(0, theta_i_rad, 20)
    arc_x = arc_radius * np.sin(angles_i)
    arc_z = arc_radius * np.cos(angles_i)
    ax.plot(arc_x, np.zeros_like(arc_x), arc_z, color='black', linewidth=1)
    mid_angle = theta_i_rad / 2
    text_x = arc_radius * 1.1 * np.sin(mid_angle)
    text_z = arc_radius * 1.1 * np.cos(mid_angle)
    ax.text(text_x, 0, text_z, f'{theta_i_deg:.0f}°', fontsize=9, ha='center', va='center', fontproperties=font_prop)

    # 反射角圆弧
    angles_r = np.linspace(0, theta_i_rad, 20)
    arc_x_r = arc_radius * np.sin(angles_r)
    arc_z_r = arc_radius * np.cos(angles_r)
    ax.plot(arc_x_r, np.zeros_like(arc_x_r), arc_z_r, color='black', linewidth=1)
    text_x_r = arc_radius * 1.1 * np.sin(mid_angle)
    text_z_r = arc_radius * 1.1 * np.cos(mid_angle)
    ax.text(text_x_r, 0, text_z_r, f'{theta_r_deg:.0f}°', fontsize=9, ha='center', va='center', fontproperties=font_prop)

    # 折射角圆弧
    if not total_ref and theta_t_deg > 0:
        theta_t_rad = np.radians(theta_t_deg)
        angles_t = np.linspace(0, theta_t_rad, 20)
        arc_x_t = arc_radius * np.sin(angles_t)
        arc_z_t = -arc_radius * np.cos(angles_t)
        ax.plot(arc_x_t, np.zeros_like(arc_x_t), arc_z_t, color='black', linewidth=1)
        mid_t = theta_t_rad / 2
        text_x_t = arc_radius * 1.1 * np.sin(mid_t)
        text_z_t = -arc_radius * 1.1 * np.cos(mid_t)
        ax.text(text_x_t, 0, text_z_t, f'{theta_t_deg:.0f}°', fontsize=9, ha='center', va='center', fontproperties=font_prop)

    # 介质标注
    ax.text(0, 1.2, 0.8, '介质1 (n1=1.0)', ha='center', fontsize=10, color='darkblue', fontproperties=font_prop)
    ax.text(0, 1.2, -0.8, f'介质2 (n2={n2:.2f})', ha='center', fontsize=10, color='darkgreen', fontproperties=font_prop)

    # 图例
    ax.legend(loc='upper left', fontsize=8, prop=font_prop)

    # 设置视角
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    return fig, theta_r_deg, theta_t_deg, r, t, total_ref


def run_reflection_simulation(theta_i_deg, n2, col_main, col_info):
    """运行反射与折射仿真"""
    try:
        fig, theta_r, theta_t, r, t, total_ref = create_reflection_plot(theta_i_deg, n2)

        with col_main:
            st.pyplot(fig)
            plt.close(fig)

        with col_info:
            st.success("✅ 反射与折射仿真完成！")
            st.info(f"""
📊 **计算结果**

**角度**
• 入射角 θi: {theta_i_deg:.1f}°
• 反射角 θr: {theta_r:.1f}°
• 折射角 θt: {theta_t:.1f}°

**菲涅耳系数 (TE波)**
• 反射系数 r: {r:.4f}
• 透射系数 t: {t:.4f}
• 反射率 R: {abs(r)**2:.4f}
• 透射率 T: {abs(t)**2 * (n2/n1):.4f}

**介质参数**
• 介质1折射率 n1: 1.0
• 介质2折射率 n2: {n2:.2f}

**状态**
• {'⚠️ 全反射！' if total_ref else '✅ 正常折射'}
            """)

        return True
    except Exception as e:
        with col_main:
            st.error(f"❌ 反射与折射仿真失败: {str(e)}")
        return False
