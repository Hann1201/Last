# propagate.py
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path
import matplotlib.pyplot as plt
c = 3e8  # 光速

# ----------------------
# 注册 SimsunExtG.ttf 字体
# ----------------------
# 使用项目目录 fonts/ 下的 simhei.ttf（兼容 Streamlit Cloud）
font_path = Path(__file__).parent / "fonts" / "simhei.ttf"
if not font_path.exists():
    raise FileNotFoundError(f"找不到字体文件: {font_path}")

fm.fontManager.addfont(str(font_path))
font_prop = fm.FontProperties(fname=str(font_path))
print(f"[字体] 使用: {font_path.name} -> {font_prop.get_name()}")


def calculate_propagate(wavelength, amplitude, er, ur, interface_pos):
    """计算电磁波传播参数"""
    k0 = 2 * np.pi / wavelength
    k = k0 * np.sqrt(er * ur)

    # 坐标
    x = np.linspace(0, 10, 600)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)

    # 电场和磁场
    eta0 = 377  # 真空阻抗
    eta = eta0 * np.sqrt(ur / er)

    # 介质中的波长
    lam_med = wavelength / np.sqrt(er * ur)
    n = np.sqrt(er * ur)

    # ========== 修正1：保证分界面处电场连续 ==========
    # 左侧（真空/区域1）：E = amplitude * sin(k0 * z)
    E = amplitude * np.sin(2 * np.pi * X / wavelength)
    H = amplitude / eta0 * np.cos(2 * np.pi * X / wavelength)

    # 右侧（介质/区域2）：需要相位匹配，保证分界面处连续
    # 分界面位置 z = interface_pos
    # 左侧在分界面处的相位：phi0 = 2*pi*interface_pos/wavelength
    # 右侧波形：E = amplitude * sin(2*pi*(z-interface_pos)/lam_med + phi0)
    # 这样 z=interface_pos 时，右侧 = amplitude*sin(phi0) = 左侧值
    mask = X >= interface_pos
    phi0 = 2 * np.pi * interface_pos / wavelength  # 分界面处的相位

    E[mask] = amplitude * np.sin(2 * np.pi * (X[mask] - interface_pos) / lam_med + phi0)
    H[mask] = amplitude / eta * np.cos(2 * np.pi * (X[mask] - interface_pos) / lam_med + phi0)

    # 传播参数
    beta = k
    vp = c / np.sqrt(er * ur)
    lam_med_value = wavelength / np.sqrt(er * ur)

    return X, Y, E, H, x, beta, vp, eta, lam_med_value, k0, eta0, n


def run_propagate_simulation(wavelength, amplitude, er, ur, interface_pos, col_main, col_info):
    """
    运行传播仿真主函数（供主文件调用）
    """
    try:
        # 计算
        X, Y, E, H, x, beta, vp, eta, lam_med, k0, eta0, n = calculate_propagate(
            wavelength, amplitude, er, ur, interface_pos
        )

        # 创建双图
        fig = create_dual_plots(X, Y, E, H, x, interface_pos, er, wavelength, eta0, eta, n, amplitude)

        # 在Streamlit中显示
        with col_main:
            st.pyplot(fig)
            plt.close(fig)

        # 显示参数信息
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

**介质参数**
• 相对介电常数 εᵣ: {er}
• 相对磁导率 μᵣ: {ur}
• 真空波长 λ₀: {wavelength:.4f} m
            """)

        return True

    except Exception as e:
        with col_main:
            st.error(f"❌ 仿真失败: {str(e)}")
        return False


def create_dual_plots(X, Y, E, H, x, interface_pos, er, wavelength, eta0, eta, n, amplitude):
    """创建双图：左侧3D图，右侧波形图"""

    # 获取第一行的电场值用于2D图
    E_1d = E[0, :]
    H_1d = H[0, :]

    fig = plt.figure(figsize=(16, 7))

    # ========== 左图：3D传播图 ==========
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')

    # 3D螺旋线 - 电场（蓝色）和磁场（橙色）
    z_3d = np.linspace(0, 10, 200)
    k0 = 2 * np.pi / wavelength
    lam_med = wavelength / n

    # 左侧（真空）
    z_left = z_3d[z_3d <= interface_pos]
    E_left_3d = amplitude * np.sin(k0 * z_left)
    H_left_3d = (amplitude / eta0) * np.cos(k0 * z_left)

    # 右侧（介质）
    z_right = z_3d[z_3d > interface_pos]
    phi0 = k0 * interface_pos
    k_med = 2 * np.pi / lam_med
    E_right_3d = amplitude * np.sin(k_med * (z_right - interface_pos) + phi0)
    H_right_3d = (amplitude / eta) * np.cos(k_med * (z_right - interface_pos) + phi0)

    # 绘制3D电场（蓝色）
    ax3d.plot(z_left, E_left_3d, np.zeros_like(z_left), 'b-', linewidth=2, label='电场 E')
    ax3d.plot(z_right, E_right_3d, np.zeros_like(z_right), 'b-', linewidth=2)

    # 绘制3D磁场（橙色）
    ax3d.plot(z_left, np.zeros_like(z_left), H_left_3d, color='orange', linewidth=2, label='磁场 H')
    ax3d.plot(z_right, np.zeros_like(z_right), H_right_3d, color='orange', linewidth=2)

    # 绘制分界面
    y_plane = np.linspace(-amplitude*1.2, amplitude*1.2, 10)
    z_plane = np.linspace(-amplitude/eta0*1.5, amplitude/eta0*1.5, 10)
    Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
    X_plane = np.full_like(Y_plane, interface_pos)
    ax3d.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='red')
    ax3d.plot([interface_pos, interface_pos], [0, 0], [0, 0], 'r--', linewidth=2, label='分界面')

    ax3d.set_xlabel('传播方向 z (m)', fontproperties=font_prop)
    ax3d.set_ylabel('电场 E (V/m)', fontproperties=font_prop)
    ax3d.set_zlabel('磁场 H (A/m)', fontproperties=font_prop)
    ax3d.set_title(f'电磁波3D传播 (εr = {er}, n = {n:.2f})', fontsize=12, fontproperties=font_prop)
    ax3d.legend(loc='upper right', fontsize=10, prop=font_prop)

    # 设置坐标范围
    ax3d.set_xlim(0, 10)
    ax3d.set_ylim(-amplitude*1.2, amplitude*1.2)
    ax3d.set_zlim(-amplitude/eta0*1.5, amplitude/eta0*1.5)

    # ========== 右图：2D波形图（电场+磁场）==========
    ax2 = fig.add_subplot(1, 2, 2)

    # ========== 修正2：使用双Y轴显示电场和磁场 ==========
    # 电场用左侧Y轴（蓝色）
    ax2_e = ax2
    line_e, = ax2_e.plot(x, E_1d, 'b-', linewidth=2.5, label='电场强度 E(z)', alpha=0.8)
    ax2_e.set_xlabel('传播方向 z (m)', fontsize=12, fontproperties=font_prop)
    ax2_e.set_ylabel('电场强度 E (V/m)', fontsize=12, fontproperties=font_prop, color='blue')
    ax2_e.tick_params(axis='y', labelcolor='blue')
    ax2_e.set_ylim(-amplitude * 1.2, amplitude * 1.2)

    # 磁场用右侧Y轴（红色）
    ax2_h = ax2_e.twinx()
    line_h, = ax2_h.plot(x, H_1d, 'r-', linewidth=2, label='磁场强度 H(z)', alpha=0.8)
    ax2_h.set_ylabel('磁场强度 H (A/m)', fontsize=12, fontproperties=font_prop, color='red')
    ax2_h.tick_params(axis='y', labelcolor='red')

    # 磁场振幅 = E/η，设置合适的范围
    h_max = amplitude / eta0  # 真空中的H振幅
    ax2_h.set_ylim(-h_max * 1.5, h_max * 1.5)

    # 绘制分界线
    ax2_e.axvline(interface_pos, color='k', linestyle='--', linewidth=2, label='介质分界面')

    # 添加区域标注
    ylim_e = ax2_e.get_ylim()
    y_pos = ylim_e[1] * 0.85

    ax2_e.text(interface_pos / 2, y_pos, '区域1 (真空)',
             ha='center', fontsize=11, fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax2_e.text(interface_pos + (x.max() - interface_pos) / 2, y_pos,
             f'区域2 (介质 εr={er})', ha='center', fontsize=11,
             fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # 合并图例
    lines = [line_e, line_h]
    labels = [l.get_label() for l in lines]
    ax2_e.legend(lines, labels, loc='upper right', fontsize=10, prop=font_prop)
    ax2_e.grid(True, alpha=0.3, linestyle='--')

    # 标题
    ax2_e.set_title(f'电磁波传播波形 (εr = {er}, n = {n:.2f})',
                   fontsize=12, fontproperties=font_prop)

    plt.tight_layout()
    return fig


# 需要导入streamlit（因为run_propagate_simulation中使用了st）
import streamlit as st