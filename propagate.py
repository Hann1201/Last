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


def calculate_propagate(wavelength, amplitude, er, interface_pos):
    """计算电磁波传播参数"""
    k0 = 2 * np.pi / wavelength
    k = k0 * np.sqrt(er)

    # 坐标
    x = np.linspace(0, 10, 600)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)

    # 电场和磁场
    eta0 = 377  # 真空阻抗
    eta = eta0 / np.sqrt(er)
    E = amplitude * np.sin(2 * np.pi * X / wavelength)
    H = amplitude / eta * np.cos(2 * np.pi * X / wavelength)

    # 在介质中修改波长
    lam_med = wavelength / np.sqrt(er)
    mask = X >= interface_pos
    E[mask] = amplitude * np.sin(2 * np.pi * X[mask] / lam_med)
    H[mask] = amplitude / eta * np.cos(2 * np.pi * X[mask] / lam_med)

    # 传播参数
    beta = k
    vp = c / np.sqrt(er)
    lam_med_value = wavelength / np.sqrt(er)

    return X, Y, E, H, x, beta, vp, eta, lam_med_value, k0


def run_propagate_simulation(wavelength, amplitude, er, interface_pos, col_main, col_info):
    """
    运行传播仿真主函数（供主文件调用）
    """
    try:
        # 计算
        X, Y, E, H, x, beta, vp, eta, lam_med, k0 = calculate_propagate(
            wavelength, amplitude, er, interface_pos
        )

        # 创建双图
        fig = create_dual_plots(X, Y, E, H, x, interface_pos, er, wavelength)

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
• 折射率 n: {np.sqrt(er):.3f}

**介质参数**
• 相对介电常数 εᵣ: {er}
• 真空波长 λ₀: {wavelength:.4f} m
            """)

        return True

    except Exception as e:
        with col_main:
            st.error(f"❌ 仿真失败: {str(e)}")
        return False


def create_dual_plots(X, Y, E, H, x, interface_pos, er, wavelength):
    """创建双图：左侧3D图，右侧波形图"""

    # 获取第一行的电场值用于2D图
    E_1d = E[0, :]
    H_1d = H[0, :]

    fig = plt.figure(figsize=(16, 7))


    # ========== 右图：2D波形图（电场+磁场）==========
    ax2 = fig.add_subplot(1, 2, 2)

    # 绘制电场和磁场
    ax2.plot(x, E_1d, 'b-', linewidth=2.5, label='电场强度 E(z)', alpha=0.8)
    ax2.plot(x, H_1d, 'r-', linewidth=2, label='磁场强度 H(z)', alpha=0.8)

    # 绘制分界线
    ax2.axvline(interface_pos, color='k', linestyle='--', linewidth=2, label='介质分界面')

    # 添加区域标注
    ylim = ax2.get_ylim()
    y_pos = ylim[1] * 0.9

    ax2.text(interface_pos / 2, y_pos, '区域1 (真空)',
             ha='center', fontsize=11, fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax2.text(interface_pos + (x.max() - interface_pos) / 2, y_pos,
             f'区域2 (介质 εr={er})', ha='center', fontsize=11,
             fontproperties=font_prop,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax2.set_xlabel('传播方向 z (m)', fontsize=12, fontproperties=font_prop)
    ax2.set_ylabel('场强度 (V/m, A/m)', fontsize=12, fontproperties=font_prop)
    ax2.set_title(f'电磁波传播波形 (εr = {er})', fontsize=12, fontproperties=font_prop)
    ax2.legend(loc='upper right', fontsize=10, prop=font_prop)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 设置y轴范围
    max_val = max(abs(E_1d.max()), abs(E_1d.min()), abs(H_1d.max()), abs(H_1d.min()))
    ax2.set_ylim(-max_val * 1.2, max_val * 1.2)

    plt.tight_layout()
    return fig


# 需要导入streamlit（因为run_propagate_simulation中使用了st）
import streamlit as st