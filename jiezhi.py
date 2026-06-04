# jiezhi1_jiezhi2.py
# 介质1到介质2的电磁波传播仿真模块

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import matplotlib.font_manager as fm
from pathlib import Path

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

# 物理常量
c = 3e8


def calculate_media_propagation(frequency, wavelength, amplitude, er1, ur1, er2, ur2, interface_pos):
    """
    计算电磁波从介质1到介质2的传播参数
    """
    # 计算频率和波长
    if frequency is not None and frequency > 0:
        f = frequency
        lam0 = c / f
    else:
        lam0 = wavelength
        f = c / lam0

    # 介质1参数
    v1 = c / np.sqrt(er1 * ur1)
    lam1 = v1 / f
    k1 = 2 * np.pi / lam1
    eta1 = 377 * np.sqrt(ur1 / er1)

    # 介质2参数
    v2 = c / np.sqrt(er2 * ur2)
    lam2 = v2 / f
    k2 = 2 * np.pi / lam2
    eta2 = 377 * np.sqrt(ur2 / er2)

    # 坐标
    x = np.linspace(0, 10, 600)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)

    # 计算电场和磁场
    E = np.zeros_like(X)
    H = np.zeros_like(X)

    mask1 = X < interface_pos
    mask2 = X >= interface_pos

    # 介质1中的波
    E[mask1] = amplitude * np.sin(k1 * X[mask1])
    H[mask1] = amplitude / eta1 * np.cos(k1 * X[mask1])

    # 介质2中的波（相位连续）
    phase_at_boundary = k1 * interface_pos
    E[mask2] = amplitude * np.sin(k2 * (X[mask2] - interface_pos) + phase_at_boundary)
    H[mask2] = amplitude / eta2 * np.cos(k2 * (X[mask2] - interface_pos) + phase_at_boundary)

    return X, Y, E, H, x, lam0, lam1, lam2, v1, v2, eta1, eta2, k1, k2, f


def run_media_propagation_simulation(frequency, wavelength, amplitude, er1, ur1, er2, ur2,
                                     interface_pos, col_main, col_info):
    """
    运行介质间传播仿真主函数
    """
    try:
        # 参数验证
        if frequency is None and wavelength is None:
            raise ValueError("请设置频率或波长")

        # 计算
        X, Y, E, H, x, lam0, lam1, lam2, v1, v2, eta1, eta2, k1, k2, f = calculate_media_propagation(
            frequency, wavelength, amplitude, er1, ur1, er2, ur2, interface_pos
        )

        # 创建图形 - 缩小尺寸
        fig = plt.figure(figsize=(12, 5))

        # 获取第一行数据
        E_1d = E[0, :]
        H_1d = H[0, :]

        # 计算合适的Y轴范围
        H_max = np.max(np.abs(H_1d)) if len(H_1d) > 0 else 1
        H_range = [-H_max * 1.2, H_max * 1.2]
        E_range = [-amplitude * 1.2, amplitude * 1.2]

        # ========== 左图：3D 曲线图 ==========
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        # 绘制电场线 (沿Z轴)
        ax1.plot(x, np.zeros_like(x), E_1d, lw=2.5, c='#1f77b4', label='电场 E')
        # 绘制磁场线 (沿Y轴)
        ax1.plot(x, H_1d, np.zeros_like(x), lw=2.5, c='#ff6b35', label='磁场 H')

        # 绘制介质分界面 - 调整显示范围
        ax1.plot([interface_pos, interface_pos], H_range, E_range, 'r--', lw=2, label='介质分界面')

        # 设置标题和标签
        ax1.set_title(f'电磁波跨介质传播 f={f / 1e6:.1f}MHz', fontsize=11, fontproperties=font_prop)
        ax1.set_xlabel('x (m)', fontsize=9, fontproperties=font_prop, labelpad=8)
        ax1.set_ylabel('磁场 H (A/m)', fontsize=9, fontproperties=font_prop, labelpad=8)
        ax1.set_zlabel('电场 E (V/m)', fontsize=9, fontproperties=font_prop, labelpad=8)

        # 设置图例
        ax1.legend(loc='upper right', prop=font_prop, fontsize=8)

        # 调整视角
        ax1.view_init(elev=25, azim=-60)

        # 设置坐标轴范围
        ax1.set_xlim([0, 10])
        ax1.set_ylim(H_range)
        ax1.set_zlim(E_range)

        # ========== 右图：2D波形对比图 ==========
        ax2 = fig.add_subplot(1, 2, 2)

        # 绘制电场和磁场随位置的变化
        ax2.plot(x, E_1d, 'b-', lw=2, label='电场 E(x)')
        ax2.plot(x, H_1d, 'orange', lw=2, label='磁场 H(x)')
        ax2.axvline(x=interface_pos, color='r', linestyle='--', lw=2, label='介质分界面')
        ax2.axhline(y=0, color='k', linestyle='-', lw=0.5, alpha=0.5)

        ax2.set_xlabel('x (m)', fontsize=10, fontproperties=font_prop)
        ax2.set_ylabel('场强', fontsize=10, fontproperties=font_prop)
        ax2.set_title('电场与磁场沿传播方向变化', fontsize=11, fontproperties=font_prop)
        ax2.legend(loc='upper right', prop=font_prop, fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 10])

        plt.tight_layout()

        # 显示图形
        with col_main:
            st.pyplot(fig)
            plt.close(fig)

        # 计算反射系数
        gamma = (eta2 - eta1) / (eta2 + eta1)
        tau = 2 * eta2 / (eta2 + eta1)

        # 显示参数
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
        return False