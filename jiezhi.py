# jiezhi1_jiezhi2.py
# 介质1到介质2的电磁波传播仿真模块

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

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

        # 创建图形
        fig = plt.figure(figsize=(16, 7))

        # 获取第一行数据
        E_1d = E[0, :]
        H_1d = H[0, :]

        # ========== 左图：3D 曲线图 ==========
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        ax1.plot(x, np.zeros_like(x), E_1d, lw=2.5, c='#1f77b4', label='电场E')
        ax1.plot(x, H_1d, np.zeros_like(x), lw=2.5, c='#ff6b35', label='磁场H')
        ax1.plot([interface_pos, interface_pos], [-2, 2], [-2, 2], 'r--', lw=2, label='介质分界面')

        ax1.set_title(f'电磁波跨介质传播 f={f / 1e6:.1f}MHz', fontsize=12)
        ax1.set_xlabel('传播方向x', fontsize=10)
        ax1.set_ylabel('磁场H', fontsize=10)
        ax1.set_zlabel('电场E', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.view_init(elev=25, azim=-60)


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