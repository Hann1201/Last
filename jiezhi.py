import numpy as np
import matplotlib.pyplot as plt

# 增加 font_prop 参数
def run_media_propagation_simulation(amplitude, interface_pos, lambda1, er1, ur1, er2, ur2, font_prop):
    c = 3e8
    f = c / lambda1
    omega = 2 * np.pi * f

    v1 = c / np.sqrt(er1 * ur1)
    v2 = c / np.sqrt(er2 * ur2)
    lambda2 = lambda1 * (v2 / v1)

    k1 = 2 * np.pi / lambda1
    k2 = 2 * np.pi / lambda2

    eta1 = 377 * np.sqrt(ur1 / er1)
    eta2 = 377 * np.sqrt(ur2 / er2)
    Gamma = (eta2 - eta1) / (eta2 + eta1)
    tau = 2 * eta2 / (eta2 + eta1)

    x_total = np.linspace(0, 10, 1000)
    t = 0

    E_left = np.zeros_like(x_total)
    H_left = np.zeros_like(x_total)
    E_right = np.zeros_like(x_total)
    H_right = np.zeros_like(x_total)

    for i, x in enumerate(x_total):
        if x <= interface_pos:
            phase_i = -k1 * x + omega * t
            phase_r = k1 * x + omega * t
            E_i = amplitude * np.cos(phase_i)
            E_r = amplitude * Gamma * np.cos(phase_r)
            E_left[i] = E_i + E_r

            H_i = (amplitude / eta1) * np.cos(phase_i)
            H_r = (-amplitude * Gamma / eta1) * np.cos(phase_r)
            H_left[i] = H_i + H_r
        else:
            phase_t = -k2 * x + omega * t
            E_t = amplitude * tau * np.cos(phase_t)
            E_right[i] = E_t
            H_right[i] = (amplitude * tau / eta2) * np.cos(phase_t)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_total, E_left, label='电场 E(z)', color='blue')
    ax.plot(x_total, H_left, label='磁场 H(z)', color='red')
    ax.plot(x_total[x_total > interface_pos], E_right[x_total > interface_pos], color='blue')
    ax.plot(x_total[x_total > interface_pos], H_right[x_total > interface_pos], color='red')

    ax.axvline(x=interface_pos, color='green', linestyle='--', label='介质分界面')

    # 所有中文都加 fontproperties=font_prop
    ax.set_xlabel('位置 z (m)', fontproperties=font_prop)
    ax.set_ylabel('场强度', fontproperties=font_prop)
    ax.set_title(f'介质间电磁波传播 (εᵣ₁={er1}, μᵣ₁={ur1} → εᵣ₂={er2}, μᵣ₂={ur2})', fontproperties=font_prop)
    
    # 图例字体
    ax.legend(prop=font_prop)
    ax.grid(True)

    info = {
        '频率 f (Hz)': f,
        '介质1波长 λ₁ (m)': lambda1,
        '介质2波长 λ₂ (m)': lambda2,
        '反射系数 Γ': Gamma,
        '透射系数 τ': tau
    }
    return fig, info
