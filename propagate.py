import numpy as np
import matplotlib.pyplot as plt

# 全局强制中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def run_propagate_simulation(amplitude, interface_pos, lambda0, er2):
    c = 3e8
    f = c / lambda0
    omega = 2 * np.pi * f

    k0 = 2 * np.pi / lambda0
    v2 = c / np.sqrt(er2)
    lambda2 = lambda0 / np.sqrt(er2)
    k2 = 2 * np.pi / lambda2

    eta0 = 377
    eta2 = 377 / np.sqrt(er2)
    Gamma = (eta2 - eta0) / (eta2 + eta0)
    tau = 2 * eta2 / (eta2 + eta0)

    x_total = np.linspace(0, 10, 1000)
    t = 0

    E_left = np.zeros_like(x_total)
    H_left = np.zeros_like(x_total)
    E_right = np.zeros_like(x_total)
    H_right = np.zeros_like(x_total)

    for i, x in enumerate(x_total):
        if x <= interface_pos:
            phase_i = -k0 * x + omega * t
            phase_r = k0 * x + omega * t
            E_i = amplitude * np.cos(phase_i)
            E_r = amplitude * Gamma * np.cos(phase_r)
            E_left[i] = E_i + E_r

            H_i = (amplitude / eta0) * np.cos(phase_i)
            H_r = (-amplitude * Gamma / eta0) * np.cos(phase_r)
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

    ax.set_xlabel('位置 z (m)')
    ax.set_ylabel('场强度')
    ax.set_title(f'电磁波传播 (εᵣ = {er2})')
    ax.legend()
    ax.grid(True)

    info = {
        '频率 f (Hz)': f,
        '真空波长 λ₀ (m)': lambda0,
        '介质波长 λ (m)': lambda2,
        '相速度 v (m/s)': v2,
        '波阻抗 η (Ω)': eta2,
        '反射系数 Γ': Gamma,
        '透射系数 τ': tau
    }
    return fig, info
