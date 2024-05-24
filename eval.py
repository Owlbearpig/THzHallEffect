from scipy.constants import e, m_e, c
import numpy as np
import matplotlib.pyplot as plt
from parse_data import ref1_magnet_fd, ref2_magnet_fd, sam1_fd_p, sam2_fd_p

B = 0.700  # T
m_eff = m_e * 0.067  # kg
w_c = e * B / m_eff  # Hz
n = 3.5  # + 1j*0.1
d = 500 * 1e-6  # m
Z0 = 377
c_thz = c  # m / s
t_sub_air = 2 * n / (1 + n)


def calc_grid(ref_x_fd, ref_y_fd, sam_x_fd, sam_y_fd, omega):
    Tx_exp, Ty_exp = sam_x_fd / ref_x_fd, sam_y_fd / ref_y_fd
    rez_x, rez_y = int(1e3), int(1e3)
    t_line = np.linspace(1, 500, rez_x) * 1e-15  # s
    N_line = np.linspace(1, 20, rez_y) * 1e17  # 1 / cm^3

    #sig_0 = N_line[0] * e ** 2 * t_line[0] / m_eff
    #print(sig_0)
    #exit()

    P = np.exp(-1j * n * d * omega / c_thz) * t_sub_air
    P = 0.85
    K = ref_y_fd / ref_x_fd
    K = 4
    grid_arr = np.zeros((rez_x, rez_y))

    def func(N_, t_):
        sig_0 = N_ * e ** 2 * t_ / m_eff
        a, b = 1 - 1j * omega * t_, w_c * t_
        sig_xx = sig_0 * a / (a ** 2 + b ** 2)
        sig_xy = sig_xx * b / a

        T_d = (1 + n + Z0 * sig_xx) ** 2 + (Z0 * sig_xy) ** 2

        Tx_e = 1 + n + Z0 * sig_xx + Z0 * sig_xy * K
        Ty_e = 1 + n + Z0 * sig_xx - Z0 * sig_xy * (1 / K)

        Tx = 2 * P * Tx_e / T_d
        Ty = 2 * P * Ty_e / T_d

        return np.abs(Tx - Tx_exp)**2 + np.abs(Ty - Ty_exp)**2

    for i, t in enumerate(t_line):
        print(f"{np.round((i / rez_x) * 100, 2)} % done")
        for j, N in enumerate(N_line):
            val = func(N, t)
            grid_arr[i, j] = val
        print(val)

    argmin = np.argmin(grid_arr)
    (min_t_idx, min_N_idx) = np.unravel_index(argmin, grid_arr.shape)
    print(grid_arr[min_t_idx, min_N_idx])
    print(t_line[min_t_idx], N_line[min_N_idx])

    grid_arr = grid_arr.T
    plt.imshow(grid_arr, aspect='auto', extent=[t_line[0], t_line[-1], N_line[0], N_line[-1]], origin="lower")
    plt.xlabel(r"$\tau$ (s)")
    plt.ylabel(r"N (1/$cm^3$)")


def main():
    selected_freq = 0.35  # THz
    freq_axis = ref1_magnet_fd[:, 0] * 1e12
    freq_idx = np.argmin(np.abs(selected_freq * 1e12 - freq_axis), axis=0)

    omega = 2 * np.pi * freq_axis[freq_idx]
    Ex_in, Ey_in = ref1_magnet_fd[freq_idx, 1], ref2_magnet_fd[freq_idx, 1]
    Ex_out, Ey_out = sam1_fd_p[freq_idx, 1], sam2_fd_p[freq_idx, 1]

    calc_grid(Ex_in, Ey_in, Ex_out, Ey_out, omega)


if __name__ == '__main__':
    main()
    plt.show()
