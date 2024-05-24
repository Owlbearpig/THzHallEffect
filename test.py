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

P = 0.85
K = 4

N_ = 1e17
t_ = 40 * 1e-15
omega = 2*np.pi*0.35*1e12

sig_0 = N_ * e ** 2 * t_ / m_eff
a, b = 1 - 1j * omega * t_, w_c * t_
sig_xx = sig_0 * a / (a ** 2 + b ** 2)
sig_xy = sig_xx * b / a
# print(sig_xy, sig_xx)
T_d = (1 + n + Z0 * sig_xx) ** 2 + (Z0 * sig_xy) ** 2

Tx_e = 1 + n + Z0 * sig_xx + Z0 * sig_xy * K
Ty_e = 1 + n + Z0 * sig_xx - Z0 * sig_xy * (1 / K)

Tx = 2 * P * Tx_e / T_d
Ty = 2 * P * Ty_e / T_d

print(Tx, Ty)
