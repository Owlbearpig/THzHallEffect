import sympy as sym
from sympy import Expr, solve
from IPython.display import display

N, t, om = sym.symbols('N tau omega')
n, k, p = sym.symbols('n k p')
Tx, Ty = sym.symbols('T_x T_y')
c = sym.symbols('c')
om_c = sym.symbols('omega_c')

a, b = 1 - 1j*om*t, om_c * t
sig = c * N * t
sig_xy, sig_xx = sig * b / (a**2 + b**2), sig * a / (a**2 + b**2)

expr1 = -(Tx/2*p)+(n+sig_xx+sig_xy*k)/((n+sig_xx)**2 + sig_xy**2)
expr2 = -(Ty/2*p)+(n+sig_xx-sig_xy/k)/((n+sig_xx)**2 + sig_xy**2)

t = solve([expr1, expr2], t, N)

print(t)
