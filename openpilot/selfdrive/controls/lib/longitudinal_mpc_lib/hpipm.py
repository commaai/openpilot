import numpy as np

# HPIPM's IPM settings as acados runs it: BALANCE mode with acados' overrides.
# The solver can give decent solutions even when it is stopped early, which is
# critical for our purpose where compute time is strictly bounded.
ITER_MAX = 10
TOL = 1e-3
MU0 = 1.
ALPHA_MIN = 1e-8
REG_PRIM = 1e-15  # matters for the zero-cost upper slacks
TAU_MIN = 1e-16
LAM_MIN = 1e-16
INIT_THR = 0.1

# Soft general constraints, each with a lower slack sl and an upper slack su, give four inequalities:
# [lower bound, upper bound, sl >= 0, su >= 0], one row per type: t = SGN * (C @ u) + SEL @ [sl, su] - d >= 0
SGN = np.array([[1.], [-1.], [0.], [0.]])
SEL = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])


def solve_qp(H, g, Q, q, B, b, C, Z, z, d):
  """
  Replica of HPIPM's IPM as run by acados on the condensed QP (delta formulation, cold start,
  conditional Mehrotra predictor-corrector):
    min 1/2 u'Hu + g'u + 1/2 x'Qx + q'x + 1/2 s'Zs + z's  s.t.  x = Bu + b,  t(u, s) >= 0
  """
  nu, nx, nc = H.shape[0], Q.shape[0], d.size
  Hr = H + REG_PRIM * np.eye(nu)
  Qr = Q + REG_PRIM * np.eye(nx)
  Zr = Z + REG_PRIM
  BtQB = B.T @ Qr @ B

  u, x, pi = np.zeros(nu), np.zeros(nx), np.zeros(nx)
  lam_t = np.empty((2,) + d.shape)
  lam, t = lam_t
  t[2:] = np.maximum(-d[2:], INIT_THR)
  s = d[2:] + t[2:]
  t[:2] = np.maximum(s - d[:2], INIT_THR)
  lam[:] = MU0 / t
  dlam_t = np.empty_like(lam_t)
  dlam, dt = dlam_t

  def step(rm):
    gam = t_inv * (rm - lam_rd)
    rs = rg_s + gam[:2] + gam[2:]
    du = H_inv @ (rhs - C.T @ (gam[0] - gam[1] - (F * rs).sum(axis=0)))
    Cdu = C @ du
    ds = -(rs + Gs * Cdu) * D_inv
    np.subtract(SGN * Cdu + SEL @ ds, rd, out=dt)
    np.multiply(-t_inv, rm + lam * dt, out=dlam)
    return -1. / min((dlam_t / lam_t).min(), -1.), du, ds

  def mu_step(alpha):
    lt = lam_t + alpha * dlam_t
    return lt[0].ravel() @ lt[1].ravel() / nc

  alpha = 1.
  for it in range(ITER_MAX + 1):
    rg_u = H @ u + g - C.T @ (lam[0] - lam[1]) + B.T @ pi
    rg_s = Z * s + z - lam[:2] - lam[2:]
    rg_x = Q @ x + q - pi
    rb = B @ u + b - x
    rd = d + t - SGN * (C @ u) - SEL @ s
    rm = lam * t
    if not (max(np.abs(rg_u).max(), np.abs(rg_s).max(), np.abs(rg_x).max()) > TOL or np.abs(rb).max() > TOL or
            np.abs(rd).max() > TOL or abs(rm.max() - TAU_MIN) > TOL):
      break
    if it == ITER_MAX or alpha <= ALPHA_MIN:
      break

    # KKT system with slacks and terminal state eliminated
    mu = rm.sum() / nc
    t_inv = 1. / t
    Gam = lam * t_inv
    lam_rd = lam * rd
    D_inv = 1. / (Zr + Gam[:2] + Gam[2:])
    Gs = SGN[:2] * Gam[:2]
    F = Gs * D_inv
    w = (Gam[:2] * (Zr + Gam[2:]) * D_inv).sum(axis=0)
    try:
      H_inv = np.linalg.inv(Hr + (C.T * w) @ C + BtQB)
    except np.linalg.LinAlgError:  # singular KKT system, fail like a NaN so the MPC resets
      return u, x, s, True
    rhs = -(rg_u + B.T @ (Qr @ rb + rg_x))

    alpha, _, _ = step(rm - TAU_MIN)
    mu_aff = mu_step(alpha)
    sigma_mu = max((mu_aff / mu)**3 * mu, TAU_MIN)
    alpha, du, ds = step(rm + dt * dlam - sigma_mu)
    if mu_step(alpha) > 2. * mu_aff:
      alpha, du, ds = step(rm - sigma_mu)

    a = alpha * ((1. - alpha) * 0.99 + alpha * 0.9999999) if alpha < 1. else alpha
    dx = B @ du + rb
    u = u + a * du
    s = s + a * ds
    x = x + a * dx
    pi = pi + a * (Qr @ dx + rg_x)
    lam_t += a * dlam_t
    np.maximum(lam_t, LAM_MIN, out=lam_t)

  failed = np.isnan(rm).any() or (it < ITER_MAX and alpha <= ALPHA_MIN)
  return u, x, s, failed
