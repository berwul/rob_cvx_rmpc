import pathlib
from scipy.linalg import sqrtm

import pickle
import numpy as np

from aux import get_linear_double_integrator_discrete_dynamics, get_linear_ordered_state_constraints, \
    get_linear_box_constraints

p_curr = pathlib.Path(__file__).parent
p_data = p_curr / "data"
D = np.load(p_data / "exp" / "disturbances.npy")
wc_x = np.load(p_data / "exp" / "box_dist_verts.npy")

m = 7
m_x = m * 2
m_p = m

_, Vs, Us, err_p, err_v = np.split(D, 5, axis=1)

Us = Us[:, :m_p]
Vs = Vs[:, :m_p]

accs = np.linalg.norm(Us, axis=1)
vels = np.linalg.norm(Vs, axis=1)

V_max = vels.max()
A_max = accs.max()
X = np.c_[accs, vels, np.ones_like(vels)]

# Position errors ignored since part of measurement errors, not model-errors
err = np.c_[err_p[:, :m_p] * 0, err_v[:, :m_p]]

dt = 1 / 100


A, B = get_linear_double_integrator_discrete_dynamics(nr_dof=m, dt=dt)

v_amp = 2
u_amp = 20

A_x, b_x = get_linear_ordered_state_constraints(m, p_amp=np.pi, v_amp=v_amp)
A_u, b_u = get_linear_box_constraints(m, amp=u_amp)

p_f = p_data / "controllers" / f"controllers.pckl"
with p_f.open("rb") as fp:
    results = pickle.load(fp)

fittness_values = []
Ls = []
rhos = []
outputs = []
L_and_rhos = []

L_12s = []

nr_samples = int(1e2)
all_losses = []

for i, (status, E, Y, c_x_sq, c_u_sq, w_bar_sq, rho, loss) in enumerate(results):
    if status == "failure" or status == "infeasible":
        fittness_values.append(np.nan)
        L_and_rhos.append((np.nan, np.nan))
        L_12s.append((np.nan, np.nan))
        outputs.append(None)
        continue
    P = np.linalg.inv(E)
    K = Y @ P
    P_sqrt = sqrtm(P)
    P_inv_sqrt = sqrtm(E)
    V = np.zeros((m_x, m_x))
    V[m_p:, m_p:] = np.eye(m_p)
    err_gt = np.linalg.norm(err @ P_sqrt.T, axis=1)
    theta, *_ = np.linalg.lstsq(X, err_gt, rcond=None)
    a, b, c = theta
    err_pred = a * accs + b * vels + c
    err_std = np.std(err_pred - err_gt)
    c = c + err_std * 2

    L_1 = a * np.linalg.norm(K @ P_inv_sqrt, ord=2)
    L_2 = b * np.linalg.norm(V @ P_inv_sqrt, ord=2)
    L = L_1 + L_2
    L_12s.append((L_1, L_2))
    rho_L = rho + L
    beta_x_a = (
            a * np.linalg.norm(np.ones(m_p) * 1)
            +
            b * np.linalg.norm(V @ np.ones(m_x))
            +
            c
    )
    s_ = np.inf
    delta_f = np.inf
    # print(rho_L, L_1, L_2)
    if rho_L < 1:
        s_ = beta_x_a / (1 - rho_L)
        delta_f = c / (1 - rho_L)
    c_x = np.linalg.norm(P_inv_sqrt.T @ A_x.T, axis=0)
    c_u = np.linalg.norm((K @ P_inv_sqrt).T @ A_u.T, axis=0)
    c_p, c_v = np.split(c_x, 2)
    angle_ratio_tight = s_ * c_p.max() / 0.1
    vel_ratio_tight = s_ * c_v.max() / v_amp
    control_ratio_tight = s_ * c_u.max() / u_amp
    fittness_values_ = [angle_ratio_tight, vel_ratio_tight, control_ratio_tight]
    fittness_value = max(fittness_values_)
    fittness_values.append(fittness_value)
    L_and_rhos.append((L, rho))
    outputs.append(
        dict(
            rho=rho,
            L=L,
            P=P,
            K=K,
            E=E,
            Y=Y,
            P_sqrt=P_sqrt,
            P_inv_sqrt=P_inv_sqrt,
            c_x_sq=c_x_sq,
            c_u_sq=c_u_sq,
            w_bar_sq=w_bar_sq,
            a=a,
            b=b,
            c=c
        )
    )
    all_losses.append(loss)


fittness_values = np.r_[fittness_values]
L_and_rhos = np.vstack(L_and_rhos)
mask = L_and_rhos.sum(axis=1) < 1
indxs, = np.nonzero(mask)

if indxs.size:
    i = indxs[fittness_values[indxs].argmin()]
    L, rho = L_and_rhos[i]
    print(
        f"Ftube controller picked: rho: {rho} rho_tilde: {L + rho: 0.2f}, fittness value: {fittness_values[i]: 0.2f}"
    )
    np.savez(
        p_data / "ftube_controller.npz",
        **outputs[i]
    )
else:
    print("No flexible tube controller was computed")


if fittness_values.size:
    i = np.nanargmin(fittness_values)
    rho_picked = outputs[i]["rho"]
    print(
        f"Rtube controller picked: rho: {rho_picked}, fittness value: {fittness_values[i]: 0.2f}"
    )
    np.savez(
        p_data / "rtube_controller.npz",
        **outputs[i]
    )
else:
    print("No rigid tube controller was computed")

