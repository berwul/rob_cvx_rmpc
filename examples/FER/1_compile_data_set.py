import pandas as pd
import numpy as np

import pathlib

import tqdm

from aux import get_linear_double_integrator_discrete_dynamics

p_curr = pathlib.Path(__file__).parent
P_FER_DATA = p_curr / "LSTM-based_inverse_dynamics_learning_for_franka_emika_panda-main-data"

assert P_FER_DATA.exists(), f"Missing folder: {P_FER_DATA}"

dir_name = "dataset_v3"
p_folder = P_FER_DATA / f"data/{dir_name}/train"

ps = list(p_folder.glob("*meas.csv"))
file_ids = [p.name.replace("_meas.csv", "") for p in ps]
dt = 1 / 100
A, B = get_linear_double_integrator_discrete_dynamics(nr_dof=7, dt=dt)

dists = []
D = []
predictions = []

for file_id in tqdm.tqdm(file_ids, total=len(file_ids), desc="Processing files"):
    suffix = "_meas"
    p_data_file = pathlib.Path(
        P_FER_DATA / f"data/{dir_name}/train/{file_id}{suffix}.csv"
    )
    df = pd.read_csv(str(p_data_file))
    joint_nrs = list(range(1, 8))
    cols = [f" q{i}_meas" for i in joint_nrs]
    qs_m = np.array(df[cols])

    cols = [f" qd{i}_meas" for i in joint_nrs]
    qds_m = np.array(df[cols])

    cols = [f" tau{i}_meas" for i in joint_nrs]
    taus_m = np.array(df[cols])

    t_m = np.array(df['# t_meas'])

    suffix = "_com"
    p_data_file = pathlib.Path(
        P_FER_DATA / f"data/{dir_name}/train/{file_id}{suffix}.csv"
    )

    suffix = "_com"
    df = pd.read_csv(str(p_data_file))
    joint_nrs = list(range(1, 8))
    cols = [f" q{i}{suffix}" for i in joint_nrs]
    qs_c = np.array(df[cols])

    cols = [f" qd{i}{suffix}" for i in joint_nrs]
    qds_c = np.array(df[cols])

    cols = [f" qdd{i}{suffix}" for i in joint_nrs]
    qdds_c = np.array(df[cols])

    t_c = np.array(df['# t_com'])

    mask = t_m < t_c.max()
    qs_m = qs_m[mask]
    qds_m = qds_m[mask]
    taus_m = taus_m[mask]
    t_m = t_m[mask]

    m = t_m.shape[0]
    n = qs_c.shape[1]

    qs_c_interp = np.zeros((m, n))
    qds_c_interp = np.zeros((m, n))
    qdds_c_interp = np.zeros((m, n))

    for i in range(n):
        qs_c_interp[:, i] = np.interp(t_m, t_c, qs_c[:, i])
        qds_c_interp[:, i] = np.interp(t_m, t_c, qds_c[:, i])
        qdds_c_interp[:, i] = np.interp(t_m, t_c, qdds_c[:, i])

    X_m = np.c_[qs_m, qds_m]
    U = qdds_c_interp
    X_pred = X_m @ A.T + U @ B.T

    X_pred = X_pred[:-1]
    X_gt = X_m[1:]

    P, V = np.split(X_m, 2, axis=1)

    accs_norm = np.linalg.norm(U[:-1], axis=-1)
    vel_norm = np.linalg.norm(V[:-1], axis=-1)

    dist = X_gt - X_pred
    delta = np.linalg.norm(dist, axis=-1)

    D.append(np.c_[accs_norm, vel_norm, delta])
    dists.append(np.c_[P[:-1], V[:-1], U[:-1], dist])


p_exp = p_curr / "data" / "exp"
p_exp.mkdir(parents=True, exist_ok=True)

dists = np.vstack(dists)
D = np.vstack(D)
acc, vel, dist = D.T

X = np.c_[acc, vel, np.ones_like(acc)]
theta, *_ = np.linalg.lstsq(X, dist, rcond=None)
a, b, c = theta

# Outlier removal
err = dist - (a * acc + b * vel + c)
err_mean = np.mean(err)
err_std = np.std(err)
thr = err_mean + err_std * 3
mask = err < thr

acc, vel, dist = D[mask].T
dists = dists[mask]

P, V, U, delta_p, delta_v = np.split(dists, 5, axis=1)
deltas = np.c_[delta_p, delta_v]

verts = np.max(np.abs(deltas), axis=0)
print(f"Writing data to: {p_exp}")
np.save(p_exp / "disturbances.npy", dists)
np.save(p_exp / "box_dist_verts.npy", verts)

