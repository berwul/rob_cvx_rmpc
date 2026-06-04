import pathlib
import pickle
import time
from itertools import product

import cvxpy as cp
import numpy as np

from aux import (
    get_linear_double_integrator_discrete_dynamics,
    get_linear_ordered_state_constraints,
    get_linear_box_constraints
)
from offline_pipeline_b_controller import compute_controllers_decoupled

dt = 1 / 100
m = 7
m_x = m * 2

A, B = get_linear_double_integrator_discrete_dynamics(nr_dof=m, dt=dt)
A_x, b_x = get_linear_ordered_state_constraints(m, p_amp=np.pi, v_amp=2.)
A_u, b_u = get_linear_box_constraints(m, amp=20.)

p_curr = pathlib.Path(__file__).parent
p_data = p_curr / "data"
p_data_cnt = p_data / "controllers"
p_data_cnt.mkdir(exist_ok=True, parents=True)

wc_x = np.load(p_data / "exp" / "box_dist_verts.npy")

wc_p, wc_v = np.split(wc_x, 2)
wc_p = wc_p
wc_v = wc_v
# Position errors ignored since part of measurement errors, not model-errors
wc_x = np.r_[wc_p * 0., wc_v]

verts_w = np.vstack(list(product(*([(-1, 1)] * m_x)))) * wc_x


assert cp.MOSEK in cp.installed_solvers(), "Missing MOSEK solver"

time_s = time.time()
results = compute_controllers_decoupled(
    dt,
    wc_x,
    u_amp=20,
    v_amp=2,
    rho_l=0.5,
    rho_u=0.999,
    nr_rhos=50,
    nr_workers=10,
)
time_e = time.time() - time_s
print(f"Time taken: {time_e}")

p_f = (p_data_cnt / f"controllers.pckl")

with p_f.open("wb") as fp:
    pickle.dump(results, fp)
