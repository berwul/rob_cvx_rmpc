import pathlib

import numpy as np
import pandas as pd
import tqdm

from aux import interpolate_equidistant, TimeStepIntegratorDiscreteGenericNoise
from controllers.flexible_tube_controller import FlexibleTubeNaiveCorridorController
from controllers.nom_controller import NominalController
from controllers.tube_controller import RobustCorridorController
from corridor_simulators.flexible_tube import FlexibleTubeSimulator
from corridor_simulators.nom import NomSimulator
from corridor_simulators.tube import TubeSimulator

np.random.seed(1)

p_curr = pathlib.Path(__file__).parent
p_data = p_curr / "data"

config_dim = 7
dt = 1 / 100
m_x = config_dim * 2
m_u = config_dim
m_p = config_dim
Q = np.eye(m_x) * 10
Q[config_dim:, config_dim:] *= .01
Q_e = np.eye(m_x) * 1e4
R = np.eye(m_u) * 1e-3
N = 20

fcont = FlexibleTubeNaiveCorridorController.from_cached_dir(
    p_data,
    Q,
    Q_e,
    R,
    N=N,
    u_amp=20,
    v_amp=2,
    dt=1e-2
)

rcont = RobustCorridorController.from_cached_dir(
    p_data,
    Q,
    Q_e,
    R,
    u_amp=20,
    v_amp=2,
    N=N,
    dt=1e-2
)


class NoNoise:

    def __call__(self, x, u):
        return np.zeros_like(x)


class LRUNoise:

    def __init__(self, W, cov):
        self.W = W
        self.cov = cov

    def __call__(self, x, u):
        _, v = np.split(x, 2)
        m, = v.shape
        z = np.abs(np.r_[u, v, 1])
        delta = np.random.multivariate_normal(self.W @ z, self.cov)
        mask_sign = np.random.random(size=m * 2) > 0.5
        delta[mask_sign] *= -1
        return delta


dist = np.load(p_data / "exp" / "disturbances.npy")
P, V, U, err_p, err_V = np.split(dist, 5, axis=1)
X = np.c_[U[:, :m_p], V[:, :m_p], np.ones_like(U[:, 0])]
Y = np.c_[err_p[:, :m_p] * 0, err_V[:, :m_p]]
theta, residuals, *_ = np.linalg.lstsq(np.abs(X), np.abs(Y), rcond=None)
stds =  np.std(Y - X @ theta, axis=0)
cov =  np.cov((Y - X @ theta).T)
noiser = LRUNoise(theta.T, cov)

nom_cont = NominalController(
    fcont.A, fcont.B, Q, Q_e, R, N, u_lim=20, v_lim=2, p_amp=np.pi,
)

simulators = [
    FlexibleTubeSimulator(
        fcont,
        TimeStepIntegratorDiscreteGenericNoise(
            fcont.A, fcont.B,
            noise=noiser,
        ), world=None, aux_controller_steps=4
    ),
    TubeSimulator(
        rcont,
        TimeStepIntegratorDiscreteGenericNoise(
            fcont.A, fcont.B,
            noise=noiser,
        ), world=None, aux_controller_steps=4
    ),
    NomSimulator(nom_cont,
                 TimeStepIntegratorDiscreteGenericNoise(
                     fcont.A, fcont.B,
                     noise=noiser,
                 ), world=None,
                 ),
    NomSimulator(
        nom_cont,
        TimeStepIntegratorDiscreteGenericNoise(
            fcont.A, fcont.B,
            noise=NoNoise(),
        ), world=None
    )
]

names = [
    "ft",
    "rt",
    "nom",
    "nom_star"
]

stats = []

for i in tqdm.trange(10, desc="Running in randomized corridors"):
    path = np.random.uniform(-np.pi / 2, np.pi / 2, size=(3, m_p))
    path_centers = interpolate_equidistant(path, delta=.01)
    path_radii = np.ones_like(path_centers[:, 0]) * 0.2
    for name, simulator in zip(names, simulators):
        (status, ts_goal), all_results = simulator.simulate(
            path_centers,
            path_radii,
            nr_steps=10_000,
            goal_tol=0.1,
            verbose=False
        )
        stats.append({"name": name, "status": status, "ts_goal": ts_goal, "run_nr": i})

pd.DataFrame(stats).to_csv(p_curr / "results" / "runs_stats.csv", index=False)
