# Copyright (c) 2025, ABB Schweiz AG
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from functools import partial
from itertools import product
import pathlib
import pickle
import multiprocessing

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm
import tqdm

from aux import (
    get_linear_box_constraints,
    get_linear_ordered_state_constraints,
    get_linear_double_integrator_discrete_dynamics
)

from offline_pipeline_c_compute_constants import get_model_and_disc_error_constants


def run_controller_pipeline(
            path_results: pathlib.Path,
            problem,
            logger
    ):
    data_system = np.load(path_results / "system.npz")
    A = data_system["A"]
    B = data_system["B"]
    data = np.load(path_results / "data_constraints.npz")
    u_amp, v_amp = data["u_amp"], data["v_amp"]
    data = np.load(path_results / "data_model_error_verts.npz")
    # box set that bounds disturbances due to model errors
    verts_acc = data["verts_acc"]
    # box set that bounds disturbances due to discretization errors
    verts_state_disc = data["verts_state_disc"]
    wc_x = (B @ verts_acc + verts_state_disc)
    path_f = path_results / "data_controllers.pckl"
    # controller_results = compute_controllers(
    #     A, B, wc_x, u_amp, v_amp, nr_rhos=20, nr_workers=10
    # )
    controller_results = compute_controllers_decoupled(
        problem.dt, wc_x, u_amp, v_amp, nr_rhos=20, nr_workers=10
    )
    with path_f.open("wb") as fp:
        pickle.dump(controller_results, fp)

    pick_ftube_controller_from_results(
        problem, logger
    )

    pick_rtube_controller_from_results(
        problem.p_cached_dir, logger
    )


def compute_controllers(
        A,
        B,
        wc_x,
        u_amp,
        v_amp,
        nr_rhos=20,
        nr_workers=10,
        rho_l=0.8,
        rho_u=0.99,
        c_v_wht=None,
        c_u_wht=None
):
    m_x, m_u = B.shape
    verts_w = np.vstack(list(product(*([(-1, 1)] * m_x)))) * wc_x
    m_p = int(m_x // 2)
    A_x, b_x = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
    A_u, b_u = get_linear_box_constraints(m_u, amp=u_amp)
    rhos = np.linspace(rho_l, rho_u, nr_rhos)
    c_p_wht = 1 / 0.1
    c_v_wht = c_v_wht or 1 / v_amp
    c_u_wht = c_u_wht or 1 / u_amp
    func_ = partial(
        solve_rpi_with_constraints,
        A,
        B,
        A_x,
        A_u,
        verts_w,
        c_p_wht,
        c_v_wht,
        c_u_wht
    )
    with multiprocessing.Pool(nr_workers) as pool:
        results = []
        for res in tqdm.tqdm(pool.imap(func_, rhos), total=rhos.size, desc="Solving LMIs"):
            results.append(res)
    return results


def compute_controllers_decoupled(
        dt,
        wc_x,
        u_amp,
        v_amp,
        nr_rhos=20,
        nr_workers=10,
        rho_l=0.8,
        rho_u=0.99,
        c_v_wht=None,
        c_u_wht=None
):
    wc_p, wc_v = np.split(wc_x, 2)
    nr_dof = wc_p.size
    A_, B_ = get_linear_double_integrator_discrete_dynamics(1, dt, method="zoh")

    m_x, m_u = B_.shape
    unit_box_ = np.vstack(list(product(*([(-1, 1)] * m_x))))

    A_x_, _ = get_linear_ordered_state_constraints(m_u, p_amp=np.pi, v_amp=v_amp)
    A_u_, _ = get_linear_box_constraints(m_u, amp=u_amp)
    rhos = np.linspace(rho_l, rho_u, nr_rhos)

    c_p_wht = 1 / 0.1
    c_v_wht = c_v_wht or 1 / v_amp
    c_u_wht = c_u_wht or 1 / u_amp

    all_results = []
    for i in range(nr_dof):
        wc_x_ = np.r_[wc_p[i], wc_v[i]]
        verts_w_ = unit_box_ * wc_x_
        func_ = partial(
            solve_rpi_with_constraints,
            A_,
            B_,
            A_x_,
            A_u_,
            verts_w_,
            c_p_wht,
            c_v_wht,
            c_u_wht
        )
        with multiprocessing.Pool(nr_workers) as pool:
            results = []
            for res in tqdm.tqdm(pool.imap(func_, rhos), total=rhos.size, desc=f"Joint {i+1}: Solving LMIs"):
                results.append(res)
        all_results.append(results)

    n_x = nr_dof * 2
    n_u = nr_dof

    all_results_final = []

    for j in range(nr_rhos):
        E = np.zeros((n_x, n_x))
        Y = np.zeros((n_u, n_x))

        c_pp_sq = []
        c_pn_sq = []
        c_vp_sq = []
        c_vn_sq = []
        c_up_sq = []
        c_un_sq = []

        w_bar_sq = []
        is_status_nok = True
        loss = 0
        for i in range(nr_dof):
            status, E_, Y_, c_x_sq_, c_u_sq_, w_bar_sq_, rho, loss_ = all_results[i][j]
            is_status_nok = status == "failure" or status == "infeasible"
            if is_status_nok:
                break
            if E_ is None:
                print(status)
                break
            e_11, e_12, e_21, e_22 = E_.ravel()
            y_1, y_2 = Y_.ravel()
            # E \in R^{n_x, n_x}
            E[i, i] = e_11
            E[i, i + n_u] = e_12
            E[i + n_u, i] = e_21
            E[i + n_u, i + n_u] = e_22
            # Y \in R^{n_u, n_x}
            Y[i, i] = y_1
            Y[i, i + n_u] = y_2

            c_pp_sq_, c_pn_sq_, c_vp_sq_, c_vn_sq_ = np.split(c_x_sq_, 4)
            c_up_sq_, c_un_sq_ = np.split(c_u_sq_, 2)
            c_pp_sq.append(c_pp_sq_)
            c_pn_sq.append(c_pn_sq_)
            c_vp_sq.append(c_vp_sq_)
            c_vn_sq.append(c_vn_sq_)

            c_up_sq.append(c_up_sq_)
            c_un_sq.append(c_pn_sq_)
            w_bar_sq.append(w_bar_sq_)
            loss += loss_

        if is_status_nok:
            all_results_final.append((status, None, None, None, None, None, rho, loss))
        else:
            c_x_sq = np.r_[c_pp_sq, c_pn_sq, c_vp_sq, c_vn_sq]
            c_u_sq = np.r_[c_up_sq, c_un_sq]
            all_results_final.append((status, E, Y, c_x_sq, c_u_sq, sum(w_bar_sq), rho, loss))
    return all_results_final


def compile_opt_problem(
        A,
        B,
        A_x,
        A_u,
        verts_w,
        c_p_wht,
        c_v_wht,
        c_u_wht,
        rho
):
    m_x, m_u = B.shape

    n_x = A_x.shape[0]
    n_p = int(n_x / 2)

    # Assume position constraints make up the first half of rows in matrix, the other half is velocity constraints.
    # A_x = [[A_p], [A_v]], A_p, A_v \in R^{n_p, n}

    n_u = A_u.shape[0]
    n_w = verts_w.shape[0]

    E = cp.Variable((m_x, m_x), PSD=True, name="E")
    Y = cp.Variable((m_u, m_x), name="Y")

    c_x_sq = cp.Variable((n_x,), name="c_x_sq")
    c_u_sq = cp.Variable((n_u,), name="c_u_sq")

    w_bar_sq = cp.Variable(name="w_bar_sq")
    objective = 1 / (2 * (1 - rho)) * (
            (n_x + n_u) * w_bar_sq
            +
            c_x_sq[:n_p].sum() * c_p_wht
            +
            c_x_sq[n_p:].sum() * c_v_wht
            +
            c_u_sq.sum() * c_u_wht
    )
    const = [
        E >> np.eye(m_x) * 1e-8,
        cp.bmat([
            [(rho ** 2) * E, (A @ E + B @ Y).T],
            [(A @ E + B @ Y), E]
        ]) >> 0,
        c_x_sq >= 0,
        c_u_sq >= 0,
        w_bar_sq >= 0
    ]
    for i in range(n_x):
        const += [
            cp.bmat([
                [c_x_sq[i][None, None], A_x[i, None] @ E],
                [(A_x[i, None] @ E).T, E],
            ])
            >> 0
        ]
    for i in range(n_u):
        const += [
            cp.bmat([
                [c_u_sq[i][None, None], A_u[i, None] @ Y],
                [(A_u[i, None] @ Y).T, E],
            ]) >> 0
        ]
    for i in range(n_w):
        const += [
            cp.bmat([
                [w_bar_sq[None, None], verts_w[i][None]],
                [verts_w[i][:, None], E],
            ]) >> 0
        ]
    problem = cp.Problem(cp.Minimize(objective), constraints=const)
    return problem


def solve_rpi_with_constraints(
        A,
        B,
        A_x,
        A_u,
        verts_w,
        c_p_wht,
        c_v_wht,
        c_u_wht,
        rho
):
    problem = compile_opt_problem(
        A,
        B,
        A_x,
        A_u,
        verts_w,
        c_p_wht,
        c_v_wht,
        c_u_wht,
        rho
    )
    try:
        loss = problem.solve(solver=cp.MOSEK, verbose=False)
        status = problem.status
        names = [
            "E", "Y", "c_x_sq", "c_u_sq", "w_bar_sq"
        ]
        E, Y, c_x_sq, c_u_sq, w_bar_sq = [problem.var_dict[name].value for name in names]
        return status, E, Y, c_x_sq, c_u_sq, w_bar_sq, rho, loss
    except Exception as e:
        # print("Caught exception")
        # traceback.print_exc()
        return "failure", None, None, None, None, None, rho, np.nan


def pick_ftube_controller_from_results(problem, logger):
    p_dir : pathlib.Path = problem.p_cached_dir
    p_f_tube_dir = p_dir / "f_controller_results"
    p_f_tube_dir.mkdir(exist_ok=True, parents=True)
    data_system = np.load(p_dir / "system.npz")
    A, B = data_system["A"], data_system["B"]

    data_const = np.load(p_dir / "data_constraints.npz")
    u_amp, v_amp = data_const["u_amp"], data_const["v_amp"]

    m_x, m_u = B.shape
    m_p = m_u

    A_x, b_x = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
    A_u, b_u = get_linear_box_constraints(m_u, amp=u_amp)

    with (p_dir / "data_controllers.pckl").open("rb") as fp:
        results = pickle.load(fp)

    L_and_rhos, fittness_values, outputs = [], [], []

    for i, (status, E, Y, c_x_sq, c_u_sq, w_bar_sq, rho, loss) in enumerate(results):
        if status == "failure" or status == "infeasible":
            continue
        P = np.linalg.inv(E)
        K = Y @ P
        P_sqrt = sqrtm(P)
        P_inv_sqrt = sqrtm(E)
        V = np.zeros((m_x, m_x))
        V[m_p:, m_p:] = np.eye(m_p)
        a, b, c = get_model_and_disc_error_constants(
            problem,
            A,
            B,
            P_sqrt,
            u_amp,
            v_amp,
            logger,
            nr_samples=100,
            max_iter_cnt=1
        )
        L = a * np.linalg.norm(K @ P_inv_sqrt, ord=2) + b * np.linalg.norm(V @ P_inv_sqrt, ord=2)
        beta_x_a = a * np.sqrt(m_p) * 1 + b * np.sqrt(m_p) * 1 + c
        rho_L = rho + L
        s_ = np.inf
        delta_f = np.inf
        if rho_L < 1:
            s_ = beta_x_a / (1 - rho_L)
            delta_f = c / (1 - rho_L)

        c_x = np.linalg.norm(P_inv_sqrt.T @ A_x.T, axis=0)
        c_u = np.linalg.norm((K @ P_inv_sqrt).T @ A_u.T, axis=0)

        angle_ratio_tight = s_ * c_x[:m_x].max() / 0.1
        vel_ratio_tight = s_ *  c_x[m_x:].max() / v_amp
        control_ratio_tight = s_ * c_u.max() / u_amp
        fittness_value = max(control_ratio_tight, angle_ratio_tight, vel_ratio_tight)
        logger.debug(f"{i}: | rho: {rho} L: {L}, rho_L: {rho_L} delta_f: {delta_f:0.2f} fv: {fittness_value}")

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
    L_and_rhos = np.vstack(L_and_rhos)
    fittness_values = np.r_[fittness_values]
    mask = L_and_rhos.sum(axis=1) < 1
    indxs, = np.nonzero(mask)
    if indxs.size:
        i = indxs[fittness_values[indxs].argmin()]
        L, rho = L_and_rhos[i]
        logger.debug(
            f"Ftube controller picked: rho: {rho} rho_tilde: {L + rho: 0.2f}, fittness value: {fittness_values[i]: 0.2f}"
        )
        np.savez(
            p_dir / "ftube_controller.npz",
            **outputs[i]
        )
    else:
        logger.error("No flexible tube controller was computed")


def pick_rtube_controller_from_results(p_dir, logger):
    data_system = np.load(p_dir / "system.npz")
    A, B = data_system["A"], data_system["B"]
    data_const = np.load(p_dir / "data_constraints.npz")
    u_amp, v_amp = data_const["u_amp"], data_const["v_amp"]
    m_x, m_u = B.shape
    m_p = m_u

    with (p_dir / "data_controllers.pckl").open("rb") as fp:
        results = pickle.load(fp)

    outputs = []
    fittness_values = []

    for (status, E, Y, c_x_sq, c_u_sq, w_bar_sq, rho, loss) in results:
        if status == "failure" or status == "infeasible":
            continue
        P = np.linalg.inv(E)
        K = Y @ P
        P_sqrt = sqrtm(P)
        P_inv_sqrt = sqrtm(E)
        V = np.zeros((m_x, m_x))
        V[m_p:, m_p:] = np.eye(m_p)
        L = 0
        w_bar = np.sqrt(w_bar_sq)
        delta = w_bar / (1 - rho)
        b_u_tilde = np.sqrt(c_u_sq) * delta
        b_x_tilde = np.sqrt(c_x_sq) * delta

        angle_ratio_tight = b_x_tilde[:m_x].max() / 0.1
        vel_ratio_tight = b_x_tilde[m_x:].max() / v_amp
        control_ratio_tight = b_u_tilde.max() / u_amp
        fittness_value = max(control_ratio_tight, angle_ratio_tight, vel_ratio_tight)
        outputs.append((rho, L, P, P_sqrt, P_inv_sqrt, K, E, Y, c_x_sq, c_u_sq, w_bar_sq))
        fittness_values.append(fittness_value)
    fittness_values = np.r_[fittness_values]
    if fittness_values.size:
        i = fittness_values.argmin()
        (rho, L, P, P_sqrt, P_inv_sqrt, K, E, Y, c_x_sq, c_u_sq, w_bar_sq) = outputs[i]
        logger.debug(
            f"rtube controller picked, fittness value: {fittness_values.min(): 0.2f} rho, {rho:0.2f}"
        )
        np.savez(
            p_dir / "rtube_controller.npz",
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
            w_bar_sq=w_bar_sq
        )
    else:
        logger.debug("No rigid tube controller was computed")
