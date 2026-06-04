import time
import numpy as np

from aux import  compute_goal_state, interpolate_equidistant
from corridor_simulators.base import BaseSimulator


class FlexibleTubeSimulator(BaseSimulator):

    def __init__(self, cont, integrator, world=None, aux_controller_steps=4):
        super().__init__(cont, integrator, world)
        self.aux_controller_steps = aux_controller_steps

    def simulate(
            self,
            path_centers,
            path_radii,
            nr_steps=1000,
            goal_tol=1e-2,
            verbose=False
    ):
        aux_controller_steps = self.aux_controller_steps
        cont = self.cont
        world = self.world
        integrator = self.integrator
        N = cont.N
        m_x, m_u = cont.A.shape
        m_p = int(m_x / 2)
        p_g = path_centers[-1]
        O_mp = np.zeros_like(p_g)
        x_g = np.r_[p_g, O_mp]
        x_0 = np.r_[path_centers[0], O_mp]
        path_traj = x_0[:m_p][None].repeat(N + 1, axis=0)
        x = x_0
        all_results = []
        ts = 0
        status = "not_in_goal"
        path_track = path_centers.copy()
        path_track = interpolate_equidistant(path_track, delta=0.001)
        x_believed = x.copy()
        # estimated tube size at the end of aux controllers horizon
        s_M = 0
        while ts < nr_steps:
            time_s = time.time()
            cs, rs = self.get_corridor_balls(path_traj, path_centers, path_radii, p_g)
            x_g_v, i_max_goal = compute_goal_state(cs, rs, path_track, p_g, return_index=True)
            time_e_corridor_planning = time.time() - time_s
            cont.set_parameter_values(x_believed, cs.T, rs, x_g_v, s_M)
            # assert (cont.delta_f * cont.r_p_0  < rs).all()
            time_mpc_s = time.time()
            try:
                success = cont.solve()
                status = cont.problem.status
            except:
                status = "failure"
                success = False
            time_mpc_e = time.time() - time_mpc_s
            if success:
                time_solver_e = cont.problem.solver_stats.solve_time
            else:
                time_solver_e = time_mpc_e
            time_e = time.time() - time_s
            x_t = x.copy()
            states_aux_controller = []
            noise = []
            if not success:
                all_results.append(
                    FlexibleTubeResults(
                        cont,
                        noise,
                        states_aux_controller,
                        times=(time_e, time_e_corridor_planning, time_mpc_e, time_solver_e),
                        x_t=x_t
                    )
                )
                break
            # predict future tube state
            x_believed = cont.get_predicted_state(k=aux_controller_steps)
            # predict future tube size
            s_M, betas = cont.predict_tube_size(x, M=aux_controller_steps)
            for k in range(aux_controller_steps):
                v = cont.get_solved_control(x, k=k)
                x = integrator.solve_time_step(x, v)
                noise.append(integrator.get_noise(x, v))  # for debug
                ts += 1
                states_aux_controller.append(np.r_[x, v])
            states_aux_controller = np.vstack(states_aux_controller)
            all_results.append(
                FlexibleTubeResults(
                    cont,
                    noise,
                    states_aux_controller,
                    times=(time_e, time_e_corridor_planning, time_mpc_e, time_solver_e),
                    x_t=x_t
                )
            )
            # debug - verify inside of tube
            dist = x_believed - x
            s_ = np.linalg.norm(self.cont.P_sqrt @ dist)
            if s_ > s_M:
                noise = np.vstack(noise)
                ss_ = np.linalg.norm(self.cont.P_sqrt @ noise.T, axis=0)
                # self.rho_d
                print(s_, s_M, self.cont.c, ss_, betas[:aux_controller_steps])
                status = "infeasible outcome"
                break
            X = cont.get_predicted_traj()
            X_shifted = self.get_shifted_trajectory(X, nr_shifts=aux_controller_steps)
            path_traj = X_shifted[:, :m_p]
            if world is not None:
                path_aux_controller = states_aux_controller[:, :m_p]
                coll_check_qs = np.vstack([path_aux_controller, path_traj])
                if self.is_path_in_collision(coll_check_qs):
                    status = "collision"
                    break
            in_goal = np.linalg.norm(x - x_g) < goal_tol
            self.step_write(ts, x, x_g, time_e, status, write=verbose)
            if status == "collision":
                break
            if in_goal:
                status = "success"
                break
        self.end_write(ts, x, x_g, status, write=verbose)
        return (status, ts), all_results


class FlexibleTubeResults:

    def __init__(self, cont, noise, states_aux_controller, times, x_t):
        self.inputs = cont.get_solution_inputs()
        self.outputs = cont.get_solution_results()
        self.noise = noise
        self.states_aux_controller = states_aux_controller
        self.times = times
        self.x_t = x_t
