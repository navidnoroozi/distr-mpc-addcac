import numpy as np
from scipy.optimize import minimize

from acdcac.fsclf import FiniteStepLyapunov

class DistributedMPCSub1:
    """
    Subsystem 1: Rectifier + DC-link
      x1 = [i_g, v_dc], neighbor = i_l, control = u_g
    """

    def __init__(self, plant, pwm_grid, fsclf,
                 horizon_N, M,
                 u_min=-1.0, u_max=1.0,
                 lambda_V=1.0, lambda_u=1e-2, lambda_du=1e-1):
        self.plant = plant
        self.pwm = pwm_grid
        self.fsclf = fsclf
        self.N = int(horizon_N)
        self.M = int(M)
        self.u_min = u_min
        self.u_max = u_max
        self.lambda_V = lambda_V
        self.lambda_u = lambda_u
        self.lambda_du = lambda_du

    def _predict_next(self, x1, i_l_neighbor, u_g, t):
        i_g, v_dc = x1
        i_l = i_l_neighbor
        v_gr = self.pwm.average_voltage(u_g)
        v_inv = 0.0

        x = (i_g, v_dc, i_l)
        i_g_n, v_dc_n, _ = self.plant.step_euler(
            x, v_gr, v_inv, t, self.plant.Ts
        )
        return (i_g_n, v_dc_n)

    def _cost(self, u_seq, x1_0, i_l_bar, t0, u_prev, V_target):
        u_seq = np.asarray(u_seq, dtype=float)
        u_prev = np.asarray(u_prev, dtype=float)
        i_g, v_dc = x1_0
        t = t0
        J = 0.0

        for k in range(self.N):
            u_k = u_seq[k]
            u_prev_k = u_prev[k]
            i_l_k = i_l_bar[k]

            V1 = self.fsclf.V_sub1((i_g, v_dc))
            J += self.lambda_V * V1
            J += self.lambda_u * (u_k**2)
            du = u_k - u_prev_k
            J += self.lambda_du * (du**2)

            i_g, v_dc = self._predict_next((i_g, v_dc), i_l_k, u_k, t)
            t += self.plant.Ts

            if (k + 1) == self.M and V_target is not None:
                V1_M = self.fsclf.V_sub1((i_g, v_dc))
                slack = max(0.0, V1_M - V_target)
                J += 1e3 * slack
        return J

    def solve(self, x1_0, i_l_bar, t0, u_prev, V_target):
        if u_prev is None:
            u_prev = np.zeros(self.N)
        u_prev = np.asarray(u_prev, dtype=float)
        u0 = np.copy(u_prev)
        bounds = [(self.u_min, self.u_max)] * self.N

        def obj(u_vec):
            return self._cost(u_vec, x1_0, i_l_bar, t0, u_prev, V_target)

        res = minimize(obj, u0, method="trust-constr", bounds=bounds)
        if not res.success:
            return np.copy(u_prev), float(obj(u_prev))
        return res.x, float(res.fun)


class DistributedMPCSub2:
    """
    Subsystem 2: Inverter / load
      x2 = [i_l], neighbor = [i_g, v_dc], control = u_l
    """

    def __init__(self, plant, pwm_load, fsclf,
                 horizon_N, M, currentReference,
                 u_min=-1.0, u_max=1.0,
                 lambda_V=1.0, lambda_u=1e-2, lambda_du=1e-1,
                 lambda_tracking=1.0):
        self.plant = plant
        self.pwm = pwm_load
        self.fsclf = fsclf
        self.N = int(horizon_N)
        self.M = int(M)
        self.currentReference = currentReference
        self.u_min = u_min
        self.u_max = u_max
        self.lambda_V = lambda_V
        self.lambda_u = lambda_u
        self.lambda_du = lambda_du
        self.lambda_tracking = lambda_tracking

    def _predict_next(self, x2, x1_neighbor, u_l, t):
        i_l = x2[0]
        i_g_k, v_dc_k = x1_neighbor
        v_inv = self.pwm.average_voltage(u_l)
        v_gr = 0.0
        x = (i_g_k, v_dc_k, i_l)
        _, _, i_l_n = self.plant.step_euler(
            x, v_gr, v_inv, t, self.plant.Ts
        )
        return (i_l_n,)

    def _cost(self, u_seq, x2_0, x1_bar, t0, u_prev, V_target):
        u_seq = np.asarray(u_seq, dtype=float)
        u_prev = np.asarray(u_prev, dtype=float)
        i_l = x2_0[0]
        t = t0
        J = 0.0

        for k in range(self.N):
            u_k = u_seq[k]
            u_prev_k = u_prev[k]
            i_g_k, v_dc_k = x1_bar[k]

            i_l_ref = self.currentReference.generateRefTrajectory(t)[0]
            J += self.lambda_tracking * (i_l - i_l_ref)**2
            V2 = self.fsclf.V_sub2((i_l,))
            J += self.lambda_V * V2
            J += self.lambda_u * (u_k**2)
            du = u_k - u_prev_k
            J += self.lambda_du * (du**2)

            (i_l,) = self._predict_next((i_l,), (i_g_k, v_dc_k), u_k, t)
            t += self.plant.Ts

            if (k + 1) == self.M and V_target is not None:
                V2_M = self.fsclf.V_sub2((i_l,))
                slack = max(0.0, V2_M - V_target)
                J += 1e3 * slack
        return J

    def solve(self, x2_0, x1_bar, t0, u_prev, V_target):
        if u_prev is None:
            u_prev = np.zeros(self.N)
        u_prev = np.asarray(u_prev, dtype=float)
        u0 = np.copy(u_prev)
        bounds = [(self.u_min, self.u_max)] * self.N

        def obj(u_vec):
            return self._cost(u_vec, x2_0, x1_bar, t0, u_prev, V_target)

        res = minimize(obj, u0, method="trust-constr", bounds=bounds)
        if not res.success:
            return np.copy(u_prev), float(obj(u_prev))
        return res.x, float(res.fun)


def sim_executor_acdcac_distributed(
    plant,
    pwm_grid,
    pwm_load,
    currentReference,
    fsclf: FiniteStepLyapunov,
    horizon_N,
    M,
    t_0,
    state0,
    sampling_rate,
    sim_time,
):
    """
    Distributed multi-step contractive MPC simulation for the AC/DC/AC converter.
    """

    sub1 = DistributedMPCSub1(plant, pwm_grid, fsclf, horizon_N, M)
    sub2 = DistributedMPCSub2(plant, pwm_load, fsclf, horizon_N, M, currentReference)

    t_log = []
    i_g_log = []
    v_dc_log = []
    i_l_log = []
    i_l_ref_log = []
    u_g_log = []
    u_l_log = []
    J1_log = []
    J2_log = []

    current_time = t_0
    state = tuple(state0)
    i_g0, v_dc0, i_l0 = state

    u_g_prev = np.zeros(horizon_N)
    u_l_prev = np.zeros(horizon_N)
    i_l_bar = np.full(horizon_N, i_l0)
    x1_bar = np.array([(i_g0, v_dc0)] * horizon_N, dtype=float)

    total_steps = int(sim_time / sampling_rate)
    outer_steps = total_steps // M

    for _ in range(outer_steps):
        i_g, v_dc, i_l = state
        V0 = fsclf.V(state)
        alpha = 0.9
        V_target = alpha * V0

        # Sub1 update
        u_g_seq, J1 = sub1.solve((i_g, v_dc), i_l_bar, current_time, u_g_prev, V_target)

        # Predict x1_bar from Sub1
        x1_pred = []
        i_g_p, v_dc_p = i_g, v_dc
        t_pred = current_time
        for k in range(horizon_N):
            u_g_k = u_g_seq[k]
            i_l_k = i_l_bar[k]
            i_g_p, v_dc_p = sub1._predict_next((i_g_p, v_dc_p), i_l_k, u_g_k, t_pred)
            t_pred += plant.Ts
            x1_pred.append((i_g_p, v_dc_p))
        x1_bar = np.array(x1_pred, dtype=float)

        # Sub2 update
        u_l_seq, J2 = sub2.solve((i_l,), x1_bar, current_time, u_l_prev, V_target)

        # Predict i_l_bar for next outer step
        i_l_pred = []
        i_l_p = i_l
        t_pred = current_time
        for k in range(horizon_N):
            u_l_k = u_l_seq[k]
            i_g_k, v_dc_k = x1_bar[k]
            (i_l_p,) = sub2._predict_next((i_l_p,), (i_g_k, v_dc_k), u_l_k, t_pred)
            t_pred += plant.Ts
            i_l_pred.append(i_l_p)
        i_l_bar = np.array(i_l_pred, dtype=float)

        # Apply first M inputs with PWM
        for m in range(M):
            u_g_k = float(u_g_seq[m])
            u_l_k = float(u_l_seq[m])

            v_gr_sub, dt_sub = pwm_grid.synthesize_over_interval(
                u_g_k, current_time, Ts=sampling_rate
            )
            v_inv_sub, _ = pwm_load.synthesize_over_interval(
                u_l_k, current_time, Ts=sampling_rate
            )
            state = plant.step_substeps(
                state, v_gr_sub, v_inv_sub, current_time, dt_sub
            )
            i_g, v_dc, i_l = state
            i_l_ref = currentReference.generateRefTrajectory(current_time)[0]

            t_log.append(current_time)
            i_g_log.append(i_g)
            v_dc_log.append(v_dc)
            i_l_log.append(i_l)
            i_l_ref_log.append(i_l_ref)
            u_g_log.append(u_g_k)
            u_l_log.append(u_l_k)
            J1_log.append(J1)
            J2_log.append(J2)

            current_time += sampling_rate

        u_g_prev = np.concatenate([u_g_seq[M:], np.full(M, u_g_seq[-1])])
        u_l_prev = np.concatenate([u_l_seq[M:], np.full(M, u_l_seq[-1])])

    return {
        "t": np.array(t_log),
        "i_g": np.array(i_g_log),
        "v_dc": np.array(v_dc_log),
        "i_l": np.array(i_l_log),
        "i_l_ref": np.array(i_l_ref_log),
        "u_g": np.array(u_g_log),
        "u_l": np.array(u_l_log),
        "J1": np.array(J1_log),
        "J2": np.array(J2_log),
    }
