import numpy as np
from scipy.optimize import minimize

from pwm.pwm_gen import PWM
from acdcac.plant_acdcac import SinglePhaseACDCACPlant

def stage_func_centralized(i_g, v_dc, i_l, i_l_ref, u_g, u_l,
                           u_g_prev, u_l_prev):
    """
    Basic centralized stage cost (you can refine later):

      - track i_l to i_l_ref
      - regulate v_dc to v_dc_ref
      - penalize control effort and slew
    """
    v_dc_ref = 400.0
    q_i = 1.0
    q_vdc = 0.01
    r_u = 0.01
    r_du = 0.1

    J_i = q_i * (i_l - i_l_ref)**2
    J_vdc = q_vdc * (v_dc - v_dc_ref)**2

    J_u = r_u * (u_g**2 + u_l**2)
    du_g = u_g - u_g_prev
    du_l = u_l - u_l_prev
    J_du = r_du * (du_g**2 + du_l**2)

    return J_i + J_vdc + J_u + J_du


def sim_executor_acdcac_centralized(stage_func,
                                    pwm_grid: PWM,
                                    pwm_load: PWM,
                                    plant: SinglePhaseACDCACPlant,
                                    currentReference,
                                    u0_g,
                                    u0_l,
                                    cont_horizon,
                                    t_0,
                                    state0,
                                    sampling_rate,
                                    sim_time):
    """
    Centralized MPC for AC/DC/AC: one optimizer decides u_g[0..N-1], u_l[0..N-1].
    Uses averaged model in the prediction, and full PWM in the plant.
    """
    N = int(cont_horizon)
    Ts = plant.Ts

    if u0_g is None:
        u0_g = [0.0] * N
    if u0_l is None:
        u0_l = [0.0] * N
    u_prev = np.array(list(u0_g) + list(u0_l), dtype=float)

    def objective(u_vec, x0, t0, u_prev_):
        u_vec = np.asarray(u_vec, dtype=float)
        u_prev_ = np.asarray(u_prev_, dtype=float)
        u_g_seq = u_vec[:N]
        u_l_seq = u_vec[N:]
        u_g_prev = u_prev_[:N]
        u_l_prev = u_prev_[N:]

        i_g, v_dc, i_l = x0
        t = t0
        J = 0.0
        for k in range(N):
            u_g_k = u_g_seq[k]
            u_l_k = u_l_seq[k]
            u_g_prev_k = u_g_prev[k]
            u_l_prev_k = u_l_prev[k]

            v_gr_avg = pwm_grid.average_voltage(u_g_k)
            v_inv_avg = pwm_load.average_voltage(u_l_k)

            i_g, v_dc, i_l = plant.step_euler(
                (i_g, v_dc, i_l),
                v_gr_avg,
                v_inv_avg,
                t,
                Ts,
            )
            t += Ts
            i_l_ref = currentReference.generateRefTrajectory(t)[0]

            J += float(stage_func(i_g, v_dc, i_l,
                                  i_l_ref,
                                  u_g_k, u_l_k,
                                  u_g_prev_k, u_l_prev_k))
        return J

    t_sim = []
    i_g_traj = []
    v_dc_traj = []
    i_l_traj = []
    i_l_ref_traj = []
    u_g_traj = []
    u_l_traj = []
    J_traj = []

    current_time = t_0
    state = tuple(state0)
    steps = int(sim_time / sampling_rate)

    for _ in range(steps):
        u0 = np.copy(u_prev)
        bounds = [(-1, 1)] * (2*N)

        def obj(u_vec):
            return objective(u_vec, state, current_time, u_prev)

        res = minimize(obj, u0, method="trust-constr", bounds=bounds)
        if not res.success:
            u_opt = u_prev
            J = obj(u_prev)
        else:
            u_opt = res.x
            J = res.fun

        u_g_seq = u_opt[:N]
        u_l_seq = u_opt[N:]
        u_g_k = float(u_g_seq[0])
        u_l_k = float(u_l_seq[0])

        # switching-level plant update
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

        t_sim.append(current_time)
        i_g_traj.append(i_g)
        v_dc_traj.append(v_dc)
        i_l_traj.append(i_l)
        i_l_ref_traj.append(i_l_ref)
        u_g_traj.append(u_g_k)
        u_l_traj.append(u_l_k)
        J_traj.append(float(J))

        u_prev = u_opt
        current_time += sampling_rate

    return {
        "t": np.array(t_sim),
        "i_g": np.array(i_g_traj),
        "v_dc": np.array(v_dc_traj),
        "i_l": np.array(i_l_traj),
        "i_l_ref": np.array(i_l_ref_traj),
        "u_g": np.array(u_g_traj),
        "u_l": np.array(u_l_traj),
        "J": np.array(J_traj),
    }
