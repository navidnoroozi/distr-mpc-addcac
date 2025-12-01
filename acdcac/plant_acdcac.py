import math

class SinglePhaseACDCACPlant:
    """
    Single-phase AC/DC/AC converter with full-bridge PWM on both sides.

    States: x = [i_g, v_dc, i_l]
      i_g  : grid-side inductor current
      v_dc : DC-link capacitor voltage
      i_l  : load-side inductor current
    """

    def __init__(self,
                 sampling_rate,
                 Lg,
                 Rg,
                 Cdc,
                 Ll,
                 Rl,
                 V_grid_rms,
                 f_grid):
        self.Ts = float(sampling_rate)
        self.Lg = float(Lg)
        self.Rg = float(Rg)
        self.Cdc = float(Cdc)
        self.Ll = float(Ll)
        self.Rl = float(Rl)
        self.V_grid_rms = float(V_grid_rms)
        self.f_grid = float(f_grid)

    # --- external voltages ---

    def _v_grid(self, t):
        V_peak = math.sqrt(2.0) * self.V_grid_rms
        return V_peak * math.sin(2.0 * math.pi * self.f_grid * t)

    def _e_load(self, t):
        # for now: purely RL load; extend if you want a motor emf
        return 0.0

    # --- continuous-time dynamics ---

    def deriv(self, state, v_gr_ctrl, v_inv_ctrl, t):
        """
        state = (i_g, v_dc, i_l)
        v_gr_ctrl: grid-side bridge AC voltage
        v_inv_ctrl: inverter bridge AC voltage
        """
        i_g, v_dc, i_l = state
        v_g = self._v_grid(t)
        e_l = self._e_load(t)

        # di_g/dt = (-Rg * i_g + v_g - v_gr_ctrl) / Lg
        di_g = (-self.Rg * i_g + v_g - v_gr_ctrl) / self.Lg

        # dv_dc/dt = (i_g - i_l) / Cdc
        dv_dc = (i_g - i_l) / self.Cdc

        # di_l/dt = (-Rl * i_l + v_inv_ctrl - e_l) / Ll
        di_l = (-self.Rl * i_l + v_inv_ctrl - e_l) / self.Ll

        return di_g, dv_dc, di_l

    # --- simple Euler integrators, like your Load class ---

    def step_euler(self, state, v_gr_ctrl, v_inv_ctrl, t, dt):
        di_g, dv_dc, di_l = self.deriv(state, v_gr_ctrl, v_inv_ctrl, t)
        i_g, v_dc, i_l = state

        i_g_n = i_g + dt * di_g
        v_dc_n = v_dc + dt * dv_dc
        i_l_n = i_l + dt * di_l
        return (i_g_n, v_dc_n, i_l_n)

    def step_substeps(self, state0, v_gr_subseq, v_inv_subseq, t0, dt_sub):
        """
        Advance the plant over a PWM sub-step sequence (same idea as
        Load.calculateLoadDynamicsSubsteps).
        """
        assert len(v_gr_subseq) == len(v_inv_subseq)
        state = state0
        t = t0
        for v_gr, v_inv in zip(v_gr_subseq, v_inv_subseq):
            state = self.step_euler(state, v_gr, v_inv, t, dt_sub)
            t += dt_sub
        return state
