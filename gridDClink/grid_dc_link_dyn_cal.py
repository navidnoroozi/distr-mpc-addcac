
import math

class GridDCLink:
    def __init__(self, sampling_rate, Rg, Lg, Cdc, v_g_peak, v_g_freq, per_unit: bool=False):
        self.Ts = float(sampling_rate)
        self.Rg = float(Rg)
        self.Lg = float(Lg)
        self.Cdc = float(Cdc)
        self.V_g = float(v_g_peak)
        self.f_g = float(v_g_freq)
        self.per_unit = per_unit

    def _vg(self, t):
        if self.per_unit:
            return self.V_g * math.cos(t) # OR math.cos(2.0*math.pi*self.f_emf*t) ?!
        else:
            return self.V_g * math.cos(2.0*math.pi*self.f_g*t)
        

    def step_euler(self, x_g, x_l, v_gr_ctrl, t, dt):
        """
        Discrete-time model for Subsystem 1: x1 = [i_g, v_dc].

        Euler of some continuous dynamics with coupling to x2.
            di_g/dt = (-Rg * i_g + v_g - v_gr_ctrl) / Lg
            dv_dc/dt = (i_g - i_l) / Cdc
        """
        i_g, v_dc = x_g
        i_l = x_l

        di_g = (- self.Rg * i_g - v_gr_ctrl + self._vg(t)) / self.Lg
        dv_dc = (i_g - i_l) / self.Cdc
        return i_g + dt * di_g, v_dc + dt * dv_dc

    def calculateLoadDynamicsSubsteps(self, i_a_0, v_subseq, t_0, dt_sub):
        """
        Advance through a sub-stepped waveform (for PWM).
        """
        i = i_a_0
        t = t_0
        for v in v_subseq:
            i = self.step_euler(i, v, t, dt_sub)
            t += dt_sub
        return [i]  # return final current as a list for consistency
