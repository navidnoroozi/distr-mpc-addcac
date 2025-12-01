
import math

class PWM:
    def __init__(self, carrier_freq, Ts, Vdc, per_unit: bool=False):
        """
        Triangular-carrier PWM for a single leg (bipolar ±Vdc).
        carrier_freq: Hz
        Ts: controller/simulation step used by the outer loop [s]
        Vdc: DC link voltage [V]
        
        """
        self.fc = float(carrier_freq)
        self.Ts = float(Ts)
        self.Vdc = float(Vdc)
        self.per_unit = bool(per_unit)

    def _triangle_01(self, t):
        """Triangle in [0,1] with period 1/fc."""
        Tc = 1.0 / self.fc
        tau = t % Tc  # t mod Tc
        half = 0.5 * Tc # to determine if t is the first half or second half
        k = t // Tc   # the floor division // rounds the result down to the nearest whole number
        if tau <= half:
            return (t - k*Tc) / half
        else:
            return -(t - (k+1)*Tc) / half

    def _u_to_d(self, u):
        d = 0.5 * (float(u) + 1.0)
        if d < 0.0: d = 0.0
        if d > 1.0: d = 1.0
        return d

    def synthesize_over_interval(self, u, t0, Ts=None, min_carrier_samples=20, min_step_samples=200):
        """
        Generate a switching voltage waveform v(t) over [t0, t0+Ts) with sub-steps.
        Returns (v_list, dt_sub), where dt_sub is the sub-step size used.
        We choose dt_sub to satisfy BOTH:
          - at least 'min_carrier_samples' samples per carrier period
          - at least 'min_step_samples' samples per interval Ts
        """
        if Ts is None:
            Ts = self.Ts
        # substep based on carrier
        T_car = 1.0 / self.fc
        dt_car = T_car / float(min_carrier_samples)
        dt_step = Ts / float(min_step_samples)
        dt_sub = min(dt_car, dt_step)
        if dt_sub <= 0.0:
            dt_sub = Ts
        n = max(1, int(math.ceil(Ts / dt_sub)))
        dt_sub = Ts / n  # make it divide Ts exactly

        d = self._u_to_d(u)
        v_list = []
        t = t0
        for _ in range(n):
            carrier = self._triangle_01(t)
            gate_high = 1 if d > carrier else 0
            v = (2*gate_high - 1) * self.Vdc  # bipolar ±Vdc
            v_list.append(v)
            t += dt_sub
        return v_list, dt_sub

    def average_voltage(self, u):
        """Ideal averaged model: E[v] over one PWM period equals Vdc * u."""
        return self.Vdc * float(u)
