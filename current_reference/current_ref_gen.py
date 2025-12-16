import math

class CurrentReference:
    def __init__(
        self,
        i_ref_peak: float,
        i_ref_freq: float,
        phi: float = 0.0,
        use_cos: bool = True,
        per_unit: bool = False
    ):
        """Sinusoidal current reference generator.

        Parameters
        ----------
        i_ref_peak : float
            Peak amplitude [A] (NOT RMS).
        i_ref_freq : float
            Frequency [Hz] if per_unit=False, otherwise normalized frequency.
        phi : float
            Phase shift [rad]. For grid PF control, phi = atan2(Q,P) is typical.
        use_cos : bool
            If True, uses cos(ωt - phi); else uses sin(ωt - phi).
            Note: your GridDCLink uses v_g(t)=V*cos(ωt), so cos() aligns current
            reference with grid voltage for unity power factor when phi=0.
        per_unit : bool
            If True, uses normalized time (ω=1).
        """
        self.i_ref_peak = float(i_ref_peak)
        self.i_ref_freq = float(i_ref_freq)
        self.phi = float(phi)
        self.use_cos = bool(use_cos)
        self.per_unit = bool(per_unit)

    def generateRefTrajectory(self, t: float):
        if self.per_unit:
            ang = float(t) - self.phi
        else:
            ang = 2.0 * math.pi * self.i_ref_freq * float(t) - self.phi

        val = self.i_ref_peak * (math.cos(ang) if self.use_cos else math.sin(ang))
        return [val]
