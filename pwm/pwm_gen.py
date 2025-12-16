
import math
from typing import Dict, List, Tuple, Optional

class PWM:
    def __init__(self, carrier_freq, Ts, Vdc, tech_type: str = 'HB', per_unit: bool=False, modulation: str="bipolar"):
        """Triangular-carrier PWM.

        Supported:
          - tech_type='HB' : half-bridge leg, bipolar output v in {+Vdc, -Vdc}
          - tech_type='FB' : full-bridge, default **bipolar** modulation producing v_ab in {+Vdc, -Vdc}

        For plotting/debug you can request gate signals:
          Sa_p, Sa_n, Sb_p, Sb_n  (upper/lower devices of leg A and B)

        Notes:
          - This is an *ideal* PWM (no dead-time, no device drops).
          - For the MPC prediction model you should continue using average_voltage(u).
        """
        self.fc = float(carrier_freq)
        self.Ts = float(Ts)
        self.Vdc = float(Vdc)
        self.per_unit = bool(per_unit)
        self.tech_type = str(tech_type).upper()
        self.modulation = str(modulation).lower()

    def _triangle_01(self, t: float) -> float:
        """Triangle in [0,1] with period 1/fc."""
        Tc = 1.0 / self.fc
        tau = t % Tc
        half = 0.5 * Tc
        k = math.floor(t / Tc)
        if tau <= half:
            return (t - k*Tc) / half
        else:
            return -(t - (k+1)*Tc) / half

    def _u_to_d(self, u: float) -> float:
        """Map u in [-1,1] to duty in [0,1]."""
        d = 0.5 * (float(u) + 1.0)
        if d < 0.0: d = 0.0
        if d > 1.0: d = 1.0
        return d

    def _full_bridge_bipolar_gates(self, gate_high: int) -> Dict[str, int]:
        """Full-bridge bipolar gating:
        - Use one comparator result gate_high in {0,1}
        - Leg A and Leg B are complementary to generate v_ab in {+Vdc, -Vdc}
        """
        Sa_p = gate_high
        Sa_n = 1 - gate_high
        Sb_p = 1 - gate_high
        Sb_n = gate_high
        return {"Sa_p": Sa_p, "Sa_n": Sa_n, "Sb_p": Sb_p, "Sb_n": Sb_n}

    def _hb_bipolar_gates(self, gate_high: int) -> Dict[str, int]:
        """Half-bridge single-leg gating (upper/lower)."""
        return {"S_p": gate_high, "S_n": 1 - gate_high}

    def synthesize_over_interval(
        self,
        u: float,
        t0: float,
        Ts: Optional[float]=None,
        min_carrier_samples: int=20,
        min_step_samples: int=200,
        return_gates: bool=True,
    ) -> Tuple[List[float], float, List[int], Optional[Dict[str, List[int]]]]:
        """Generate switching waveform over [t0, t0+Ts).

        Returns:
          v_list: list of inverter output voltage samples at substep grid
          dt_sub: substep size
          s_list: list of line switching states (+1/-1) consistent with v_list = s*Vdc (FB bipolar) or s*Vdc (HB bipolar)
          gates:  dict of gate traces (lists), or None if return_gates=False

        For FB+bipolar: v_ab = s * Vdc with s in {+1,-1}.
        For HB+bipolar: v = s * Vdc with s in {+1,-1}.
        """
        if Ts is None:
            Ts = self.Ts

        T_car = 1.0 / self.fc
        dt_car = T_car / float(max(1, min_carrier_samples))
        dt_step = Ts / float(max(1, min_step_samples))
        dt_sub = min(dt_car, dt_step)
        if dt_sub <= 0.0:
            dt_sub = Ts
        n = max(1, int(math.ceil(Ts / dt_sub)))
        dt_sub = Ts / n  # exact division

        d = self._u_to_d(u)

        v_list: List[float] = []
        s_list: List[int] = []

        gates: Optional[Dict[str, List[int]]] = {} if return_gates else None
        if return_gates:
            if self.tech_type == "FB":
                for k in ["Sa_p","Sa_n","Sb_p","Sb_n"]:
                    gates[k] = []
            else:
                for k in ["S_p","S_n"]:
                    gates[k] = []

        t = t0
        for _ in range(n):
            carrier = self._triangle_01(t)
            gate_high = 1 if d > carrier else 0

            if self.tech_type == "FB":
                if self.modulation != "bipolar":
                    # Not implemented in this project code yet; keep deterministic.
                    # Fall back to bipolar.
                    pass
                g = self._full_bridge_bipolar_gates(gate_high)
                # v_ab = (Sa_p - Sb_p) * Vdc, where each is 0/1
                v = (g["Sa_p"] - g["Sb_p"]) * self.Vdc  # yields ±Vdc
                s = 1 if v >= 0 else -1
                if return_gates:
                    for k in ["Sa_p","Sa_n","Sb_p","Sb_n"]:
                        gates[k].append(int(g[k]))
            else:  # HB
                g = self._hb_bipolar_gates(gate_high)
                v = (2*gate_high - 1) * self.Vdc
                s = 1 if gate_high == 1 else -1
                if return_gates:
                    for k in ["S_p","S_n"]:
                        gates[k].append(int(g[k]))

            v_list.append(float(v))
            s_list.append(int(s))
            t += dt_sub

        return v_list, dt_sub, s_list, gates

    def average_voltage(self, u: float) -> float:
        """Ideal averaged model voltage.
        - HB: E[v] = Vdc*u
        - FB (bipolar): E[v_ab] = Vdc*u
        """
        return self.Vdc * float(u)
