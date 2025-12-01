from dataclasses import dataclass
import math
from typing import Tuple

@dataclass
class PuValues:
    f_req: float
    carrier_freq: float
    t_0: float
    Ts: float
    sim_time: float
    R: float
    L: float
    e_peak: float
    Vdc: float
    i_ref_peak: float
    i_a_0: float

@dataclass
class PerUnitBases:
    """
    Single‑phase per‑unit base definitions for a two‑level converter.

    Choose (S_base, V_rms_base, Vdc_base, f_base). All other bases are derived.
    Time is normalized with t_base = 1 / omega_base, so t_pu = t / t_base = t * omega_base.
    Currents are normalized with RMS current base; for peak currents use I_peak_base = sqrt(2)*I_base.
    Voltages: use RMS or peak bases consistently. Here we expose both for convenience.
    """
    S_base: float           # [VA] apparent power base (single‑phase)
    V_rms_base: float       # [V_rms] AC RMS base
    Vdc_base: float         # [V] DC‑link base
    f_base: float           # [Hz] fundamental frequency base

    def __post_init__(self) -> None:
        # Fundamental quantities
        self.omega_base: float = 2.0 * math.pi * float(self.f_base)
        self.t_base: float = 1.0 / self.omega_base

        # Voltage bases
        self.V_rms_base = float(self.V_rms_base)
        self.V_peak_base: float = self.V_rms_base * math.sqrt(2.0)

        # Current / impedance bases
        self.S_base = float(self.S_base)
        self.I_base: float = self.S_base / self.V_rms_base            # RMS current base
        self.I_peak_base: float = self.I_base * math.sqrt(2.0)        # Peak current base

        self.Z_base: float = (self.V_rms_base ** 2) / self.S_base
        self.R_base: float = self.Z_base
        self.L_base: float = self.Z_base / self.omega_base
        self.C_base: float = 1.0 / (self.Z_base * self.omega_base)

        # DC base
        self.Vdc_base = float(self.Vdc_base)

    # ---------- scalar helpers ----------
    def f_to_pu(self, f: float) -> float:
        """Frequency per‑unit (relative to f_base)."""
        return float(f) / self.f_base

    def t_to_pu(self, t: float) -> float:
        """Time per‑unit using t_base = 1/omega_base."""
        return float(t) / self.t_base  # == t * omega_base

    def t_from_pu(self, t_pu: float) -> float:
        return float(t_pu) * self.t_base

    def v_rms_to_pu(self, v_rms: float) -> float:
        return float(v_rms) / self.V_rms_base

    def v_peak_to_pu(self, v_peak: float) -> float:
        return float(v_peak) / self.V_peak_base

    def vdc_to_pu(self, vdc: float) -> float:
        return float(vdc) / self.Vdc_base

    def i_rms_to_pu(self, i_rms: float) -> float:
        return float(i_rms) / self.I_base

    def i_peak_to_pu(self, i_peak: float) -> float:
        return float(i_peak) / self.I_peak_base

    def R_to_pu(self, R: float) -> float:
        return float(R) / self.R_base

    def L_to_pu(self, L: float) -> float:
        return float(L) / self.L_base

    def C_to_pu(self, C: float) -> float:
        return float(C) / self.C_base

    # ---------- batch conversion used by main ----------
    def convert_2_pu_values(
        self,
        f_req: float,
        carrier_freq: float,
        t_0: float,
        sampling_time: float,
        sim_time: float,
        resistance: float,
        inductance: float,
        back_emf_peak: float,
        Vdc: float,
        i_ref_peak: float,
        i_a_0: float
    ) -> Tuple[float,...]:
        """
        Converts a typical set of inputs to per‑unit.
        IMPORTANT: sampling_time and sim_time are **seconds**; we convert with t_base (multiply by omega).
        Currents and back‑emf are **peak** values (consistent with many control implementations).
        """
        f_req_pu = self.f_to_pu(f_req)
        # It is fine to express the carrier relative to f_base if you truly need it in p.u.
        carrier_freq_pu = self.f_to_pu(carrier_freq)

        t_0_pu = self.t_to_pu(t_0)
        Ts_pu = self.t_to_pu(sampling_time)
        sim_time_pu = self.t_to_pu(sim_time)

        R_pu = self.R_to_pu(resistance)
        L_pu = self.L_to_pu(inductance)

        e_peak_pu = self.v_peak_to_pu(back_emf_peak)
        Vdc_pu = self.vdc_to_pu(Vdc)

        i_ref_peak_pu = self.i_peak_to_pu(i_ref_peak)
        i_a_0_pu = self.i_peak_to_pu(i_a_0)

        return PuValues(f_req_pu, carrier_freq_pu, t_0_pu, Ts_pu, sim_time_pu,
                R_pu, L_pu, e_peak_pu, Vdc_pu, i_ref_peak_pu, i_a_0_pu)
