
import math
class CurrentReference:
    def __init__(self, i_ref_peak: float, i_ref_freq: float, per_unit: bool=False) -> float:
        self.i_ref_peak = i_ref_peak
        self.i_ref_freq = i_ref_freq
        self.per_unit = per_unit
    def generateRefTrajectory(self, t):
        if self.per_unit:
            return [ self.i_ref_peak * math.sin(t) ]  # OR math.sin(self.i_ref_freq*t) ?!
        else:
            return [ self.i_ref_peak * math.sin(2.0*math.pi*self.i_ref_freq*t) ]
