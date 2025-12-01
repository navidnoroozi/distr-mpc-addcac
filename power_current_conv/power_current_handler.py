import math

class RequiredPowerCurrentHandler:
    def __init__(self, P_req = 3e3, Q_req = 0.0, V_rms_req = 230.0):
        """
        Initializes the Power and Current Handler.

        Parameters:
        - P_req: Required active power (W).
        - Q_req: Required reactive power (VAR).
        - V_rms_req: Required RMS voltage (V).
        """
        self.P_req = P_req
        self.Q_req = Q_req
        self.V_rms_req = V_rms_req

    def calculateCurrentMagnitudeAndPhase(self):
        """
        Calculates the required peak current magnitude and the phase angle based on the power requirements.

        Returns:
        - Required peak current magnitude (A).
        - Phase angle (radians).
        Note:
        The returned current is the peak value, not RMS.
        """
        S_req = math.sqrt(self.P_req**2 + self.Q_req**2)  # Apparent power in VA
        I_peak_req = S_req / self.V_rms_req  # Peak current magnitude in A
        phi_req = math.atan2(self.Q_req, self.P_req)  # Phase angle in radians
        return I_peak_req, phi_req