
class CostFunction:
    def __init__(self, stage_func):
        self.stage_func = stage_func

    def calculateCostFunc(self, i_a_0, t_0, u0, cont_horizon, u_seq, pwm, load, currentReference):
        """
        IMPORTANT CHANGE:
        Use the *averaged* model for prediction inside the cost (MPC prediction model).
        That is, we DO NOT simulate switching here. This keeps the optimization convex/smooth
        and aligns with standard averaged-model MPC.
        """
        cost_func = 0.0
        # Build voltage sequence using averaged model V = Vdc * u
        v_seq = [pwm.average_voltage(u) for u in u_seq]

        # Predict current over the horizon (one step per control sample)
        i_pred = i_a_0
        t = t_0
        for k in range(cont_horizon):
            # Simple Euler using load.Ts as the prediction step
            # Here we assume load.Ts equals the control step; if not, adjust accordingly.
            i_pred = load.step_euler(i_pred, v_seq[k], t, load.Ts)
            t += load.Ts

            # Stage cost with current reference at each step
            i_ref = currentReference.generateRefTrajectory(t)[0]
            u_seq_current = u_seq[k]
            u_seq_last = u0[k]
            cost_func += float(self.stage_func(i_pred, i_ref, u_seq_current, u_seq_last))

        return cost_func
