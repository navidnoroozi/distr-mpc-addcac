
class CostFunction:
    def __init__(self, stage_func, subsystem: dict = None):
        if subsystem['name'] == 'load':
            self.stage_func_load = stage_func
        else:
            self.stage_func_grid = stage_func
            self.V_dc_ref = subsystem['V_dc_ref']

    def calculateCostFuncLoad(self, x_l_0, x_g_bar, t_0, u0, cont_horizon, u_seq, pwm, load, referenceTrajectory):
        """
        IMPORTANT CHANGE:
        Use the *averaged* model for prediction inside the cost (MPC prediction model).
        That is, we DO NOT simulate switching here. This keeps the optimization convex/smooth
        and aligns with standard averaged-model MPC.
        """
        cost_func = 0.0
        # Build voltage sequence using averaged model V = Vdc * u
        # v_seq = [pwm.average_voltage(u) for u in u_seq]

        # Predict current over the horizon (one step per control sample)
        x_l_pred = x_l_0
        _, v_dc_bar = x_g_bar
        t = t_0
        i_l_pred = [x_l_pred]
        i_l_ref = [referenceTrajectory.generateRefTrajectory(t)[0]]
        for k in range(cont_horizon):
            # Simple Euler using load.Ts as the prediction step
            # Here we assume load.Ts equals the control step; if not, adjust accordingly.
            x_l_pred = load.step_euler(x_l_pred, v_dc_bar[k], u_seq[k], t, load.Ts)  # v_seq -> u_seq here: u_seq[k] is the load-side inverter control input
            t += load.Ts

            # Stage cost with current reference at each step
            i_l_pred.append(x_l_pred)
            i_l_ref.append(referenceTrajectory.generateRefTrajectory(t)[0])
            cost_func += float(self.stage_func_load(i_l_pred[k], i_l_ref[k]))

        return cost_func, i_l_pred, i_l_ref

    def calculateCostFuncGrid(self, x_g_0, x_N_bar, t_0, u0, cont_horizon, 
                            u_seq, pwm, griddclink, referenceTrajectory):
        """
        IMPORTANT CHANGE:
        Use the *averaged* model for prediction inside the cost (MPC prediction model).
        That is, we DO NOT simulate switching here. This keeps the optimization convex/smooth
        and aligns with standard averaged-model MPC.
        """
        cost_func = 0.0
        # Build voltage sequence using averaged model V = Vdc * u
        # v_seq = [pwm.average_voltage(u) for u in u_seq]

        # Predict current over the horizon (one step per control sample)
        x_g_pred = x_g_0
        x_l_bar, u2_bar = zip(*x_N_bar)
        t = t_0
        i_g_pred = [x_g_pred[0]]
        v_dc_pred = [x_g_pred[1]]
        i_g_ref = [referenceTrajectory.generateRefTrajectory(t)[0]]
        for k in range(cont_horizon):
            # Simple Euler using griddclink.Ts as the prediction step
            # Here we assume griddclink.Ts equals the control step; if not, adjust accordingly.
            x_g_pred = griddclink.step_euler(x_g_pred, x_l_bar[k], u_seq[k], u2_bar[k], t, griddclink.Ts) # v_seq -> u_seq here: u_seq[k] is the grid-side inverter control input
            t += griddclink.Ts

            # Stage cost with current reference at each step
            i_g_pred.append(x_g_pred[0])
            v_dc_pred.append(x_g_pred[1])
            i_g_ref.append(referenceTrajectory.generateRefTrajectory(t)[0])
            cost_func += float(self.stage_func_grid(i_g_pred[k], i_g_ref[k], v_dc_pred[k], self.V_dc_ref))
        return cost_func, (i_g_pred, v_dc_pred), i_g_ref