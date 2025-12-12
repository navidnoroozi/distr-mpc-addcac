
from scipy.optimize import minimize

class MPCSSolver:
    def __init__(self, cost_func):
        self.cost_func = cost_func

    def solveMPC(self, pwm, plant, referenceTrajectory, t_0, x_0, x_N_bar, cont_horizon=10, u0=None, subsystem: str='load'):
        if u0 is None:
            u0 = [0.0] * cont_horizon
        if subsystem == 'load':
            def objective(u_seq):
                J, _, _ = self.cost_func.calculateCostFuncLoad(x_0, x_N_bar, t_0, u0, cont_horizon, 
                                                               u_seq, pwm, plant, referenceTrajectory)
                return J
        else:  # 'grid'
            def objective(u_seq):
                J, _, _ = self.cost_func.calculateCostFuncGrid(x_0, x_N_bar, t_0, u0, cont_horizon, 
                                                           u_seq, pwm, plant, referenceTrajectory)
                return J

        bounds = [(-1, 1)] * cont_horizon
        res = minimize(objective, u0, method='trust-constr', bounds=bounds)

        if not res.success:
            return u0, objective(u0)
        return res.x, res.fun