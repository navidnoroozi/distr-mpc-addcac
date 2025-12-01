# acdcac/fsclf.py

import numpy as np


class FiniteStepLyapunov:
    """
    Finite-step control Lyapunov function for the interconnected system

        x = (x1, x2) = ( [i_g, v_dc], [i_l] ).

    Theoretical structure (matches your distributed MPC theory):

        - Local Lyapunov-like functions:
              W1 : R^2 -> R_{\ge 0}
              W2 : R^1 -> R_{\ge 0}

        - Small-gain scaling functions σ1, σ2 ∈ K∞.

        - Global fs-CLF:
              V(x) = max_i σ_i^{-1}( W_i(x_i) ).

    Here we implement:

        W1(x1) = (x1 - x1_eq)^T P1 (x1 - x1_eq),
        W2(x2) = (x2 - x2_eq)^T P2 (x2 - x2_eq),

        σ1(r) = c1 * r,  σ2(r) = c2 * r   (c1,c2 > 0),

        so that

            σ1^{-1}(s) = s / c1,
            σ2^{-1}(s) = s / c2,

        and

            V(x) = max( W1(x1)/c1,  W2(x2)/c2 ).

    You can plug in:
        - P1, P2 from local discrete-time Lyapunov equations
          of your linearized subsystems,
        - c1, c2 chosen to satisfy your small-gain inequalities.
    """

    def __init__(
        self,
        x_eq=None,
        P1=None,
        P2=None,
        c1=1.0,
        c2=1.0,
    ):
        """
        Parameters
        ----------
        x_eq : array-like, shape (3,)
            Equilibrium state [i_g*, v_dc*, i_l*].
            Default: [0, 400, 0].

        P1 : array-like, shape (2,2)
            Positive definite matrix for W1.  If None, uses identity.

        P2 : array-like, shape (1,1) or scalar
            Positive definite matrix for W2.  If None, uses 1.0.

        c1, c2 : float
            Positive scalings defining σ1(r) = c1 * r, σ2(r) = c2 * r.
            These are where you encode the small-gain scalings σ_i.
        """
        if x_eq is None:
            x_eq = [0.0, 400.0, 0.0]  # typical converter operating point
        self.x_eq = np.asarray(x_eq, dtype=float).reshape(3)

        # Local equilibrium components
        self.x1_eq = self.x_eq[0:2]  # (i_g*, v_dc*)
        self.x2_eq = self.x_eq[2:3]  # (i_l*)

        # Local P-matrices
        if P1 is None:
            self.P1 = np.eye(2)  # default: ||x1 - x1_eq||^2
        else:
            self.P1 = np.asarray(P1, dtype=float).reshape(2, 2)

        if P2 is None:
            # scalar case => 1x1 matrix
            self.P2 = np.array([[1.0]], dtype=float)
        else:
            P2_arr = np.asarray(P2, dtype=float)
            if P2_arr.ndim == 0:
                self.P2 = np.array([[float(P2_arr)]], dtype=float)
            else:
                self.P2 = P2_arr.reshape(1, 1)

        # σ_i(r) = c_i * r
        assert c1 > 0 and c2 > 0, "c1,c2 must be positive."
        self.c1 = float(c1)
        self.c2 = float(c2)

    # ------------------------------------------------------------------
    # Local Lyapunov-like functions W1, W2
    # ------------------------------------------------------------------

    def W1(self, x1):
        """
        Local CLF candidate for Subsystem 1: x1 = (i_g, v_dc).

        W1(x1) = (x1 - x1_eq)^T P1 (x1 - x1_eq).
        """
        x1 = np.asarray(x1, dtype=float).reshape(2)
        e1 = x1 - self.x1_eq
        return float(e1 @ (self.P1 @ e1))

    def W2(self, x2):
        """
        Local CLF candidate for Subsystem 2: x2 = (i_l,).

        W2(x2) = (x2 - x2_eq)^T P2 (x2 - x2_eq).
        """
        x2 = np.asarray(x2, dtype=float).reshape(1)
        e2 = x2 - self.x2_eq
        return float(e2 @ (self.P2 @ e2))

    # ------------------------------------------------------------------
    # σ_i and σ_i^{-1}
    # ------------------------------------------------------------------

    def sigma1(self, r):
        return self.c1 * float(r)

    def sigma2(self, r):
        return self.c2 * float(r)

    def sigma1_inv(self, s):
        return float(s) / self.c1

    def sigma2_inv(self, s):
        return float(s) / self.c2

    # ------------------------------------------------------------------
    # Global fs-CLF V
    # ------------------------------------------------------------------

    def V(self, x):
        """
        Global fs-CLF:

            V(x) = max( σ1^{-1}(W1(x1)), σ2^{-1}(W2(x2)) ),

        with x = (i_g, v_dc, i_l).
        """
        x = np.asarray(x, dtype=float).reshape(3)
        x1 = x[0:2]
        x2 = x[2:3]

        W1_val = self.W1(x1)
        W2_val = self.W2(x2)

        V1 = self.sigma1_inv(W1_val)
        V2 = self.sigma2_inv(W2_val)

        return float(max(V1, V2))

    # Convenience wrappers used by the distributed MPC code
    def V_sub1(self, x1):
        """
        Local contribution for Subsystem 1, consistent with the global V:

            V1(x1) := σ1^{-1}( W1(x1) ).

        This is what appears in the local contractive constraints / costs.
        """
        W1_val = self.W1(x1)
        return self.sigma1_inv(W1_val)

    def V_sub2(self, x2):
        """
        Local contribution for Subsystem 2, consistent with the global V:

            V2(x2) := σ2^{-1}( W2(x2) ).
        """
        W2_val = self.W2(x2)
        return self.sigma2_inv(W2_val)
