# controller2_node.py
import zmq
import numpy as np
from scipy.optimize import minimize

from comm_schema import make_envelope
from acdcac.fsclf import FiniteStepLyapunov

def solve_local_ocp_sub2(x2, x1_bar, N, M, V0, Ts, u_prev, fsclf, currentReference, t0):
    """
    Local OCP-2 for Subsystem 2.
    Placeholder: track i_l_ref, penalize CLF, input.
    """
    if u_prev is None:
        u_prev = np.zeros(N)
    u_prev = np.asarray(u_prev, dtype=float)

    def cost(u_seq):
        u_seq = np.asarray(u_seq, dtype=float)
        i_l = x2[0]
        t = t0
        J = 0.0
        for k in range(N):
            u_k = u_seq[k]
            i_l_ref = currentReference.generateRefTrajectory(t)[0]
            J += (i_l - i_l_ref)**2 + 1e-2 * u_k**2 + 1e-1 * (u_k - u_prev[k])**2
            t += Ts
        x2_pred = [x2] * N
        V0_sub = fsclf.V_sub2(x2)
        V_M_sub = fsclf.V_sub2(x2_pred[min(M-1, N-1)])
        return J, x2_pred, V0_sub, V_M_sub

    def obj(u_seq):
        J, _, _, _ = cost(u_seq)
        return J

    u0 = u_prev.copy()
    bounds = [(-1, 1)] * N
    res = minimize(obj, u0, method="trust-constr", bounds=bounds)
    if not res.success:
        u_opt = u_prev
    else:
        u_opt = res.x
    J, x2_pred, V0_sub, V_M_sub = cost(u_opt)
    return u_opt, x2_pred, J, V0_sub, V_M_sub

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    # *** SET YOUR COORDINATOR IP HERE ***
    # If this runs on PC2 and coordinator on PC1, use PC1's LAN IP, e.g. "192.168.0.10".
    coordinator_ip = "127.0.0.1"
    addr = f"tcp://{coordinator_ip}:5554"
    print(f"[Controller2] Connecting to coordinator at {addr}")
    sock.connect(addr)

    fsclf = FiniteStepLyapunov(x_eq=[0.0, 400.0, 0.0])

    # Here you’d construct the same CurrentReference you use in your app
    from current_reference.current_ref_gen import CurrentReference
    currentReference = CurrentReference(i_ref_peak=10.0, f_ref=50.0, per_unit=False)

    u_prev = None

    print("[Controller2] Started.")
    while True:
        # 1) Send HELLO
        sock.send_json({"hello_from": "sub2"})

        # 2) Receive job from coordinator
        msg = sock.recv_json()

        if msg.get("msg_type") == "shutdown":
            print("[Controller2] Shutting down.")
            break

        step = msg["step"]
        outer_step = msg["outer_step"]
        Ts = msg["Ts"]
        M = msg["M"]
        payload = msg["payload"]

        t0 = payload["t_sim"]
        i_g = payload["state"]["i_g"]
        v_dc = payload["state"]["v_dc"]
        i_l = payload["state"]["i_l"]
        x2 = (i_l,)
        x1_bar_dicts = payload["neighbor_prediction"]["x1_bar"]
        x1_bar = [(d["i_g"], d["v_dc"]) for d in x1_bar_dicts]
        V0 = payload["fsclf"]["V0"]
        N = payload["horizon_N"]

        u_opt, x2_pred, J2, V0_sub, V_M_sub = solve_local_ocp_sub2(
            x2, x1_bar, N, M, V0, Ts, u_prev, fsclf, currentReference, t0
        )
        u_prev = u_opt

        reply_payload = {
            "status": "ok",
            "local_cost": float(J2),
            "u_seq": u_opt.tolist(),
            "i_l_pred": [float(xx[0]) for xx in x2_pred],
            "contractive": {
                "V0_sub": float(V0_sub),
                "V_M_sub": float(V_M_sub),
                "alpha": payload["fsclf"]["alpha"],
                "satisfied": V_M_sub <= payload["fsclf"]["alpha"] * V0_sub,
            },
        }
        reply = make_envelope(
            msg_type="controller2_to_coord",
            sim_id=msg["sim_id"],
            sender="sub2",
            receiver="coordinator",
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=reply_payload,
        )

        # 3) Send reply
        sock.send_json(reply)

        # 4) Wait for ACK to complete REQ/REP cycle
        ack = sock.recv_json()
        print("[Controller2] Received ACK from coordinator:", ack)

if __name__ == "__main__":
    main()
