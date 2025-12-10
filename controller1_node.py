# controller1_node.py
import zmq
import numpy as np
from scipy.optimize import minimize

from comm_schema import make_envelope
from acdcac.fsclf import FiniteStepLyapunov

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    # *** SET YOUR COORDINATOR IP HERE ***
    # Use "127.0.0.1" if coordinator runs on the same PC.
    # Use the LAN IP of PC1 (e.g. "192.168.0.10") if this runs on another PC.
    coordinator_ip = "127.0.0.1"
    addr = f"tcp://{coordinator_ip}:5553"
    print(f"[C1] Connecting to coordinator at {addr}")
    sock.connect(addr)

    print("[C1] Socket type:", sock.getsockopt(zmq.TYPE))  # should be 3 (REQ)

    fsclf = FiniteStepLyapunov(x_eq=[0.0, 400.0, 0.0])
    u_prev = None

    while True:
        # 1) Send HELLO
        hello_msg = {"hello_from": "sub1"}
        print("[C1] Sending HELLO:", hello_msg)
        sock.send_json(hello_msg)

        # 2) Receive job from coordinator
        msg = sock.recv_json()
        print("[C1] Received job:", msg)

        if msg.get("msg_type") == "shutdown":
            print("[C1] Shutting down on coordinator request.")
            break

        step = msg["step"]
        outer_step = msg["outer_step"]
        Ts = msg["Ts"]
        M = msg["M"]
        payload = msg["payload"]

        x1 = (payload["state"]["i_g"], payload["state"]["v_dc"])
        i_l = payload["state"]["i_l"]
        i_l_bar = np.array(payload["neighbor_prediction"]["i_l_bar"], dtype=float)
        V0 = payload["fsclf"]["V0"]
        N = payload["horizon_N"]

        print(f"[C1] step={step}, outer_step={outer_step}, Ts={Ts}, M={M}, N={N}")
        print(f"[C1] x1={x1}, i_l={i_l}, V0={V0}")
        print(f"[C1] i_l_bar={i_l_bar}")

        # ---------------------------------------------------------
        # 3) Solve local OCP (placeholder)
        # ---------------------------------------------------------
        if u_prev is None:
            u_prev = np.zeros(N)

        def cost(u_seq):
            u_seq = np.asarray(u_seq, dtype=float)
            i_g, v_dc = x1
            J = 0.0
            for k in range(N):
                u_k = u_seq[k]
                V1 = fsclf.V_sub1((i_g, v_dc))
                J += V1 + 1e-2 * u_k**2 + 1e-1 * (u_k - u_prev[k])**2
            x1_pred = [x1] * N
            V0_sub = fsclf.V_sub1(x1)
            V_M_sub = fsclf.V_sub1(x1_pred[min(M-1, N-1)])
            return J, x1_pred, V0_sub, V_M_sub

        def obj(u_seq):
            J, _, _, _ = cost(u_seq)
            return J

        u0 = u_prev.copy()
        bounds = [(-1, 1)] * N

        print("[C1] Starting local optimization ...")
        res = minimize(obj, u0, method="trust-constr", bounds=bounds)
        if not res.success:
            print("[C1] Warning: solver did not converge:", res.message)
            u_opt = u_prev
        else:
            u_opt = res.x

        J1, x1_pred, V0_sub, V_M_sub = cost(u_opt)
        u_prev = u_opt

        print(f"[C1] Local cost J1={J1}, V0_sub={V0_sub}, V_M_sub={V_M_sub}")

        reply_payload = {
            "status": "ok",
            "local_cost": float(J1),
            "u_seq": u_opt.tolist(),
            "x1_pred": [
                {"i_g": float(xx[0]), "v_dc": float(xx[1])} for xx in x1_pred
            ],
            "contractive": {
                "V0_sub": float(V0_sub),
                "V_M_sub": float(V_M_sub),
                "alpha": payload["fsclf"]["alpha"],
                "satisfied": V_M_sub <= payload["fsclf"]["alpha"] * V0_sub,
            },
        }

        reply = make_envelope(
            msg_type="controller1_to_coord",
            sim_id=msg["sim_id"],
            sender="sub1",
            receiver="coordinator",
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=reply_payload,
        )

        # 4) Send reply back
        print("[C1] Sending reply to coordinator.")
        sock.send_json(reply)

        # 5) Wait for ACK to complete REQ/REP cycle
        ack = sock.recv_json()
        print("[C1] Received ACK from coordinator:", ack)


if __name__ == "__main__":
    main()
