# plant1_node.py
import zmq
from typing import Tuple
from comm_schema import make_envelope

def f1_discrete(x1, x2, u1, Ts):
    """
    Toy discrete-time model for Subsystem 1: x1 = [i_g, v_dc].
    Replace with your actual rectifier+dc-link discrete model.

    Example: Euler of some continuous dynamics with coupling to x2.
        i_g+  = i_g + Ts * (-a*i_g + b*(v_dc - k*x2 + u1))
        v_dc+ = v_dc + Ts * (c*i_g - d*x2)
    """
    i_g, v_dc = x1
    i_l = x2[0]
    a, b, c, d, k = 50.0, 100.0, 10.0, 5.0, 1.0

    di_g = -a * i_g + b * (v_dc - k * i_l + u1)
    dv_dc = c * i_g - d * i_l

    i_g_next = i_g + Ts * di_g
    v_dc_next = v_dc + Ts * dv_dc
    return (i_g_next, v_dc_next)

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    # Coordinator IP+port, adjust IP for your topology
    # If Plant1 runs on same PC as coordinator, 127.0.0.1 is fine.
    coordinator_ip = "127.0.0.1"
    addr = f"tcp://{coordinator_ip}:5551"
    print(f"[Plant1] Connecting to coordinator at {addr}")
    sock.connect(addr)

    # Local state
    x1 = (0.0, 400.0)
    Ts = 1e-4   # just to have it locally

    print("[Plant1] Started.")
    while True:
        # 1) send a "ready" ping to coordinator
        sock.send_json({"hello_from": "plant1"})

        # 2) receive job from coordinator
        msg = sock.recv_json()

        if msg.get("msg_type") == "shutdown":
            print("[Plant1] Shutting down.")
            break

        step = msg["step"]
        outer_step = msg["outer_step"]
        Ts = msg["Ts"]
        M = msg["M"]

        payload = msg["payload"]
        u1 = payload["u1"]           # local input
        x2 = payload["x2"]           # neighbor state, e.g. {"i_l": ...}

        x2_vec = (x2["i_l"],)
        x1_next = f1_discrete(x1, x2_vec, u1, Ts)
        x1 = x1_next

        reply_payload = {
            "x1_next": {
                "i_g": x1[0],
                "v_dc": x1[1],
            }
        }
        reply = make_envelope(
            msg_type="plant1_to_coord",
            sim_id=msg["sim_id"],
            sender="plant1",
            receiver="coordinator",
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=reply_payload,
        )

        # 3) send reply to coordinator
        sock.send_json(reply)

        # 4) wait for ACK to complete REQ/REP cycle
        ack = sock.recv_json()
        print("[Plant1] Received ACK from coordinator:", ack)

if __name__ == "__main__":
    main()
