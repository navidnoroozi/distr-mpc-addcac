# plant2_node.py
import zmq
from comm_schema import make_envelope

def f2_discrete(x2, x1, u2, Ts):
    """
    Toy discrete-time model for Subsystem 2: x2 = [i_l].
    Replace with your actual inverter discrete model.
    Example:
        i_l+ = i_l + Ts * (-a*i_l + b*(v_dc*u2 - R*i_l))
    """
    i_l = x2[0]
    i_g, v_dc = x1
    a = 50.0
    b = 100.0

    di_l = -a * i_l + b * (v_dc * u2 - 0.1 * i_l)
    i_l_next = i_l + Ts * di_l
    return (i_l_next,)

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    # *** SET YOUR COORDINATOR IP HERE ***
    # If Plant2 runs on PC2 and coordinator on PC1, use PC1's LAN IP here.
    coordinator_ip = "127.0.0.1"
    addr = f"tcp://{coordinator_ip}:5552"
    print(f"[Plant2] Connecting to coordinator at {addr}")
    sock.connect(addr)

    x2 = (0.0,)
    Ts = 1e-4

    print("[Plant2] Started.")
    while True:
        # 1) send HELLO / ready ping
        sock.send_json({"hello_from": "plant2"})

        # 2) receive job from coordinator
        msg = sock.recv_json()

        if msg.get("msg_type") == "shutdown":
            print("[Plant2] Shutting down.")
            break

        step = msg["step"]
        outer_step = msg["outer_step"]
        Ts = msg["Ts"]
        M = msg["M"]
        payload = msg["payload"]
        u2 = payload["u2"]
        x1 = payload["x1"]  # {"i_g": ..., "v_dc": ...}

        x1_vec = (x1["i_g"], x1["v_dc"])
        x2_next = f2_discrete(x2, x1_vec, u2, Ts)
        x2 = x2_next

        reply_payload = {
            "x2_next": {
                "i_l": x2[0],
            }
        }
        reply = make_envelope(
            msg_type="plant2_to_coord",
            sim_id=msg["sim_id"],
            sender="plant2",
            receiver="coordinator",
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=reply_payload,
        )

        # 3) send reply
        sock.send_json(reply)

        # 4) wait for ACK to complete REQ/REP cycle
        ack = sock.recv_json()
        print("[Plant2] Received ACK from coordinator:", ack)

if __name__ == "__main__":
    main()
