# plant_node.py
import zmq, math
from typing import Tuple
from comm_schema import make_envelope
from gridDClink.grid_dc_link_dyn_cal import GridDCLink
from load.load_dyn_cal import Load

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    # Coordinator IP+port, adjust IP for your topology
    # If Plant runs on same PC as coordinator, 127.0.0.1 is fine.
    coordinator_ip = "127.0.0.1"
    addr = f"tcp://{coordinator_ip}:5551"
    print(f"[Plant1] Connecting to coordinator at {addr}")
    sock.connect(addr)

    ## Simulation parameters
    # Carrier frequency
    carrier_freq = 1e4 # in kHz
    # Overall sampling time
    sampling_time = 1/carrier_freq
    # Grid frequency in Hz
    f_grid = 50.0
    # Grid / DC Source Voltage
    V_rms_req = 230.0  # RMS voltage in V
    V_dc = V_rms_req * math.sqrt(2)  # V

    # Grid + DC Link parameters
    Lg=10e-3
    Rg=10.0
    Cdc=5e-3
    griddclink = GridDCLink(sampling_time, Rg, Lg, Cdc, V_dc, f_grid, per_unit=False)

    # Load parameters
    f_load = f_grid
    Ll=5e-3
    Rl=5.0
    back_emf_peak = 0
    load = Load(sampling_time, Rl, Ll, back_emf_peak, f_load, per_unit=False)

    # Local state
    x1 = (0.0, V_dc)  # i_g, v_dc
    x2 = 0.0  # i_l
    Ts = 1e-4   # just to have it locally

    print("[Plant1] Started.")
    while True:
        # 1) Send a "ready" ping to coordinator
        sock.send_json({"hello_from": "plant"})

        # 2) Receive job from coordinator
        msg = sock.recv_json()

        if msg.get("msg_type") == "shutdown":
            print("[Plant] Shutting down.")
            break

        step = msg["step"]
        outer_step = msg["outer_step"]
        Ts = msg["Ts"]
        M = msg["M"]

        payload = msg["payload"]
        u1 = payload["u1"]           # local input
        u2 = payload["u2"]           # neighbor state, e.g. {"i_l": ...}

        x1_next = griddclink.step_euler(x1, x2, u1, step*Ts, Ts)
        x2_next = load.step_euler(x2, u2, step*Ts, Ts)
        x1 = x1_next
        x2 = x2_next

        reply_payload = {
            "x1_next": {
                "i_g": x1[0],
                "v_dc": x1[1],
            },
            "x2_next": {
                "i_l": x2,
            }
        }
        reply = make_envelope(
            msg_type="plant_to_coord",
            sim_id=msg["sim_id"],
            sender="plant",
            receiver="coordinator",
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=reply_payload,
        )

        # 3) Send reply to coordinator
        print("[P] Sending reply to coordinator.")
        sock.send_json(reply)

        # 4) wait for ACK to complete REQ/REP cycle
        ack = sock.recv_json()
        print("[Plant] Received ACK from coordinator:", ack)

if __name__ == "__main__":
    main()
