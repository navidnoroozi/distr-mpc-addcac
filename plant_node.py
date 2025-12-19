# plant_node.py
import zmq, math
from comm_schema import make_envelope
from gridDClink.grid_dc_link_dyn_cal import GridDCLink
from load.load_dyn_cal import Load
from pwm.pwm_gen import PWM

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    # Coordinator IP+port, adjust IP for your topology
    # If Plant runs on same PC as coordinator, 127.0.0.1 is fine.
    coordinator_ip = "127.0.0.1"
    addr = f"tcp://{coordinator_ip}:5551"
    print(f"[Plant] Connecting to coordinator at {addr}")
    sock.connect(addr)

    # --- base simulation parameters (must match controllers) ---
    carrier_freq = 1e4
    sampling_time = 1.0 / carrier_freq
    f_grid = 50.0
    V_rms_req = 230.0
    V_dc0 = V_rms_req * math.sqrt(2)

    # Grid + DC Link parameters
    Lg = 10e-3
    Rg = 1.0
    Cdc = 5e-3
    griddclink = GridDCLink(sampling_time, Rg, Lg, Cdc, V_dc0, f_grid, per_unit=False)

    # Load parameters
    f_load = f_grid
    Ll = 10e-3
    Rl = 1.0
    back_emf_peak = 0.0
    load = Load(sampling_time, Rl, Ll, back_emf_peak, f_load, per_unit=False)

    # PWM blocks (full bridge, bipolar) for grid converter and load converter
    pwm_grid = PWM(carrier_freq, sampling_time, V_dc0, tech_type="FB", per_unit=False, modulation="bipolar")
    pwm_load = PWM(carrier_freq, sampling_time, V_dc0, tech_type="FB", per_unit=False, modulation="bipolar")

    # Local state
    x1 = (0.0, V_dc0)  # (i_g, v_dc)
    x2 = 0.0           # i_l

    print("[Plant] Started.")
    while True:
        # 1) Send ready ping
        sock.send_json({"hello_from": "plant"})

        # 2) Receive job from coordinator
        msg = sock.recv_json()
        if msg.get("msg_type") == "shutdown":
            print("[Plant] Shutting down.")
            break

        step = int(msg["step"])
        outer_step = int(msg["outer_step"])
        Ts = float(msg["Ts"])
        M = int(msg["M"])
        payload = msg["payload"]
        u1 = float(payload["u1"])   # grid-side modulation input in [-1,1]
        u2 = float(payload["u2"])   # load-side modulation input in [-1,1]

        t0 = step * Ts

        # Update PWM DC-link value to the *current* v_dc
        pwm_grid.Vdc = float(x1[1])
        pwm_load.Vdc = float(x1[1])

        # Synthesize switching waveforms over this control step Ts
        v1_seq, dt_sub, s1_seq, gates1 = pwm_grid.synthesize_over_interval(u1, t0, Ts=Ts, return_gates=True)
        v2_seq, dt_sub2, s2_seq, gates2 = pwm_load.synthesize_over_interval(u2, t0, Ts=Ts, return_gates=True)
        # ensure dt match (they should, if settings identical)
        if abs(dt_sub2 - dt_sub) > 1e-15:
            # resample to the smaller dt by re-synthesizing; simplest is re-run with same min_step_samples
            v2_seq, dt_sub, s2_seq, gates2 = pwm_load.synthesize_over_interval(
                u2, t0, Ts=Ts, min_carrier_samples=20, min_step_samples=max(200, len(v1_seq)), return_gates=True
            )

        # Integrate the coupled plant using substeps
        i_g, v_dc = x1
        i_l = x2
        t = t0
        i_g, v_dc = griddclink.step_euler((i_g, v_dc), i_l, u1, u2, t, dt_sub)
        i_l = load.step_euler(i_l, v_dc, u2, t, dt_sub)

        x1 = (float(i_g), float(v_dc))
        x2 = float(i_l)

        # Provide last switching states for plotting (sampled at end of Ts)
        s1_last = int(s1_seq[-1]) if s1_seq else 0
        s2_last = int(s2_seq[-1]) if s2_seq else 0

        gates1_last = {k: int(v[-1]) for k, v in (gates1 or {}).items()}
        gates2_last = {k: int(v[-1]) for k, v in (gates2 or {}).items()}

        reply_payload = {
            "x1_next": {"i_g": x1[0], "v_dc": x1[1]},
            "x2_next": {"i_l": x2},
            "switching": {
                "s1": s1_last,
                "s2": s2_last,
                "gates1": gates1_last,
                "gates2": gates2_last,
            },
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

        # 4) Receive ACK to complete REQ cycle
        ack = sock.recv_json()
        print("[Plant] Received ACK from coordinator:", ack)

if __name__ == "__main__":
    main()
