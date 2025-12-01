# coordinator_c1.py (excerpt)
import zmq
from comm_schema import make_envelope

context = zmq.Context()
sock_c1 = context.socket(zmq.REP)
sock_c1.bind("tcp://0.0.0.0:5553")

def handle_controller1(step, outer_step, Ts, M, N, state, i_l_bar, V0):
    # 1) Wait for HELLO (REQ side send_json)
    hello = sock_c1.recv_json()
    print("[Coord] C1 hello:", hello)

    # 2) Send job to controller
    payload_c1 = {
        "state": state,  # e.g. {"i_g":..., "v_dc":..., "i_l":...}
        "horizon_N": N,
        "neighbor_prediction": {"i_l_bar": i_l_bar},
        "fsclf": {"V0": float(V0), "alpha": 0.9},
    }
    msg_c1 = make_envelope(
        msg_type="plant_to_controller",
        sim_id="acdcac_demo_001",
        sender="coordinator",
        receiver="sub1",
        step=step,
        outer_step=outer_step,
        Ts=Ts,
        M=M,
        payload=payload_c1,
    )
    sock_c1.send_json(msg_c1)

    # 3) Get reply with optimal sequence
    reply1 = sock_c1.recv_json()
    print("[Coord] C1 reply:", reply1)

    # Optionally send an ACK
    sock_c1.send_json({"ack": True})
    return reply1
