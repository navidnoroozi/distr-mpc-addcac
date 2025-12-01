# coordinator.py
import zmq
import numpy as np
from comm_schema import make_envelope
from acdcac.fsclf import FiniteStepLyapunov


def main():
    context = zmq.Context()

    # REP sockets for the four nodes (Plant1, Plant2, Controller1, Controller2)
    sock_p1 = context.socket(zmq.REP)
    sock_p1.bind("tcp://*:5551")

    sock_p2 = context.socket(zmq.REP)
    sock_p2.bind("tcp://*:5552")

    sock_c1 = context.socket(zmq.REP)
    sock_c1.bind("tcp://*:5553")

    sock_c2 = context.socket(zmq.REP)
    sock_c2.bind("tcp://*:5554")

    sim_id = "acdcac_demo_001"
    Ts = 1e-4
    M = 3
    N = 10
    sim_time = 0.01
    total_steps = int(sim_time / Ts)
    outer_steps = total_steps // M

    # global state x = (i_g, v_dc, i_l)
    i_g, v_dc, i_l = 0.0, 400.0, 0.0
    step = 0
    outer_step = 0
    t_sim = 0.0

    fsclf = FiniteStepLyapunov(x_eq=[0.0, 400.0, 0.0])

    # initial neighbor trajectories
    i_l_bar = [i_l] * N
    x1_bar = [{"i_g": i_g, "v_dc": v_dc}] * N

    print("[Coordinator] Started.")
    for outer_step in range(outer_steps):
        x = np.array([i_g, v_dc, i_l])
        V0 = fsclf.V(x)

        # ------------------------------------------------------------------
        # --- CONTROLLER 1 HANDSHAKE (REQ on other side) -------------------
        # Pattern on REP side: recv(hello) -> send(job) -> recv(reply) -> send(ack)
        # ------------------------------------------------------------------
        hello_c1 = sock_c1.recv_json()
        # you can inspect hello_c1 if you like:
        # print("[Coordinator] C1 hello:", hello_c1)

        payload_c1 = {
            "state": {"i_g": i_g, "v_dc": v_dc, "i_l": i_l},
            "horizon_N": N,
            "neighbor_prediction": {
                "i_l_bar": i_l_bar,
            },
            "fsclf": {"V0": float(V0), "alpha": 0.9},
        }
        msg_c1 = make_envelope(
            msg_type="plant_to_controller",
            sim_id=sim_id,
            sender="coordinator",
            receiver="sub1",
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=payload_c1,
        )
        # send job to Controller 1
        sock_c1.send_json(msg_c1)

        # receive optimization result from Controller 1
        reply1 = sock_c1.recv_json()
        u1_seq = reply1["payload"]["u_seq"]
        x1_pred = reply1["payload"]["x1_pred"]

        # complete the REP cycle with an ACK
        ack_c1 = {
            "msg_type": "ack",
            "sim_id": sim_id,
            "sender": "coordinator",
            "receiver": "sub1",
            "step": step,
            "outer_step": outer_step,
        }
        sock_c1.send_json(ack_c1)

        # build x1_bar from predicted (for Sub2)
        x1_bar = x1_pred

        # ------------------------------------------------------------------
        # --- CONTROLLER 2 HANDSHAKE (REQ on other side) -------------------
        # Pattern: recv(hello) -> send(job) -> recv(reply) -> send(ack)
        # ------------------------------------------------------------------
        hello_c2 = sock_c2.recv_json()
        # print("[Coordinator] C2 hello:", hello_c2)

        payload_c2 = {
            "state": {"i_g": i_g, "v_dc": v_dc, "i_l": i_l},
            "horizon_N": N,
            "neighbor_prediction": {
                "x1_bar": x1_bar,
            },
            "fsclf": {"V0": float(V0), "alpha": 0.9},
            "t_sim": t_sim,
        }
        msg_c2 = make_envelope(
            msg_type="plant_to_controller",
            sim_id=sim_id,
            sender="coordinator",
            receiver="sub2",
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=payload_c2,
        )
        # send job to Controller 2
        sock_c2.send_json(msg_c2)

        # receive optimization result from Controller 2
        reply2 = sock_c2.recv_json()
        u2_seq = reply2["payload"]["u_seq"]
        i_l_pred = reply2["payload"]["i_l_pred"]
        i_l_bar = i_l_pred

        # complete REP cycle with ACK
        ack_c2 = {
            "msg_type": "ack",
            "sim_id": sim_id,
            "sender": "coordinator",
            "receiver": "sub2",
            "step": step,
            "outer_step": outer_step,
        }
        sock_c2.send_json(ack_c2)

        # ------------------------------------------------------------------
        # --- APPLY FIRST M STEPS TO PLANTS --------------------------------
        # For each inner step m, we handshake with Plant1 and Plant2:
        # REP side: recv(hello) -> send(job) -> recv(reply) -> send(ack)
        # ------------------------------------------------------------------
        for m in range(M):
            u1 = float(u1_seq[m])
            u2 = float(u2_seq[m])

            # ---- Plant 1 step ----
            hello_p1 = sock_p1.recv_json()
            # print("[Coordinator] P1 hello:", hello_p1)

            payload_p1 = {
                "u1": u1,
                "x2": {"i_l": i_l},
            }
            msg_p1 = make_envelope(
                msg_type="coord_to_plant1",
                sim_id=sim_id,
                sender="coordinator",
                receiver="plant1",
                step=step,
                outer_step=outer_step,
                Ts=Ts,
                M=M,
                payload=payload_p1,
            )
            # send job to Plant 1
            sock_p1.send_json(msg_p1)

            # receive updated x1 from Plant 1
            rep_p1 = sock_p1.recv_json()
            i_g = rep_p1["payload"]["x1_next"]["i_g"]
            v_dc = rep_p1["payload"]["x1_next"]["v_dc"]

            # send ACK to complete REP cycle
            ack_p1 = {
                "msg_type": "ack",
                "sim_id": sim_id,
                "sender": "coordinator",
                "receiver": "plant1",
                "step": step,
                "outer_step": outer_step,
            }
            sock_p1.send_json(ack_p1)

            # ---- Plant 2 step ----
            hello_p2 = sock_p2.recv_json()
            # print("[Coordinator] P2 hello:", hello_p2)

            payload_p2 = {
                "u2": u2,
                "x1": {"i_g": i_g, "v_dc": v_dc},
            }
            msg_p2 = make_envelope(
                msg_type="coord_to_plant2",
                sim_id=sim_id,
                sender="coordinator",
                receiver="plant2",
                step=step,
                outer_step=outer_step,
                Ts=Ts,
                M=M,
                payload=payload_p2,
            )
            # send job to Plant 2
            sock_p2.send_json(msg_p2)

            # receive updated x2 from Plant 2
            rep_p2 = sock_p2.recv_json()
            i_l = rep_p2["payload"]["x2_next"]["i_l"]

            # send ACK to complete REP cycle
            ack_p2 = {
                "msg_type": "ack",
                "sim_id": sim_id,
                "sender": "coordinator",
                "receiver": "plant2",
                "step": step,
                "outer_step": outer_step,
            }
            sock_p2.send_json(ack_p2)

            # here you’d log (step, t_sim, i_g, v_dc, i_l, u1, u2, etc.)
            print(
                f"[Coordinator] step={step} t={t_sim:.6f} "
                f"i_g={i_g:.3f} v_dc={v_dc:.2f} i_l={i_l:.3f}"
            )

            step += 1
            t_sim = step * Ts

    print("[Coordinator] Simulation finished.")

    # (Optional) you can now wait for final 'hello' from each node and send a
    # 'shutdown' message if you want to terminate them cleanly.


if __name__ == "__main__":
    main()
