# coordinator.py
import zmq, math
import numpy as np
from comm_schema import make_envelope
from acdcac.fsclf import FiniteStepLyapunov
import matplotlib.pyplot as plt


def main():
    context = zmq.Context()

    # REP sockets for the four nodes (Plant, Controller1, Controller2)
    sock_p = context.socket(zmq.REP)
    sock_p.bind("tcp://*:5551")

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

    # Grid / DC Source Voltage
    V_rms_req = 230.0  # RMS voltage in V
    V_dc = V_rms_req * math.sqrt(2)  # V
    # global state x = (i_g, v_dc, i_l)
    # initial conditions
    i_g, v_dc, i_l = 0.0, V_dc, 0.0
    step = 0
    outer_step = 0
    t_sim = 0.0

    fsclf = FiniteStepLyapunov(x_eq=[0.0, 230.0*math.sqrt(2), 0.0])

    # initial neighbor trajectories
    i_l_bar = [i_l] * N
    x1_bar = [{"i_g": i_g, "v_dc": v_dc}] * N

    # ----------------------------------------------------------------------
    # LOGGING BUFFERS
    # ----------------------------------------------------------------------
    t_log = []
    i_g_log = []
    v_dc_log = []
    i_l_log = []
    u1_log = []
    u2_log = []

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
        print("[Coordinator] C1 hello:", hello_c1)

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
        print("[Coordinator] C2 hello:", hello_c2)

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
        # For each inner step m, we handshake with Plant:
        # REP side: recv(hello) -> send(job) -> recv(reply) -> send(ack)
        # ------------------------------------------------------------------
        for m in range(M):
            u1 = float(u1_seq[m])
            u2 = float(u2_seq[m])

            # ---- Plant step ----
            hello_p = sock_p.recv_json()
            print("[Coordinator] P hello:", hello_p)

            payload_p = {
                "u1": u1,
                "u2": u2,
                # "x2": {"i_l": i_l},
            }
            msg_p = make_envelope(
                msg_type="coord_to_plant",
                sim_id=sim_id,
                sender="coordinator",
                receiver="plant",
                step=step,
                outer_step=outer_step,
                Ts=Ts,
                M=M,
                payload=payload_p,
            )
            # send job to Plant
            sock_p.send_json(msg_p)

            # receive updated x1 and x2 from Plant
            rep_p = sock_p.recv_json()
            i_g = rep_p["payload"]["x1_next"]["i_g"]
            v_dc = rep_p["payload"]["x1_next"]["v_dc"]
            i_l = rep_p["payload"]["x2_next"]["i_l"]

            # send ACK to complete REP cycle
            ack_p = {
                "msg_type": "ack",
                "sim_id": sim_id,
                "sender": "coordinator",
                "receiver": "plant",
                "step": step,
                "outer_step": outer_step,
            }
            sock_p.send_json(ack_p)

            ## here you’d log (step, t_sim, i_g, v_dc, i_l, u1, u2, etc.) ##
            print(
                f"[Coordinator] step={step} t={t_sim:.6f} "
                f"i_g={i_g:.3f} v_dc={v_dc:.2f} i_l={i_l:.3f}"
            )
            # Logging
            t_log.append(t_sim)
            i_g_log.append(i_g)
            v_dc_log.append(v_dc)
            i_l_log.append(i_l)
            u1_log.append(u1)
            u2_log.append(u2)

            # increment time
            step += 1
            t_sim = step * Ts

    print("[Coordinator] Simulation finished.")

    # ------------------------------------------------------------------
    # CLEAN SHUTDOWN HANDSHAKE
    # Each node will, after the last ACK, go back to the top of its loop
    # and send another "hello_from". Here we receive that hello and reply
    # with a 'shutdown' message. Nodes then break and terminate.
    # ------------------------------------------------------------------

    shutdown_payload = {}  # not used by nodes; msg_type is enough

    def send_shutdown(sock, receiver_name):
        hello = sock.recv_json()  # wait for final hello_from
        # print(f"[Coordinator] Final hello from {receiver_name}:", hello)
        msg_shutdown = make_envelope(
            msg_type="shutdown",
            sim_id=sim_id,
            sender="coordinator",
            receiver=receiver_name,
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload=shutdown_payload,
        )
        sock.send_json(msg_shutdown)

    # order not critical, just do all four
    send_shutdown(sock_c1, "sub1")
    send_shutdown(sock_c2, "sub2")
    send_shutdown(sock_p, "plant")

    # close sockets and context
    sock_p.close()
    sock_c1.close()
    sock_c2.close()
    context.term()

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    if t_log:
        # Voltage plot
        plt.figure()
        plt.title("DC-link voltage v_dc")
        plt.plot(t_log, v_dc_log, label="v_dc")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.grid(True)
        plt.legend()

        # Currents plot
        plt.figure()
        plt.title("Currents i_g and i_l")
        plt.plot(t_log, i_g_log, label="i_g")
        plt.plot(t_log, i_l_log, label="i_l")
        plt.xlabel("Time [s]")
        plt.ylabel("Current [A]")
        plt.grid(True)
        plt.legend()

        plt.show()

    # ------------------------------------------------------------------
    # RETURN "SIMULATION FINISHED" RESULT
    # ------------------------------------------------------------------
    sim_result = {
        "status": "finished",
        "n_steps": len(t_log),
        "t_final": t_log[-1] if t_log else 0.0,
        "i_g_final": i_g,
        "v_dc_final": v_dc,
        "i_l_final": i_l,
    }

    print("Coordinator: simulation result:", sim_result)
    return sim_result


if __name__ == "__main__":
    main()
