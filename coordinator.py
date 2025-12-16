# coordinator.py
import zmq, math
import numpy as np
import matplotlib.pyplot as plt
from comm_schema import make_envelope
from acdcac.fsclf import FiniteStepLyapunov

def _current_ref(t: float, I_peak: float, f: float, phase_rad: float=0.0) -> float:
    return I_peak * math.cos(2.0 * math.pi * f * t - phase_rad)

def main():
    context = zmq.Context()

    # REP sockets for nodes
    sock_p = context.socket(zmq.REP)
    sock_p.bind("tcp://*:5551")

    sock_c1 = context.socket(zmq.REP)
    sock_c1.bind("tcp://*:5553")

    sock_c2 = context.socket(zmq.REP)
    sock_c2.bind("tcp://*:5554")

    # ------------------------------------------------------------
    # Simulation settings (keep consistent with controller scripts)
    # ------------------------------------------------------------
    sim_id = "acdcac_demo_001"
    Ts = 1e-4
    M = 3
    N = M  # as requested: control horizon equals finite-step M
    sim_time = 0.1  # total simulation time [s]
    total_steps = int(sim_time / Ts)
    outer_steps = total_steps // M

    # Electrical scenario (for plotting refs)
    f_grid = 50.0
    f_load = 50.0
    V_rms_req = 230.0
    V_dc_ref = V_rms_req * math.sqrt(2)

    # Power requirement -> current magnitude and phase (P,Q at unity PF -> phase=0)
    P_req = 3e3
    Q_req = 0.0
    S = math.sqrt(P_req**2 + Q_req**2)
    I_rms = S / V_rms_req
    I_peak = math.sqrt(2.0) * I_rms
    phase = math.atan2(Q_req, P_req) if P_req != 0 else 0.0

    # Global state x = (i_g, v_dc, i_l)
    i_g, v_dc, i_l = 0.0, V_dc_ref, 0.0
    step = 0
    t_sim = 0.0

    fsclf = FiniteStepLyapunov(x_eq=[0.0, V_dc_ref, 0.0])

    # Initial neighbor trajectories (length N)
    i_l_bar = [i_l] * N
    x1_bar = [{"i_g": i_g, "v_dc": v_dc}] * N

    # ------------------------------------------------------------
    # LOGGING BUFFERS (per inner step Ts)
    # ------------------------------------------------------------
    t_log, i_g_log, v_dc_log, i_l_log = [], [], [], []
    i_g_ref_log, i_l_ref_log, v_dc_ref_log = [], [], []
    e_ig_log, e_il_log, e_vdc_log = [], [], []
    u1_log, u2_log = [], []

    s1_log, s2_log = [], []
    # gate signals per leg (optional, sampled at end of Ts)
    g1_Sa_p, g1_Sa_n, g1_Sb_p, g1_Sb_n = [], [], [], []
    g2_Sa_p, g2_Sa_n, g2_Sb_p, g2_Sb_n = [], [], [], []

    # Costs (per OUTER step, i.e. per optimization)
    t_outer_log, J1_log, J2_log = [], [], []

    print("[Coordinator] Started.")
    for outer_step in range(outer_steps):
        x = np.array([i_g, v_dc, i_l], dtype=float)
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
            "neighbor_prediction": {"i_l_bar": i_l_bar},
            "fsclf": {"V0": float(V0), "alpha": 0.9},
            "t_sim": t_sim,
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
        J1 = float(reply1["payload"].get("local_cost", float("nan")))

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
        # you can inspect hello_c2 if you like:
        print("[Coordinator] C2 hello:", hello_c2)

        payload_c2 = {
            "state": {"i_g": i_g, "v_dc": v_dc, "i_l": i_l},
            "horizon_N": N,
            "neighbor_prediction": {"x1_bar": x1_bar},
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
        J2 = float(reply2["payload"].get("local_cost", float("nan")))
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
        
        #---------------- Log costs at the start of this outer interval ----------------
        t_outer_log.append(t_sim)
        J1_log.append(J1)
        J2_log.append(J2)

        # ---------------- Apply full M-step sequence to plant ----------------
        # For each inner step m, we handshake with Plant:
        # REP side: recv(hello) -> send(job) -> recv(reply) -> send(ack)
        # ------------------------------------------------------------------
        for m in range(M):
            u1 = float(u1_seq[m])
            u2 = float(u2_seq[m])

            # handshake with Plant
            hello_p = sock_p.recv_json()
            # you can inspect hello_p if you like:
            print("[Coordinator] P hello:", hello_p)
            
            payload_p = {"u1": u1, "u2": u2}
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

            i_g = float(rep_p["payload"]["x1_next"]["i_g"])
            v_dc = float(rep_p["payload"]["x1_next"]["v_dc"])
            i_l = float(rep_p["payload"]["x2_next"]["i_l"])

            sw = rep_p["payload"].get("switching", {})
            s1 = int(sw.get("s1", 0))
            s2 = int(sw.get("s2", 0))
            g1 = sw.get("gates1", {}) or {}
            g2 = sw.get("gates2", {}) or {}

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

            # references + errors at *this* time stamp
            i_g_ref = _current_ref(t_sim, I_peak, f_grid, phase)
            i_l_ref = _current_ref(t_sim, I_peak, f_load, phase)
            vdc_ref = V_dc_ref

            t_log.append(t_sim)
            i_g_log.append(i_g)
            v_dc_log.append(v_dc)
            i_l_log.append(i_l)

            i_g_ref_log.append(i_g_ref)
            i_l_ref_log.append(i_l_ref)
            v_dc_ref_log.append(vdc_ref)

            e_ig_log.append(i_g_ref - i_g)
            e_il_log.append(i_l_ref - i_l)
            e_vdc_log.append(vdc_ref - v_dc)

            u1_log.append(u1)
            u2_log.append(u2)

            s1_log.append(s1)
            s2_log.append(s2)

            # gate traces (full-bridge) if present
            g1_Sa_p.append(int(g1.get("Sa_p", 0)))
            g1_Sa_n.append(int(g1.get("Sa_n", 0)))
            g1_Sb_p.append(int(g1.get("Sb_p", 0)))
            g1_Sb_n.append(int(g1.get("Sb_n", 0)))

            g2_Sa_p.append(int(g2.get("Sa_p", 0)))
            g2_Sa_n.append(int(g2.get("Sa_n", 0)))
            g2_Sb_p.append(int(g2.get("Sb_p", 0)))
            g2_Sb_n.append(int(g2.get("Sb_n", 0)))

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
    def send_shutdown(sock, receiver_name):
        hello = sock.recv_json()  # node will send another hello
        print(f"[Coordinator] Received shutdown hello from {receiver_name}: ", hello)

        msg_shutdown = make_envelope(
            msg_type="shutdown",
            sim_id=sim_id,
            sender="coordinator",
            receiver=receiver_name,
            step=step,
            outer_step=outer_step,
            Ts=Ts,
            M=M,
            payload={},
        )
        sock.send_json(msg_shutdown)

    # order not critical, just do all three
    send_shutdown(sock_c1, "sub1")
    send_shutdown(sock_c2, "sub2")
    send_shutdown(sock_p, "plant")

    sock_p.close(); sock_c1.close(); sock_c2.close(); context.term()

    # ---------------- PLOTTING ----------------
    if t_log:
        # Currents tracking
        plt.figure()
        plt.title("Currents tracking")
        plt.plot(t_log, i_g_log, label="i_g")
        plt.plot(t_log, i_g_ref_log, "--", label="i_g_ref")
        plt.plot(t_log, i_l_log, label="i_l")
        plt.plot(t_log, i_l_ref_log, "--", label="i_l_ref")
        plt.xlabel("Time [s]"); plt.ylabel("Current [A]")
        plt.grid(True); plt.legend()

        # Current errors
        plt.figure()
        plt.title("Current tracking errors")
        plt.plot(t_log, e_ig_log, label="e_i_g = i_g_ref - i_g")
        plt.plot(t_log, e_il_log, label="e_i_l = i_l_ref - i_l")
        plt.xlabel("Time [s]"); plt.ylabel("Error [A]")
        plt.grid(True); plt.legend()

        # DC-link voltage tracking + error
        plt.figure()
        plt.title("DC-link voltage v_dc tracking")
        plt.plot(t_log, v_dc_log, label="v_dc")
        plt.plot(t_log, v_dc_ref_log, "--", label="v_dc_ref")
        plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]")
        plt.grid(True); plt.legend()

        plt.figure()
        plt.title("DC-link voltage error")
        plt.plot(t_log, e_vdc_log, label="e_v_dc = v_dc_ref - v_dc")
        plt.xlabel("Time [s]"); plt.ylabel("Error [V]")
        plt.grid(True); plt.legend()

        # Control inputs
        plt.figure()
        plt.title("Control inputs (modulation commands)")
        plt.plot(t_log, u1_log, label="u_1")
        plt.plot(t_log, u2_log, label="u_2")
        plt.xlabel("Time [s]"); plt.ylabel("u [-]")
        plt.grid(True); plt.legend()

        # Switching (line states) + gate samples (end-of-step)
        plt.figure()
        plt.title("Switching signals (sampled at end of Ts)")
        plt.step(t_log, s1_log, where="post", label="s_1 (grid FB line state)")
        plt.step(t_log, s2_log, where="post", label="s_2 (load FB line state)")
        plt.xlabel("Time [s]"); plt.ylabel("s in {+1,-1}")
        plt.grid(True); plt.legend()

        plt.figure()
        plt.title("Gate signals (sampled at end of Ts) - Converter 1 (grid)")
        plt.step(t_log, g1_Sa_p, where="post", label="Sa_p")
        plt.step(t_log, g1_Sa_n, where="post", label="Sa_n")
        plt.step(t_log, g1_Sb_p, where="post", label="Sb_p")
        plt.step(t_log, g1_Sb_n, where="post", label="Sb_n")
        plt.xlabel("Time [s]"); plt.ylabel("Gate")
        plt.ylim(-0.2, 1.2)
        plt.grid(True); plt.legend()

        plt.figure()
        plt.title("Gate signals (sampled at end of Ts) - Converter 2 (load)")
        plt.step(t_log, g2_Sa_p, where="post", label="Sa_p")
        plt.step(t_log, g2_Sa_n, where="post", label="Sa_n")
        plt.step(t_log, g2_Sb_p, where="post", label="Sb_p")
        plt.step(t_log, g2_Sb_n, where="post", label="Sb_n")
        plt.xlabel("Time [s]"); plt.ylabel("Gate")
        plt.ylim(-0.2, 1.2)
        plt.grid(True); plt.legend()

        # Costs (per optimization / outer step)
        if t_outer_log:
            plt.figure()
            plt.title("Local MPC costs per outer step")
            plt.plot(t_outer_log, J1_log, label="J1 (subsystem 1)")
            plt.plot(t_outer_log, J2_log, label="J2 (subsystem 2)")
            plt.xlabel("Time [s]"); plt.ylabel("Cost")
            plt.grid(True); plt.legend()

        plt.show()

    # Return value for logging/tests
    sim_result = {
        "status": "finished",
        "n_steps": len(t_log),
        "t_final": t_log[-1] if t_log else 0.0,
        "i_g_final": i_g,
        "v_dc_final": v_dc,
        "i_l_final": i_l,
        "J1_last": J1_log[-1] if J1_log else None,
        "J2_last": J2_log[-1] if J2_log else None,
    }
    print("Coordinator: simulation result:", sim_result)
    return sim_result

if __name__ == "__main__":
    main()
