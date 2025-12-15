# controller2_node.py
import zmq, math
import numpy as np
from comm_schema import make_envelope
from current_reference.current_ref_gen import CurrentReference
from mpc_contr.mpc_contr_calc import MPCSSolver
from cost_fun.cost_func_calc import CostFunction
from pwm.pwm_gen import PWM
from load.load_dyn_cal import Load
from power_current_conv.power_current_handler import RequiredPowerCurrentHandler

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    # *** SET YOUR COORDINATOR IP HERE ***
    # Use "127.0.0.1" if coordinator runs on the same PC.
    # If this runs on PC2 and coordinator on PC1, use PC1's LAN IP, e.g. "192.168.0.10".
    coordinator_ip = "127.0.0.1"
    addr = f"tcp://{coordinator_ip}:5554"
    print(f"[C2] Connecting to coordinator at {addr}")
    sock.connect(addr)

    print("[C2] Socket type:", sock.getsockopt(zmq.TYPE))  # should be 3 (REQ)

    ## Simulation parameters
    # Carrier frequency
    carrier_freq = 1e4 # in kHz
    # Overall sampling time
    sampling_time = 1/carrier_freq
    # Load frequency in Hz
    f_load = 50.0
    # DC Source Voltage
    V_rms_req = 230.0  # RMS voltage in V
    V_dc = V_rms_req * math.sqrt(2)  # V
    # Power requirements
    P_req = 3e3  # Active power in W
    Q_req = 0.0    # Reactive power in VAR
    powerCurrentHandler = RequiredPowerCurrentHandler(P_req, Q_req, V_rms_req)
    i_ref_peak, _ = powerCurrentHandler.calculateCurrentMagnitudeAndPhase()

    # Load parameters
    Ll=5e-3
    Rl=5.0
    back_emf_peak = 0
    load = Load(sampling_time, Rl, Ll, back_emf_peak, f_load, per_unit=False)
    
    # PWM and Current Reference objects
    referenceTrajectory = CurrentReference(i_ref_peak, f_load, per_unit=False)
    pwm_load = PWM(carrier_freq, sampling_time, V_dc, tech_type = 'FB', per_unit=False)
    
    # stage cost
    def stage_func(i_l, i_l_ref):
        return (i_l - i_l_ref)**2
    
    # setup MPC solver
    # fsclf = FiniteStepLyapunov(x_eq=[0.0, 400.0, 0.0])
    cost_func = CostFunction(stage_func, subsystem={'name': 'load', 'V_dc_ref': None})
    solver = MPCSSolver(cost_func)

    u_prev = None

    print("[Controller2] Started.")
    while True:
        # 1) Send HELLO
        hello_msg = {"hello_from": "sub2"}
        print("[C1] Sending HELLO:", hello_msg)
        sock.send_json(hello_msg)
        
        # 2) Receive job from coordinator
        msg = sock.recv_json()
        print("[C2] Received job:", msg)

        if msg.get("msg_type") == "shutdown":
            print("[C2] Shutting down on coordinator request.")
            break

        step = msg["step"]
        outer_step = msg["outer_step"]
        Ts = msg["Ts"]
        M = msg["M"]
        payload = msg["payload"]

        x2 = payload["state"]["i_l"]
        i_g, v_dc = (payload["state"]["i_g"], payload["state"]["v_dc"])
        x1_bar_dicts = payload["neighbor_prediction"]["x1_bar"]
        x1_bar = [(d["i_g"], d["v_dc"]) for d in x1_bar_dicts]
        i_g_bar, v_dc_bar = x1_bar
        V0 = payload["fsclf"]["V0"]
        N = payload["horizon_N"]

        print(f"[C2] step={step}, outer_step={outer_step}, Ts={Ts}, M={M}, N={N}")
        print(f"[C2] x2={x2}, i_g={i_g}, v_dc={v_dc}, V0={V0}")
        print(f"[C2] i_g_bar={i_g_bar}")
        print(f"[C2] v_dc_bar={v_dc_bar}")

        # ---------------------------------------------------------
        # 3) Solve local OCP for sub2
        # ---------------------------------------------------------

        if u_prev is None:
            u_prev = np.zeros(N)

        current_time = step * M * Ts
        # Solve MPC (averaged model inside)
        print("[C2] Starting local optimization ...")
        u_opt, J2 = solver.solveMPC(pwm_load, load, referenceTrajectory, 
                                   t_0=current_time, x_0=x2, x_N_bar=x1_bar, cont_horizon=N, u0=u_prev, subsystem='load')

        _, x2_pred, i_l_ref = cost_func.calculateCostFuncLoad(x2, x1_bar, current_time, u_prev, N, 
                                                           u_opt, pwm_load, load, referenceTrajectory)
        i_l_0 = x2
        i_l_pre = x2_pred
        i_l_ref_0 = referenceTrajectory.generateRefTrajectory(current_time)[0]
        V0_sub = stage_func(i_l_0, i_l_ref_0)
        V_M_sub = stage_func(i_l_pre[-1], i_l_ref[-1])
        u_prev = u_opt

        reply_payload = {
            "status": "ok",
            "local_cost": float(J2),
            "u_seq": u_opt.tolist(),
            "i_l_pred": [float(xx) for xx in x2_pred],
            "contractive": {
                "V0_sub": float(V0_sub),
                "V_M_sub": float(V_M_sub),
                "alpha": payload["fsclf"]["alpha"],
                "satisfied": bool(V_M_sub <= payload["fsclf"]["alpha"] * V0_sub),
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

        # 4) Send reply back
        print("[C2] Sending reply to coordinator.")
        sock.send_json(reply)

        # 5) Wait for ACK to complete REQ/REP cycle
        ack = sock.recv_json()
        print("[C2] Received ACK from coordinator:", ack)

if __name__ == "__main__":
    main()