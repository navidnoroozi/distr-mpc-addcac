# controller1_node.py
import zmq, math
import numpy as np
from comm_schema import make_envelope
from current_reference.current_ref_gen import CurrentReference
from mpc_contr.mpc_contr_calc import MPCSSolver
from cost_fun.cost_func_calc import CostFunction
from pwm.pwm_gen import PWM
from gridDClink.grid_dc_link_dyn_cal import GridDCLink
from power_current_conv.power_current_handler import RequiredPowerCurrentHandler

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
    # Power requirements
    P_req = 3e3  # Active power in W
    Q_req = 0.0    # Reactive power in VAR
    powerCurrentHandler = RequiredPowerCurrentHandler(P_req, Q_req, V_rms_req)
    i_ref_peak, _ = powerCurrentHandler.calculateCurrentMagnitudeAndPhase()

    # Grid + DC Link parameters
    Lg=10e-3
    Rg=10.0
    Cdc=5e-3
    griddclink = GridDCLink(sampling_time, Rg, Lg, Cdc, V_dc, f_grid, per_unit=False)
    
    # PWM and Current Reference objects
    referenceTrajectory = CurrentReference(i_ref_peak, f_grid, per_unit=False)
    pwm_grid = PWM(carrier_freq, sampling_time, V_dc, tech_type = 'FB', per_unit=False)
    
    # stage cost
    def stage_func(i_g, i_g_ref, v_dc, V_dc_ref):
        return (i_g - i_g_ref)**2 + (v_dc - V_dc_ref)**2
    
    # setup MPC solver
    # fsclf = FiniteStepLyapunov(x_eq=[0.0, 400.0, 0.0])
    cost_func = CostFunction(stage_func, subsystem={'name': 'grid', 'V_dc_ref': V_dc})
    solver = MPCSSolver(cost_func)

    u_prev = None

    print("[Controller1] Started.")
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
        # 3) Solve local OCP for sub1
        # ---------------------------------------------------------
        if u_prev is None:
            u_prev = np.zeros(N)

        current_time = step * M * Ts
        # Solve MPC (averaged model inside)
        print("[C1] Starting local optimization ...")
        u_opt, J1 = solver.solveMPC(pwm_grid, griddclink, referenceTrajectory, 
                                   t_0=current_time, x_0=x1, x_N_bar=i_l_bar, cont_horizon=N, u0=u_prev, subsystem='grid')


        _, x1_pred, i_g_ref = cost_func.calculateCostFuncGrid(x1, i_l_bar, current_time, u_prev, N, 
                                                           u_opt, pwm_grid, griddclink, referenceTrajectory)
        i_g_0, v_dc_0 = x1
        i_g_pre, v_dc_pre = x1_pred
        i_g_ref_0 = referenceTrajectory.generateRefTrajectory(current_time)[0]
        V0_sub = stage_func(i_g_0, i_g_ref_0, v_dc_0, V_dc)
        V_M_sub = stage_func(i_g_pre[-1], i_g_ref[-1], v_dc_pre[-1], V_dc)
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
                "satisfied": bool(V_M_sub <= payload["fsclf"]["alpha"] * V0_sub),
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