from flask import Flask, request, jsonify
import numpy as np

from pwm.pwm_gen import PWM
from current_reference.current_ref_gen import CurrentReference
from acdcac.plant_acdcac import SinglePhaseACDCACPlant
from acdcac.fsclf import FiniteStepLyapunov
from acdcac.scenario_centralized import (
    sim_executor_acdcac_centralized,
    stage_func_centralized,
)
from acdcac.distributed_mpc_acdcac import (
    sim_executor_acdcac_distributed,
)

app = Flask(__name__)

def build_default_system():
    Ts = 100e-6
    plant = SinglePhaseACDCACPlant(
        sampling_rate=Ts,
        Lg=10e-3,
        Rg=10.0,
        Cdc=5e-3,
        Ll=5e-3,
        Rl=5.0,
        V_grid_rms=230.0,
        f_grid=50.0,
    )
    pwm_grid = PWM(carrier_freq=10e3, Ts=Ts, Vdc=400.0)
    pwm_load = PWM(carrier_freq=10e3, Ts=Ts, Vdc=400.0)
    currentReference = CurrentReference(
        i_ref_peak=10.0,  # physical A (adjust as needed)
        f_ref=50.0,
        per_unit=False,
    )
    fsclf = FiniteStepLyapunov(x_eq=[0.0, 400.0, 0.0])
    return Ts, plant, pwm_grid, pwm_load, currentReference, fsclf

@app.route("/api/centralized", methods=["POST"])
def api_centralized():
    data = request.get_json() or {}
    horizon = int(data.get("horizon", 10))
    sim_time = float(data.get("sim_time", 0.1))

    Ts, plant, pwm_grid, pwm_load, currentReference, _ = build_default_system()
    res = sim_executor_acdcac_centralized(
        stage_func=stage_func_centralized,
        pwm_grid=pwm_grid,
        pwm_load=pwm_load,
        plant=plant,
        currentReference=currentReference,
        u0_g=[0.0]*horizon,
        u0_l=[0.0]*horizon,
        cont_horizon=horizon,
        t_0=0.0,
        state0=(0.0, 400.0, 0.0),
        sampling_rate=Ts,
        sim_time=sim_time,
    )

    # convert np arrays to lists for JSON
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in res.items()}
    return jsonify(out)

@app.route("/api/distributed", methods=["POST"])
def api_distributed():
    data = request.get_json() or {}
    horizon = int(data.get("horizon", 10))
    M = int(data.get("M", 3))
    sim_time = float(data.get("sim_time", 0.1))

    Ts, plant, pwm_grid, pwm_load, currentReference, fsclf = build_default_system()
    res = sim_executor_acdcac_distributed(
        plant=plant,
        pwm_grid=pwm_grid,
        pwm_load=pwm_load,
        currentReference=currentReference,
        fsclf=fsclf,
        horizon_N=horizon,
        M=M,
        t_0=0.0,
        state0=(0.0, 400.0, 0.0),
        sampling_rate=Ts,
        sim_time=sim_time,
    )
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in res.items()}
    return jsonify(out)

@app.route("/")
def index():
    return (
        "<h1>AC/DC/AC MPC Simulator</h1>"
        "<p>POST JSON to /api/centralized or /api/distributed with keys "
        "`horizon`, `sim_time` (and `M` for distributed).</p>"
    )

if __name__ == "__main__":
    app.run(debug=True)
