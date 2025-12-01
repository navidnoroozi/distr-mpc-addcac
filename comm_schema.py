# comm_schema.py
import time
from typing import Any, Dict

def make_envelope(
    msg_type: str,
    sim_id: str,
    sender: str,
    receiver: str,
    step: int,
    outer_step: int,
    Ts: float,
    M: int,
    payload: Dict[str, Any],
    version: str = "1.0",
) -> Dict[str, Any]:
    return {
        "msg_type": msg_type,   # "plant_to_controller", "controller_to_plant", ...
        "version": version,
        "sim_id": sim_id,
        "sender": sender,
        "receiver": receiver,
        "timestamp_wall": time.time(),
        "step": int(step),
        "outer_step": int(outer_step),
        "Ts": float(Ts),
        "M": int(M),
        "payload": payload,
    }
