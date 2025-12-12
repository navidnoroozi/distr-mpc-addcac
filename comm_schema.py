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
        "version": version,     # protocol version
        "sim_id": sim_id,       # simulation identifier
        "sender": sender,       # sender identifier, "sub1", "sub2", "plant"
        "receiver": receiver,   # receiver identifier, "sub1", "sub2", "plant"
        "timestamp_wall": time.time(),  # wall-clock time
        "step": int(step),              # plain sample index in plant time (i.e. step = outer_step * M at update instants)
        "outer_step": int(outer_step),  # index of the multi-step update k (where t=kMTs ​ )
        "Ts": float(Ts),                # sampling time [s]
        "M": int(M),                    # multi-step length
        "payload": payload,             # message-specific content
    }
