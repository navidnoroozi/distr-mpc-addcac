# zmq_debug.py
import zmq
import json
import traceback
from typing import Any, Dict

class DebugSocket:
    def __init__(self, sock: zmq.Socket, name: str):
        self.sock = sock
        self.name = name
        self.send_count = 0
        self.recv_count = 0

    def send_json(self, msg: Dict[str, Any]):
        self.send_count += 1
        short = self._short_summary(msg)
        print(f"[{self.name}] SEND #{self.send_count}: {short}")
        try:
            return self.sock.send_json(msg)
        except Exception as e:
            print(f"[{self.name}] ERROR on send_json: {e}")
            traceback.print_exc()
            raise

    def recv_json(self):
        self.recv_count += 1
        try:
            msg = self.sock.recv_json()
        except Exception as e:
            print(f"[{self.name}] ERROR on recv_json: {e}")
            traceback.print_exc()
            raise

        short = self._short_summary(msg)
        print(f"[{self.name}] RECV #{self.recv_count}: {short}")
        return msg

    def _short_summary(self, msg: Dict[str, Any]) -> str:
        if not isinstance(msg, dict):
            return f"(non-dict) {str(msg)[:80]}"
        msg_type = msg.get("msg_type", "?")
        sender = msg.get("sender", "?")
        receiver = msg.get("receiver", "?")
        step = msg.get("step", "?")
        outer = msg.get("outer_step", "?")
        return f"type={msg_type} from={sender} to={receiver} step={step} outer={outer}"

    # convenient pass-through for other zmq methods
    def __getattr__(self, item):
        return getattr(self.sock, item)
