# plant2_ping_client.py
import zmq
import time

def main():
    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    coord_ip = "127.0.0.1"  # or LAN IP of PC1
    addr = f"tcp://{coord_ip}:5552"
    print(f"[P2Ping] Connecting to {addr}")
    sock.connect(addr)

    for i in range(3):
        sock.send_json({"cmd": "ping", "seq": i})
        job = sock.recv_json()
        print("[P2Ping] Got job:", job)
        sock.send_json({"result": f"done_{i}"})
        ack = sock.recv_json()
        print("[P2Ping] Got ack:", ack)
        time.sleep(1.0)

    sock.send_json({"cmd": "quit"})
    reply = sock.recv_json()
    print("[P2Ping] Final reply:", reply)

if __name__ == "__main__":
    main()
