# coordinator_ping_c1.py
import zmq

def main():
    context = zmq.Context()
    sock_c1 = context.socket(zmq.REP)
    sock_c2 = context.socket(zmq.REP)
    sock_p1 = context.socket(zmq.REP)
    sock_p2 = context.socket(zmq.REP)
    sock_p1.bind("tcp://0.0.0.0:5551")
    sock_p2.bind("tcp://0.0.0.0:5552")
    sock_c1.bind("tcp://0.0.0.0:5553")
    sock_c2.bind("tcp://0.0.0.0:5554")
    print("[CoordPing] Listening on 0.0.0.0:555x")
    while True:
        # Testing Coordinator <> Plant1
        hello = sock_p1.recv_json()
        print("[CoordPing] RECV hello:", hello)
        if hello.get("cmd") == "quit":
            sock_p1.send_json({"reply": "bye P1"})
            print("[CoordPing] Shutting down.")
            break
        sock_p1.send_json({"reply": "job_for_you P1"})
        result = sock_p1.recv_json()
        print("[CoordPing] RECV result:", result)
        sock_p1.send_json({"reply": "ack"})

        # Testing Coordinator <> Plant2
        hello = sock_p2.recv_json()
        print("[CoordPing] RECV hello:", hello)
        if hello.get("cmd") == "quit":
            sock_p2.send_json({"reply": "bye P2"})
            print("[CoordPing] Shutting down.")
            break
        sock_p2.send_json({"reply": "job_for_you P2"})
        result = sock_p2.recv_json()
        print("[CoordPing] RECV result:", result)
        sock_p2.send_json({"reply": "ack"})
        
        # Testing Coordinator <> Controller1
        hello = sock_c1.recv_json()
        print("[CoordPing] RECV hello:", hello)
        if hello.get("cmd") == "quit":
            sock_c1.send_json({"reply": "bye C1"})
            print("[CoordPing] Shutting down.")
            break
        sock_c1.send_json({"reply": "job_for_you C1"})
        result = sock_c1.recv_json()
        print("[CoordPing] RECV result:", result)
        sock_c1.send_json({"reply": "ack"})
        
        # Testing Coordinator <> Controller2
        hello = sock_c2.recv_json()
        print("[CoordPing] RECV hello:", hello)
        if hello.get("cmd") == "quit":
            sock_c2.send_json({"reply": "bye C2"})
            print("[CoordPing] Shutting down.")
            break
        sock_c2.send_json({"reply": "job_for_you C2"})
        result = sock_c2.recv_json()
        print("[CoordPing] RECV result:", result)
        sock_c2.send_json({"reply": "ack"})

if __name__ == "__main__":
    main()