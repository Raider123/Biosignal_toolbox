import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5555")

# Subscribe to all messages (empty string as the filter)
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("Subscriber is active")

while True:
    empfangen = socket.recv_string()
    print(empfangen)