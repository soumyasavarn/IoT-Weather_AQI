import socket
s = socket.socket()
s.connect(("127.0.0.1", 1883))
print("Connected!")
s.close()
