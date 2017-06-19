import os
import socket
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect("/tmp/processing.socket")
client.send('wtf'.encode('UTF-8'))
