import socketio
import sys
import cv2

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')

@sio.event
def from_server(data):
    print('message received with ', data)

@sio.event
def disconnect():
    print('disconnected from server')

# sio.connect('http://localhost:3000/')
sio.connect('http://35.194.177.92:3000/')
if cv2.waitKey(1) & 0xFF == ord('q'): sys.exit()