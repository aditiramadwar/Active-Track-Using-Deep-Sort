import numpy as np
from djitellopy import tello
import time


def trackObject(cx, w, pd, pError, drone, fbRange, ar):
    area = ar
    x= cx
    fb = 0
    error = x - w // 2
    speed = pd[0] * error + pd[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    drone.send_rc_control(0, fb, 0, speed)
    return error

def init_drone():
    me = tello.Tello()
    me.connect()
    me.streamon()
    battery = me.get_battery()
    if battery <=30:
        print("Battery level too low")
        exit(0)
    me.takeoff()
    me.send_rc_control(0, 0, 25, 0)
    time.sleep(6.6)
    me.send_rc_control(0, 0, 0, 0)

    return me