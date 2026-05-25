import cv2

from src.ops.device.hikvision_camera.camera import CameraWrapper

cam = CameraWrapper()
cam.enable_device_by_ip('192.168.6.109')
cam.set_exposure_time(90000)
img = cam.get_image()
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
