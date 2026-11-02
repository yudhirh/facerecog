import os
import cv2
from app import ensure_opencv_haarcascade

path = ensure_opencv_haarcascade()
print(path)
print(os.path.exists(path))
print(os.listdir(os.path.dirname(path))[:10])
