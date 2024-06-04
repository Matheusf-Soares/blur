import cv2
import numpy as np
import scipy
import scipy.ndimage
from scipy import signal

PATH_XML = 'haarcascade_frontalface_default.xml'

def generate_kernel(kernel_len=31, desvio_padrao=30):
    generate_kernel1d = signal.gaussian(kernel_len, std=desvio_padrao).reshape(kernel_len, 1)
    generate_kernel2d = np.outer(generate_kernel1d, generate_kernel1d)

    return generate_kernel2d

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + PATH_XML)
video_capture = cv2.VideoCapture(0)

kernel = generate_kernel()
# kernel_tile = np.tile(kernel, (3, 1, 1))
kernel_sum = kernel.sum()
kernel = kernel / kernel_sum
kernel_3d = np.atleast_3d(kernel)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for x, y, w, h in faces:

        frame[y: y + h, x: x + h] = scipy.ndimage.convolve(frame[y: y + h, x: x + h], kernel_3d, mode='nearest')
        # frame[y: y + h, x: x + h] = cv2.GaussianBlur(frame[y: y + h, x: x + h], (23, 23), sigmaX=20, sigmaY=20)


    cv2.imshow('Meu Vídeo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

video_capture.release()
cv2.destroyAllWindows()