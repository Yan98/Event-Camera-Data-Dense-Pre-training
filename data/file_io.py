import h5py
import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def read_event_h5(path):
    file = h5py.File(path, 'r')
    length = len(file['x'])
    events = np.zeros([length, 4], dtype=np.float32)
    events[:, 0] = file['x']
    events[:, 1] = file['y']
    events[:, 2] = file['t']
    events[:, 3] = file['p']
    file.close()
    return events
