import os
os.environ["KMP_BLOCKTIME"] = "0"
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.utils.data.dataset as dataset
from .event_utils import eventsToVoxel
from .file_io import read_event_h5

def isin(i,xs):
    for x in xs:
        if x in i:
            return True
    return False

TRAIN_SCENCE = ['westerndesert', 'seasidetown', 'amusement', 'carwelding', 'seasonsforest', 'office2', 'japanesealley', 'ocean', 'abandonedfactory_night', 'endofworld', 'office', 'soulcity', 'oldtown', 'seasonsforest_winter', 'abandonedfactory']
TEST_SCENCE = ['hospital','gascola', 'neighborhood']

class TartanairPretrainDataset(dataset.Dataset):
    def __init__(self, args, train = False, aug_params=None):
        super().__init__()
        self.args = args
        self.event_bins = args.event_bins
        self.event_polarity = False if args.no_event_polarity else True
        self.train = train
        
        self.aug_params = aug_params

        self.fetch_valids()
        self.data_length = len(self.data)

    def fetch_valids(self):
        data = [i.strip().split(" ") for i in open(self.args.file, 'r').readlines()]
        scence = TRAIN_SCENCE if self.train else TEST_SCENCE 
        data = [i for i in data if isin(i[1],scence)]
        self.data = data

    def load_data_by_index(self, index):
        event_filename = self.data[index][1]
        if event_filename.endswith(".npy"):
            return np.load(event_filename)
        else:
            return read_event_h5(event_filename)
    
    def crop(self, event):

        crop_size = self.aug_params['crop_size']
        
        height, width = crop_size
        
        if self.train:
            y0 = np.random.randint(0, 480 - crop_size[0]) 
            x0 = np.random.randint(0, 640 - crop_size[1])
        else:
            y0 = (480 - crop_size[0])//2
            x0 = (640 - crop_size[1])//2

        if len(event.shape)==2:
            valid_events = (event[:, 0] >= x0) & (event[:, 0] <= x0 + crop_size[1] - 1) &\
                         (event[:, 1] >= y0) & (event[:, 1] <= y0 + crop_size[0] - 1)
    
            event = event[valid_events]
            event[:,0] = event[:,0] - x0
            event[:,1] = event[:,1] - y0
            
            if event.shape[0] < 10:
                c = 1 + int(self.event_polarity)
                event  = np.zeros((self.event_bins*c,height,width))
            else:
                event = eventsToVoxel(event, num_bins=self.event_bins, height=height,
                                                width=width, event_polarity=self.event_polarity, temporal_bilinear=True)
        else:
            event = event[...,y0:y0+crop_size[0], x0:x0+crop_size[1]]
               
        event = torch.from_numpy(event)
        return event

    def __getitem__(self, index):
        index = index % self.data_length
        events1_nparray = \
            self.load_data_by_index(index)
        event = self.crop(events1_nparray)
        return event    
    
    def __len__(self):
        return self.data_length
