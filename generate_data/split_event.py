#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import h5py
import os
import numpy as np
from tqdm import tqdm
import argparse

def save_events_h5(events, event_file):
    ex = events[:, 0].astype(np.uint16)
    ey = events[:, 1].astype(np.uint16)
    et = events[:, 2].astype(np.float32)
    ep = events[:, -1].astype(np.int8)

    file = h5py.File(event_file, 'w')
    file.create_dataset('x', data=ex, dtype=np.uint16, compression="lzf")
    file.create_dataset('y', data=ey, dtype=np.uint16, compression="lzf")
    file.create_dataset('p', data=ep, dtype=np.int8, compression="lzf")
    file.create_dataset('t', data=et, dtype=np.float32, compression="lzf")
    file.close()

def read_event_h5(path):
    file = h5py.File(path, 'r')
    events=np.float32(file["events"])[:,[1,2,0,3]]
    file.close()
    return events

def isskip(file, number_of_files):
    path = os.path.sep.join(file.split("/")[:-1]) 
    path = os.path.join(path,"event_left")
    
    if len(glob.glob(os.path.join(path,"*.hdf5"))) == number_of_files:
        return True 
    

def save_events(events,file,number_of_files):
    path = os.path.sep.join(file.split("/")[:-1]) 
    path = os.path.join(path,"event_left")
    
    os.makedirs(path,exist_ok=True)
   
    value = []
    for i in range(number_of_files+1):
        value.append(i*10**6)
    index = np.searchsorted(events[:,2], value)
    
    for i in range(number_of_files):
        assert index[i] <= index[i+1]
        save_events_h5(
            events[index[i]:index[i+1]],
            os.path.join(path,f"{str(i).zfill(6)}_{str(i+1).zfill(6)}_event.hdf5")
            )

def valid_file(file):
    
    image_files = file.split("/")[:-1] + ["image_left", "*.png"]
    image_files = len(glob.glob(os.path.sep.join(image_files))) - 1
    
    if isskip(file, image_files):
        return 
    
    events = read_event_h5(file)
    
    print(f"Process {file}")
    if np.round(events[-1,2]/10**6) == (image_files-1):
        save_events(events,file, image_files) 
    else:
        print("==========================")
        print(file,events[-1,2]/10**6,(image_files-1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_template', default="./dataset/**/event_left.h5", type=str, required=True)
    option = parser.parse_args()
    for file in tqdm(glob.glob(option.file_template,recursive=True)):
        valid_file(file)    
