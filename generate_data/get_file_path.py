#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import tqdm
import os
import argparse


def get_event_first(x):
    path = x.split("/")
    name1,name2 = path[-1].split(".")[0].split("_")[:2]
    name = f"{str(int(name1)-1).zfill(6)}_{str(int(name2)-1).zfill(6)}_event.hdf5"
    path = path[:-1] + [str(name)]
    return "/".join(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    option = parser.parse_args()
    FILESIZE = 100 * 1024
    with open("file_path.txt", "w") as f:
        c = 0
        for event_second in tqdm.tqdm(sorted(glob.glob(f"{option.dataset_root}/**/event_left/*.hdf5",recursive=True))):
            if event_second.endswith("000000_000001_event.hdf5"):
                continue        
            event_first= get_event_first(event_second)
            if not os.path.exists(event_first):
                print("No file")
                print(event_first)
                continue
            
            if os.path.getsize(event_first) < FILESIZE or os.path.getsize(event_second) < FILESIZE:
                continue
            
            path = " ".join([event_first, event_second])
            f.write(path + "\n")
        c += 1
        print("Down")
        print(c) 
    