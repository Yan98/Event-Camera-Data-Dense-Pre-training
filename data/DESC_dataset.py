import torch
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch.nn.functional as f
from torch.utils.data import Dataset
from PIL import Image
from joblib import Parallel, delayed
from . import seg_utils as data_util
import random
import albumentations as A

class DatasetProvider:
    def __init__(self, dataset_path, mode: str = 'train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20,
                 nr_events_window=-1, nr_bins_per_data=5, require_paired_data=False, normalize_event=False,
                 separate_pol=False, semseg_num_classes=11, augmentation=False,
                 fixed_duration=False, resize=False, extra_aug = False, **kwargs):
        dataset_path = Path(dataset_path)
        train_path = dataset_path / 'train'
        val_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)
        assert val_path.is_dir(), str(val_path)
        self.mode = mode
        if mode == 'train':
            train_sequences = list()
            train_sequences_namelist = ['zurich_city_00_a', 'zurich_city_01_a', 'zurich_city_02_a',
                                        'zurich_city_04_a', 'zurich_city_05_a', 'zurich_city_06_a',
                                        'zurich_city_07_a', 'zurich_city_08_a']
            for child in train_path.iterdir():
                if any(k in str(child) for k in train_sequences_namelist):
                    train_sequences.append(Sequence(child, 'train', event_representation, nr_events_data, delta_t_per_data,
                                                    nr_events_window, nr_bins_per_data, require_paired_data, normalize_event
                                                    , separate_pol, semseg_num_classes, True, fixed_duration
                                                    , extra_aug = extra_aug, resize=resize))
                else:
                    continue

            self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
            self.train_dataset.require_paired_data = require_paired_data

        elif mode == 'val':
            val_sequences = list()
            val_sequences_namelist = ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']
            for child in val_path.iterdir():
                if any(k in str(child) for k in val_sequences_namelist):
                    val_sequences.append(Sequence(child, 'val', event_representation, nr_events_data, delta_t_per_data,
                                                  nr_events_window, nr_bins_per_data, require_paired_data, normalize_event
                                                  , separate_pol, semseg_num_classes, False, fixed_duration
                                                  , resize=resize))
                else:
                    continue

            self.val_dataset = torch.utils.data.ConcatDataset(val_sequences)
            self.val_dataset.require_paired_data = require_paired_data


    def get_dataset(self):
        if self.mode == "train":
            return self.train_dataset
        else:
            return self.val_dataset

class Sequence(Dataset):
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_00_a)
    # ├── semantic
    # │   ├── left
    # │   │   ├── 11classes
    # │   │   │   │
    # │   │   │   ├── 000000.png
    # │   │   │   └── ...
    # │   │   └── 19classes
    # │   │   │   │
    # │   │   │   ├── 000000.png
    # │   │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     └── left
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path, mode: str='train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20, nr_events_per_data: int = 100000,
                 nr_bins_per_data: int = 5, require_paired_data=False, normalize_event=False, separate_pol=False,
                 semseg_num_classes: int = 11, augmentation=False, fixed_duration=False, remove_time_window: int = 250,
                 resize=False, extra_aug = False):
        
        assert nr_bins_per_data >= 1
        assert seq_path.is_dir()
        self.sequence_name = seq_path.name
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.resize = resize
        self.shape_resize = None
        if self.resize:
            self.shape_resize = [448, 640]

        # Set event representation
        self.nr_events_data = nr_events_data
        self.num_bins = nr_bins_per_data
        assert nr_events_per_data > 0
        self.nr_events_per_data = nr_events_per_data
        self.event_representation = event_representation
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.voxel_grid = data_util.VoxelGrid(self.num_bins, self.height, self.width, normalize=self.normalize_event)

        self.locations = ['left']
        self.semseg_num_classes = semseg_num_classes
        self.augmentation = augmentation

        # Save delta timestamp
        self.fixed_duration = fixed_duration
        if self.fixed_duration:
            delta_t_ms = nr_events_data * delta_t_per_data
            self.delta_t_us = delta_t_ms * 1000
        self.remove_time_window = remove_time_window

        self.require_paired_data = require_paired_data

        # load timestamps
        self.timestamps = np.loadtxt(str(seq_path / (str(seq_path.name)+'_semantic_timestamps.txt')), dtype='int64') #np.loadtxt(str(seq_path / 'semantic' / 'timestamps.txt'), dtype='int64')

        # load label paths
        if self.semseg_num_classes == 11:
            label_dir = seq_path / '11classes'  #seq_path / 'semantic' / '11classes' / 'data'
        elif self.semseg_num_classes == 19:
            label_dir = seq_path / '19classes'  #seq_path / 'semantic' / '19classes' / 'data'
        else:
            raise ValueError
        assert label_dir.is_dir()
        label_pathstrings = list()
        for entry in label_dir.iterdir():
            assert str(entry.name).endswith('.png')
            label_pathstrings.append(str(entry))
        label_pathstrings.sort()
        self.label_pathstrings = label_pathstrings

        assert len(self.label_pathstrings) == self.timestamps.size, f"{len(self.label_pathstrings) } / {self.timestamps.size}"

        # load images paths
        if self.require_paired_data:
            raise NotImplementedError()

        # Remove several label paths and corresponding timestamps in the remove_time_window.
        # This is necessary because we do not have enough events before the first label.
        self.timestamps = self.timestamps[(self.remove_time_window // 100 + 1) * 2:]
        del self.label_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
        assert len(self.label_pathstrings) == self.timestamps.size
        if self.require_paired_data:
            raise NotImplementedError()
            
        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'
            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = data_util.EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]
        

        self.aug = A.ReplayCompose([
            A.PadIfNeeded(min_height=500, min_width=700),
            A.RandomResizedCrop(height=448, width=640,interpolation=cv2.INTER_NEAREST),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            ])

    def apply_augmentation(self, events, label):
        label = label.numpy()
        label = data_util.shiftUpId(label)
        A_data = self.aug(image=events[0, :, :].numpy(), mask=label)
        label = A_data['mask']
        label = data_util.shiftDownId(label)
        events_tensor = torch.zeros((events.shape[0], 448, 640))
        for k in range(events.shape[0]):
            events_tensor[k, :, :] = torch.from_numpy(
                A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
        return events_tensor, torch.from_numpy(label).long()


    def events_to_voxel_grid(self, x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))
    @staticmethod
    def get_label(filepath):
        assert filepath.is_file()
        label = Image.open(str(filepath))
        label = np.array(label)
        return label

    def __len__(self):
        return (self.timestamps.size + 1) // 2

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def generate_event_tensor(self, job_id, events, event_tensor, nr_events_per_data):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end].copy()
        event_representation = data_util.generate_input_representation(events_temp, self.event_representation,
                                                                  (self.height, self.width), separate_pol=self.separate_pol, normalize_event=self.normalize_event)
        event_tensor[(job_id * self.num_bins):((job_id+1) * self.num_bins), :, :] = torch.from_numpy(event_representation)
    

    def __getitem__(self, index):
        
        label_path = Path(self.label_pathstrings[index * 2])
        if self.resize:
            segmentation_mask = cv2.imread(str(label_path), 0)
            segmentation_mask = cv2.resize(segmentation_mask, (self.shape_resize[1], self.shape_resize[0]),
                                           interpolation=cv2.INTER_NEAREST)
            label = np.array(segmentation_mask)
        else:
            label = self.get_label(label_path)

        ts_end = self.timestamps[index * 2]

        for location in self.locations:
            if self.fixed_duration:
                ts_start = ts_end - self.delta_t_us
                event_tensor = None
                self.delta_t_per_data_us = self.delta_t_us / self.nr_events_data
                for i in range(self.nr_events_data):
                    t_s = ts_start + i * self.delta_t_per_data_us
                    t_end = ts_start + (i+1) * self.delta_t_per_data_us
                    event_data = self.event_slicers[location].get_events(t_s, t_end)

                    p = event_data['p']
                    t = event_data['t']
                    x = event_data['x']
                    y = event_data['y']

                    xy_rect = self.rectify_events(x, y, location)
                    x_rect = xy_rect[:, 0]
                    y_rect = xy_rect[:, 1]

                    if self.event_representation == 'voxel_grid':
                        event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
                    else:
                        events = np.stack([x_rect, y_rect, t, p], axis=1)
                        event_representation = data_util.generate_input_representation(events, self.event_representation,
                                                                  (self.height, self.width))
                        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)

                    if event_tensor is None:
                        event_tensor = event_representation
                    else:
                        event_tensor = torch.cat([event_tensor, event_representation], dim=0)

            else:
                num_bins_total = self.nr_events_data * self.num_bins
                event_tensor = torch.zeros((num_bins_total, self.height, self.width))
                self.nr_events = self.nr_events_data * self.nr_events_per_data
                event_data = self.event_slicers[location].get_events_fixed_num(ts_end, self.nr_events)

                if self.nr_events >= event_data['t'].size:
                    start_index = 0
                else:
                    start_index = -self.nr_events

                p = event_data['p'][start_index:]
                t = event_data['t'][start_index:]
                x = event_data['x'][start_index:]
                y = event_data['y'][start_index:]
                nr_events_loaded = t.size

                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                nr_events_temp = nr_events_loaded // self.nr_events_data
                events = np.stack([x_rect, y_rect, t, p], axis=-1)
                Parallel(n_jobs=8, backend="threading")(
                    delayed(self.generate_event_tensor)(i, events, event_tensor, nr_events_temp) for i in range(self.nr_events_data))

            # remove 40 bottom rows
            event_tensor = event_tensor[:, :-40, :]

            if self.resize:
                event_tensor = f.interpolate(event_tensor.unsqueeze(0),
                                             size=(self.shape_resize[0], self.shape_resize[1]),
                                             mode='bilinear', align_corners=True).squeeze(0)

            label_tensor = torch.from_numpy(label).long()

            if self.augmentation:
                value_flip = round(random.random())
                if value_flip > 0.5:
                    event_tensor = torch.flip(event_tensor, [2])
                    label_tensor = torch.flip(label_tensor, [1])
                    
                event_tensor, label_tensor = self.apply_augmentation(event_tensor,label_tensor)



        return {"event_voxel":event_tensor, "label":label_tensor}



        