#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import namedtuple
import math
from typing import Dict, Tuple
import h5py
import hdf5plugin
from numba import jit



class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError

class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2*pol-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:
                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int, max_events_per_data: int = -1) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        # print(t_start_us, self.t_offset)
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            print('Error', 'start', t_start_us, 'end', t_end_us)
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        # if max_events_per_data != -1 and events['t'].size > max_events_per_data:
        #     idx = np.round(np.linspace(0, events['t'].size - 1, max_events_per_data)).astype(int)
        #     for key in events.keys():
        #         events[key] = events[key][idx]
        return events

    def get_events_fixed_num(self, t_end_us: int, nr_events: int = 100000) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) with fixed number of events
        Parameters
        ----------
        t_end_us: end time in microseconds
        nr_events: number of events to load
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        # t_start_us = t_end_us - 1000
        # assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        # t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        # print(t_start_us, self.t_offset)
        t_end_lower_ms, t_end_upper_ms = self.get_conservative_ms(t_end_us)
        t_end_lower_ms_idx = self.ms2idx(t_end_lower_ms)
        t_end_upper_ms_idx = self.ms2idx(t_end_upper_ms)

        if t_end_lower_ms_idx is None or t_end_upper_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_end_lower_ms_idx:t_end_upper_ms_idx])
        _, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_end_us, t_end_us)
        t_end_us_idx = t_end_lower_ms_idx + idx_end_offset
        t_start_us_idx = t_end_us_idx - nr_events
        if t_start_us_idx < 0:
            t_start_us_idx = 0

        for dset_str in self.events.keys():
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])

        return events

    def get_events_fixed_num_recurrent(self, t_start_us_idx: int, t_end_us_idx: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) with fixed number of events
        Parameters
        ----------
        t_start_us_idx: start id
        t_end_us_idx: end id
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us_idx < t_end_us_idx

        events = dict()
        for dset_str in self.events.keys():
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])

        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    def get_conservative_ms(ts_us: int) -> Tuple[int, int]:
        """Convert time in microseconds into milliseconds
        ----------
        ts_us:    time in microseconds
        Returns
        -------
        ts_lower_ms:    lower millisecond
        ts_upper_ms:    upper millisecond
        """
        ts_lower_ms = math.floor(ts_us / 1000)
        ts_upper_ms = math.ceil(ts_us / 1000)
        return ts_lower_ms, ts_upper_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]
    

def generate_input_representation(events, event_representation, shape, nr_temporal_bins=5, separate_pol=True, normalize_event= False):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    if event_representation == 'histogram':
        return generate_event_histogram(events, shape)
    elif event_representation == 'voxel_grid':
        return generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol, normalize_event)


def generate_event_histogram(events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    height, width = shape
    x, y, t, p = events.T
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    
    mask = (x < width) & (x >= 0) & (y < height) & (height >= 0) 
    x=x[mask]
    y=y[mask]
    p=p[mask]
    
    p[p == 0] = -1  # polarity should be +1 / -1
    img_pos = np.zeros((height * width,), dtype="float32")
    img_neg = np.zeros((height * width,), dtype="float32")

    np.add.at(img_pos, x[p == 1] + width * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + width * y[p == -1], 1)

    histogram = np.stack([img_neg, img_pos], 0).reshape((2, height, width))

    return histogram


def normalize_voxel_grid(events):
    """Normalize event voxel grids"""
    nonzero_ev = (events != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        events = mask * (events - mean) / stddev

    return events

def normalize_voxel_grid_numpy(voxel_grid):
    """Normalize event voxel grids"""
    mask = np.nonzero(voxel_grid)
    if mask[0].shape[0] > 0:
        mean = voxel_grid[mask].mean()
        std = voxel_grid[mask].std()
        if std > 0:
            voxel_grid[mask] = (voxel_grid[mask] - mean) / std
        else:
            voxel_grid[mask] = voxel_grid[mask] - mean
    return voxel_grid


def generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol=True, normalize_event=False):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param nr_temporal_bins: number of bins in the temporal axis of the voxel grid
    :param shape: dimensions of the voxel grid
    """
    height, width = shape
    assert(events.shape[1] == 4)
    assert(nr_temporal_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # events[:, 2] = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    xs = events[:, 0].astype(np.int64)
    ys = events[:, 1].astype(np.int64)
    # ts = events[:, 2]
    # print(ts[:10])
    ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT

    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = np.abs(pols) * (1.0 - dts)
    vals_right = np.abs(pols) * dts
    pos_events_indices = pols == 1

    # Positive Voxels Grid
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)

    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height, vals_left[valid_indices_pos])

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              (tis[valid_indices_pos] + 1) * width * height, vals_right[valid_indices_pos])

    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)

    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              tis[valid_indices_neg] * width * height, vals_left[valid_indices_neg])

    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              (tis[valid_indices_neg] + 1) * width * height, vals_right[valid_indices_neg])

    voxel_grid_positive = np.reshape(voxel_grid_positive, (nr_temporal_bins, height, width))
    voxel_grid_negative = np.reshape(voxel_grid_negative, (nr_temporal_bins, height, width))

    if separate_pol:
        if normalize_event:
            fn = normalize_voxel_grid_numpy
        else:
            fn = lambda x:x
        return fn(np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0))

    voxel_grid = voxel_grid_positive - voxel_grid_negative
    if normalize_event:
        voxel_grid = normalize_voxel_grid_numpy(voxel_grid)
    return voxel_grid

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels_6_Cityscapes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 0, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 1, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 1, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 1, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 2, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 2, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 2, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 3, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 3, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 1, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 4, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 4, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 5, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 5, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 5, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 5, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 5, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 5, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

Id2label_6_Cityscapes = {label.id: label for label in reversed(labels_6_Cityscapes)}

labels_11_Cityscapes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 5, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 6, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 1, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 9, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 2, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 4, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 10, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 10, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 7, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 7, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 0, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 3, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 3, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 8, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 8, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 8, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 8, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 8, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 8, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

Id2label_11_Cityscapes = {label.id: label for label in reversed(labels_11_Cityscapes)}


def fromIdToTrainId(imgin, Id2label):
    imgout = imgin.copy()
    for id in Id2label:
        imgout[imgin == id] = Id2label[id].trainId
    return imgout


def shiftUpId(imgin):
    imgout = imgin.copy() + 1
    return imgout


def shiftDownId(imgin):
    imgout = imgin.copy()
    imgout[imgin == 0] = 256  # ignore label + 1
    imgout -= 1
    return imgout