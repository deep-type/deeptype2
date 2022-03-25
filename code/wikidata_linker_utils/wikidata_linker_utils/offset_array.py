from os.path import exists
import numpy as np
from .successor_mask import (
    convert_to_offset_array, make_dense, make_sparse, padded_gather, batch_is_related,
    batch_is_related_pair, batch_is_related_pair_broadcast
)

SPARSE_ATTRIBUTE_SUFFIX = "_values.sparse.npy"


def count_non_zero(dense):
    return len(np.nonzero(dense[1:] - dense[:-1])[0]) + int(dense[0] != 0)


def should_compress(dense):
    nonzeros = count_non_zero(dense)
    return (2 * nonzeros + 1) < 0.5 * len(dense)


class OffsetArray(object):
    def __init__(self, values, offsets):
        self.values = values
        self.offsets = offsets

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int64, np.int32)):
            end = self.offsets[idx]
            start = 0 if idx == 0 else self.offsets[idx - 1]
            return self.values[start:end]
        else:
            idx = np.asarray(idx)
            ends = self.offsets[idx]
            starts = self.offsets[idx - 1]
            starts[idx == 0] = 0
            return [self.values[start:end] for start, end in zip(starts, ends)]
    
    def padded_gather(self, idx, pad_with=-1):
        if isinstance(idx, (int, np.int64, np.int32)):
            return self.__getitem__(idx)
        else:
            if not isinstance(idx, np.ndarray):
                idx = np.array(idx)
            if idx.ndim != 1:
                idx_flat = idx.reshape(-1)
            else:
                idx_flat = idx
            res = padded_gather(idx_flat, self.offsets, self.values, pad_with)
            if idx.ndim != 1:
                res = res.reshape(list(idx.shape) + [res.shape[1]])
            return res

    def batch_is_related(self, sources, destinations=None, direct=True, indirect=False, condition=None, max_relatable=-1, related_or_empty=False):
        assert isinstance(sources, np.ndarray)
        assert direct or indirect, "direct or indirect must be true."
        if destinations is None:
            assert isinstance(indirect, bool) and indirect is False
            assert isinstance(direct, bool) and direct is True
            if sources.ndim != 2:
                sources_2d = sources.reshape(sources.shape[0], -1)
            else:
                sources_2d = sources
            res = batch_is_related(sources_2d, self.offsets, self.values, condition=condition, max_relatable=max_relatable, related_or_empty=related_or_empty)
            if sources.ndim != 2:
                res = res.reshape(sources.shape)
        else:
            assert isinstance(destinations, np.ndarray)
            # destinations can be a broadcasted object. (e.g. [BATCH, BEAMS, etc...])
            assert destinations.ndim == sources.ndim, "when passing destinations and sources, both arrays must have same rank."
            should_broadcast = False
            for axis in range(destinations.ndim - 1):
                should_broadcast = destinations.shape[axis] != sources.shape[axis]
                if should_broadcast:
                    break
            if should_broadcast or destinations.ndim == 3:
                assert destinations.ndim <= 3 and destinations.ndim > 0
                if destinations.ndim < 3:
                    sources_3d = sources.reshape([1 for _ in range(3 - sources.ndim)] + list(sources.shape))
                    destinations_3d = destinations.reshape([1 for _ in range(3 - destinations.ndim)] + list(destinations.shape))
                else:
                    sources_3d, destinations_3d = sources, destinations
                res = batch_is_related_pair_broadcast(sources_3d, destinations_3d, self.offsets, self.values, direct=direct, indirect=indirect, condition=condition, max_relatable=max_relatable, related_or_empty=related_or_empty)
                if destinations.ndim < 3:
                    res = res.reshape([max(s1, s2) for s1, s2 in zip(sources.shape[:-1], destinations.shape[:-1])]+ [sources.shape[-1]])
            else:
                if sources.ndim != 2:
                    sources_2d = sources.reshape(sources.shape[0], -1)
                    destinations_2d = destinations.reshape(destinations.shape[0], -1)
                else:
                    sources_2d = sources
                    destinations_2d = destinations
                res = batch_is_related_pair(sources_2d, destinations_2d, self.offsets, self.values, direct=direct, indirect=indirect, condition=condition, max_relatable=max_relatable, related_or_empty=related_or_empty)
                if sources.ndim != 2:
                    res = res.reshape(sources.shape)
        return res

    def insert(self, idx, values, update_offsets=True):
        assert isinstance(idx, int), "can only do inserts using an integer index"
        prev_length = len(self.offsets)
        if idx == prev_length:
            self.offsets = np.concatenate([self.offsets, [self.offsets[-1] + len(values)]])
            update_offsets = False
        old_end = self.offsets[idx]
        self.values = np.concatenate([self.values[:old_end], np.asarray(values).astype(np.int32), self.values[old_end:]])
        # shift everyone forward by this new amount
        if update_offsets:
            self.offsets[idx:] += len(values)

    def is_empty(self, idx):
        end = self.offsets[idx]
        start = 0 if idx == 0 else self.offsets[idx - 1]
        return start == end

    def size(self):
        return self.offsets.shape[0]

    def __len__(self):
        return self.size()

    def edges(self):
        num_edges = np.zeros(len(self.offsets), dtype=np.int32)
        num_edges[0] = self.offsets[0]
        num_edges[1:] = self.offsets[1:] - self.offsets[:-1]
        return num_edges

    @classmethod
    def load(cls, path, compress=True, mmap_mode=None):
        values = np.load(path + "_values.npy")
        if exists(path + "_offsets.sparse.npy"):
            offsets_compressed = np.load(path + "_offsets.sparse.npy", mmap_mode=mmap_mode)
            offsets = make_dense(offsets_compressed, cumsum=True)
        else:
            # legacy mode, load dense versions:
            offsets = np.load(path + "_offsets.npy", mmap_mode=mmap_mode)
            if compress:
                if should_compress(offsets):
                    offsets_compressed = make_sparse(offsets)
                    np.save(path + "_offsets.sparse.npy", offsets_compressed)
            # optionally delete the old version here
        return OffsetArray(
            values,
            offsets
        )


def convert_dict_to_offset_array(dictionary, num_values):
    offsets = np.zeros(num_values, dtype=np.int32)
    total_num_values = sum(len(v) for _, v in dictionary.items())
    values = np.zeros(total_num_values, dtype=np.int32)
    position = 0
    for key, value in sorted(dictionary.items(), key=lambda x: x[0]):
        values[position:position + len(value)] = value
        position += len(value)
        offsets[key] = len(value)
    np.cumsum(offsets, out=offsets)
    return values, offsets


def save_values_offsets(values, offsets, path):
    np.save(path + "_values.npy", values)
    if should_compress(offsets):
        compressed_offsets = make_sparse(offsets)
        np.save(path + "_offsets.sparse.npy", compressed_offsets)
    else:
        np.save(path + "_offsets.npy", offsets)


def save_record_with_offset(path, index2indices, total_size=None):
    if isinstance(index2indices, dict):
        if total_size is None:
            raise ValueError("cannot leave total_size None "
                             "when using a dict.")
        values, offsets = convert_dict_to_offset_array(index2indices, total_size)
    else:
        values, offsets = convert_to_offset_array(index2indices)
    save_values_offsets(values, offsets, path)


def load_sparse(path):
    compressed = np.load(path)
    dense = make_dense(compressed, cumsum=False)
    non_zero_indices = compressed[1::2]
    mask = np.zeros(len(dense), dtype=np.bool)
    mask[non_zero_indices] = True
    return dense, mask


class SparseAttribute(object):
    def __init__(self, dense, mask):
        self.dense = dense
        self.mask = mask

    def __lt__(self, value):
        return np.logical_and(self.dense < value, self.mask)

    def __le__(self, value):
        return np.logical_and(self.dense <= value, self.mask)

    def __gt__(self, value):
        return np.logical_and(self.dense > value, self.mask)

    def __ge__(self, value):
        return np.logical_and(self.dense >= value, self.mask)

    def __eq__(self, value):
        return np.logical_and(self.dense == value, self.mask)

    @classmethod
    def load(cls, path):
        dense, mask = load_sparse(path + SPARSE_ATTRIBUTE_SUFFIX)
        return SparseAttribute(
            dense, mask
        )
