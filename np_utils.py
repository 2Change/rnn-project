from __future__ import division

import numpy as np


def batch(arrays, batch_size, shuffle=False, include_last=False):
    """

    Returns a one-time generator that iterates over the specified arrays
    returning one batch at a time

    :param arrays: single array or list of arrays (should have equal length)
    :param batch_size: dimension of the batch size
    :param shuffle: input arrays are initially shuffled maintaining correspondent elements
                    of different arrays at the same index
    :param include_last: include last batch when the length of the arrays is not divisible
                        by batch_size.  The returned arryas will have dimension 0 < batch_size

    :return a generator that returns
                * a list of batches if arrays is a list of 2 or more elements
                * a batch if arrays is a single array

    """
    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]

    if shuffle:
        permutation = np.random.permutation(arrays[0].shape[0])
        arrays = [array[permutation] for array in arrays]

    nb_elements = len(arrays[0])
    num_batches = nb_elements // batch_size
    for batch_num in range(num_batches):
        arrays_batch = [array[(batch_num) * batch_size: (batch_num + 1) * batch_size] for array in arrays]
        if len(arrays_batch) == 1:
            arrays_batch = arrays_batch[0]
        yield arrays_batch

    if include_last and num_batches * batch_size < nb_elements:
        last_arrays_batch = [array[num_batches * batch_size:] for array in arrays]
        if len(last_arrays_batch) == 1:
            last_arrays_batch = last_arrays_batch[0]
        yield last_arrays_batch


def get_batch(inputs, batch_num, batch_size=64, offset=0):
    """

    Get a specific batch from a list of arrays
    :param inputs: list of input arrays (should have equal length)
    :param batch_num: number of required batch
    :param batch_size: dimension of the batch size
    :param offset: offset from start of the array

    :return a list of batch from the specified arrays

    """
    return [input[offset + (batch_num) * batch_size: offset + (batch_num + 1) * batch_size] for input in inputs]

def shuffle(arrays):
    """

    Shuffle one or more arrays maintaining correspondent elements
    of different arrays at the same index

    :param arrays: single array or list of arrays (should have equal length)

    :return a generator that returns
                * a list of shuffled arrays if the input has 2 or more elements
                * the shuffled array if input contains only one element

    """
    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]

    permutation = np.random.permutation(arrays[0].shape[0])
    arrays = [array[permutation] for array in arrays]

    if len(arrays) == 1:
        arrays = arrays[0]
    return arrays

def to_onehot(classes, num_classes=None):
    from keras.utils import to_categorical

    return to_categorical(classes, num_classes)

def normalize_between(values, new_range, old_range=None):

    """

    Change the range of an input array between the specified min and max values

    :param values: input array
    :param new_range: new minimum and maximum (tuple, list, ..)
    :param old_range: old minimum and maximum of the input values. If none, they will be computed from the input array
    :return: new array with the new specified range

    Author: Riccardo Albertazzi
    """

    new_min, new_max = new_range

    if old_range is None:
        old_min = np.min(values)
        old_max = np.max(values)
    else:
        old_min, old_max = old_range

    return (new_max - new_min) / (old_max - old_min) * (values - old_min) + new_min
