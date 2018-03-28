import os
import numpy as np
import random
from os.path import join
from keras.utils import to_categorical
from utils import load_image_as_np_array, preprocess_images_tf
from utils import take

from image_generators import get_class_to_idx_dict, count_num_videos


def get_dataset_split_structure(dataset_split_path):
    
    #
    # This is the structure of dataset_structure; it resembles the directory structure of dataset_dir/train:
    # {
    #     class1: [video1, video2, ...],
    #     ...
    #     classN: [video1, video2, ...]
    # } 
    #
    
    return {cl: sorted(os.listdir(join(dataset_split_path, cl))) for cl in os.listdir(dataset_split_path)}



def train_generator_single_images(dataset_dir, batch_size, cache_size=5000, additional_data=None):
    
    base_dir = join(dataset_dir, 'train')
    
    dataset_structure = get_dataset_split_structure(base_dir)
    all_classes = list(dataset_structure.keys())
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
    
    tot_videos = next(count_num_videos(dataset_dir, 'train'))
    num_calls = tot_videos // batch_size
    yield num_calls
    
    cache = {}
    
    while True:
        
        inception_frames = []
        labels = []
        add_data = []
        
        for _ in range(batch_size):
            
            random_class = random.choice(all_classes)
            class_idx = class_to_idx_dict[random_class]
            
            random_video = random.choice(dataset_structure[random_class])
            
            cache_key = join(random_class, random_video)
            video_array = cache.get(cache_key, None)
                
            if video_array is None:
            
                full_video_path = join(base_dir, random_class, random_video)
                video_array = np.load(full_video_path)
                
                if len(cache) < cache_size:
                    cache[cache_key] = video_array
           
        
            random_frame_idx = np.random.randint(len(video_array))
            
            random_frame = video_array[random_frame_idx]
            
            inception_frames.append(random_frame)
            labels.append(class_idx)
            
            if additional_data is not None:
                add_data.append(additional_data[random_class][random_video][random_frame_idx])
                
        assert len(labels) == len(inception_frames)
        
        inception_frames = np.array(inception_frames)
        labels = to_categorical(np.array(labels), len(all_classes))
        
        if additional_data is not None:
            
            add_data = np.array(add_data)
            yield [add_data, inception_frames], [labels]
            
        else:
            yield inception_frames, labels
        
        
        
def valid_generator_single_images(dataset_dir, additional_data=None):
    
    base_dir = join(dataset_dir, 'valid')
    
    dataset_structure = get_dataset_split_structure(base_dir)
    all_classes = dataset_structure.keys()
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
    
    tot_videos = next(count_num_videos(dataset_dir, 'valid'))
    yield tot_videos
    
    
    while True:
        
        for cl in all_classes:
            
            class_idx = class_to_idx_dict[cl]
            
            for video in dataset_structure[cl]:
                
                inception_frames = np.load(join(base_dir, cl, video))
                
                labels = [class_idx] * len(inception_frames)
                labels = to_categorical(np.array(labels), len(all_classes))
                
                if additional_data is not None:
                    
                    yield [additional_data[cl][video], inception_frames], [labels]
                    
                else:
                    yield inception_frames, labels
                


def dataset_loader(dataset_dir, split_key, mode):
    
    base_dir = join(dataset_dir, split_key)
    
    dataset_structure = get_dataset_split_structure(base_dir)
    all_classes = dataset_structure.keys()
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
        
    for cl in all_classes:

        class_idx = class_to_idx_dict[cl]

        for video in dataset_structure[cl]:

            inception_features = np.load(join(base_dir, cl, video))
            
            if mode == 'rnn':
                yield inception_features, class_idx
            elif mode == 'cnn': 
                for inception_feature in inception_features:
                    yield inception_feature, class_idx
                

def load_whole_dataset(dataset_dir, mode, split_keys):
    
    for split_key in split_keys:
        
        data = list(dataset_loader(dataset_dir, split_key, mode))
        X, Y = map(np.array, zip(*data))
        
        yield X, to_categorical(Y)
                
                
                
def _sequential_infinite_iterator_rnn(base_dir, dataset_structure):
    
    """
        This iterator yields infinitely all videos from a dataset in a sequential way.
        Returns:
            frames: np.array with shape (nb_frames, height, width, channels) of a video
            class_idx: The class index (a single number)
    """
    
    all_classes = dataset_structure.keys()
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
    
    while True:

        for cl in all_classes:

            class_idx = class_to_idx_dict[cl]

            for video in dataset_structure[cl]:

                inception_frames = np.load(join(base_dir, cl, video))

                yield cl, video, inception_frames, class_idx
                
                
                
def _random_infinite_iterator_rnn(base_dir, dataset_structure, cache_size=4096):
    
    """
        This iterator yields infinitely all videos from a dataset in a random way.
        Returns:
            frames: np.array with shape (nb_frames, height, width, channels) of a video
            class_idx: The class index (a single number)
    """
    
    all_classes = list(dataset_structure.keys())
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
    
    
    cache = {}
    
    while True:
            
        random_class = random.choice(all_classes)
        class_idx = class_to_idx_dict[random_class]
            
        random_video = random.choice(dataset_structure[random_class])
        
        cache_key = join(random_class, random_video)
        inception_frames = cache.get(cache_key, None)
        
        if inception_frames is None:
                
            inception_frames = np.load(join(base_dir, random_class, random_video))
            
            if len(cache) < cache_size:
                cache[cache_key] = inception_frames
        
        yield random_class, random_video, inception_frames, class_idx
        
        
        
def _get_iterator(split_key):
    
    key_to_iterator = {
        'train': _random_infinite_iterator_rnn,
        'valid': _sequential_infinite_iterator_rnn
    }
    
    return key_to_iterator[split_key]
    
    

def frames_generator_rnn(dataset_dir, split_key, batch_size, additional_data=None):
    
    """
        Internally this method uses _sequential_infinite_iterator_rnn to iterate on the validation set.
        If the batch_size is not divisible by the number of videos, the last videos will not be
        returned, but they will be returned at the next iteration over the valid set, so we might
        end with statistics that will be slightly different if computed multiple times on the
        validation set returned by this generator.
    """
    
    base_dir = join(dataset_dir, split_key)
    
    dataset_structure = get_dataset_split_structure(base_dir)
    all_classes = dataset_structure.keys()
    
    d_iterator = _get_iterator(split_key)(base_dir, dataset_structure)
    
    tot_videos = next(count_num_videos(dataset_dir, split_key))
    num_calls = tot_videos // batch_size
    yield num_calls
    
    
    while True:
        
        for _ in range(num_calls):
            
            data = take(d_iterator, batch_size)
            class_names, video_names, videos, labels = map(np.array, zip(*data))
            labels = to_categorical(labels, len(all_classes))
            
            if additional_data is not None:
                
                additional_data_batch = np.array([additional_data[cl][v] 
                                                  for cl, v in zip(class_names, video_names)])
                
                yield [additional_data_batch, videos], [labels]
                    
                
            else:
                yield videos, labels
            
            
def sparse_frame_generator(x, y, target_frames, batch_size):
    while True:
        # pick batch_size indexes between 0 and len(x)
        batch_indexes = np.random.choice(len(x), batch_size)
        batch_x, batch_y = x[batch_indexes], y[batch_indexes]
        batch_x = np.array([extract_n_random_frames(video, target_frames) for video in batch_x])
        yield batch_x, batch_y
        
        
def extract_n_random_frames(video, target_frames):
    # no need to explain
    return video[np.sort(np.random.choice(len(video), target_frames))]
    
    
    
    
    
    
    
    