import os
import numpy as np
import random
from os.path import join
from keras.utils import to_categorical
from utils import load_image_as_np_array, preprocess_images_tf
from utils import take


def get_class_to_idx_dict(classes):
    
    return { cl: idx for idx, cl in enumerate(sorted(classes)) }



def count_num_videos(dataset_dir, *split_keys):
    for split_key in split_keys:
        yield sum(len(os.listdir(join(dataset_dir, split_key, cl))) for cl in os.listdir(join(dataset_dir, split_key)))

    
def get_dataset_split_structure(dataset_split_path):
    
    #
    # This is the structure of dataset_structure; it resembles the directory structure of dataset_dir/train:
    # {
    #     class1: 
    #     {
    #          video1: ['frame1', 'frame2', ...],
    #          video2: ['frame1', 'frame2', ...],
    #          ...
    #          videoN: ...
    #     },
    #     ...
    #     classN
    # } 
    #
    
    return {cl: {video : sorted(os.listdir(join(dataset_split_path, cl, video)))
                 for video in os.listdir(join(dataset_split_path, cl))}
            for cl in os.listdir(dataset_split_path)}


    
def train_generator_single_images(dataset_dir, batch_size):
    
    base_dir = join(dataset_dir, 'train')
    
    dataset_structure = get_dataset_split_structure(base_dir)
    all_classes = list(dataset_structure.keys())
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
    
    
    while True:
        
        images = []
        labels = []
        
        for _ in range(batch_size):
            
            random_class = random.choice(all_classes)
            class_idx = class_to_idx_dict[random_class]
            
            random_video = random.choice(list(dataset_structure[random_class].keys()))
            random_frame = random.choice(dataset_structure[random_class][random_video])
            
            full_frame_path = join(base_dir, random_class, random_video, random_frame)
            image = load_image_as_np_array(full_frame_path)
            
            images.append(image)
            labels.append(class_idx)
                
        assert len(labels) == len(images)
        
        images = np.array(images)
        labels = to_categorical(np.array(labels), len(all_classes))
        
        yield preprocess_images_tf(images), labels
        
        
        
def valid_generator_single_images(dataset_dir):
    
    base_dir = join(dataset_dir, 'valid')
    
    dataset_structure = get_dataset_split_structure(base_dir)
    all_classes = dataset_structure.keys()
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
    
    while True:
        
        for cl in all_classes:
            
            class_idx = class_to_idx_dict[cl]
            
            for video in dataset_structure[cl]:
                
                frames = np.array([load_image_as_np_array(join(base_dir, cl, video, frame)) 
                                   for frame in dataset_structure[cl][video]])
                
                labels = [class_idx] * len(frames)
                labels = to_categorical(np.array(labels), len(all_classes))
                
                yield preprocess_images_tf(frames), labels

        

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

                frames = np.array([load_image_as_np_array(join(base_dir, cl, video, frame)) 
                                   for frame in dataset_structure[cl][video]])

                yield frames, class_idx
                
                
                
def _random_infinite_iterator_rnn(base_dir, dataset_structure):
    
    """
        This iterator yields infinitely all videos from a dataset in a random way.
        Returns:
            frames: np.array with shape (nb_frames, height, width, channels) of a video
            class_idx: The class index (a single number)
    """
    
    all_classes = list(dataset_structure.keys())
    class_to_idx_dict = get_class_to_idx_dict(all_classes)
    
    
    while True:
            
        random_class = random.choice(all_classes)
        class_idx = class_to_idx_dict[random_class]
            
        random_video = random.choice(list(dataset_structure[random_class].keys()))

        frames = np.array([load_image_as_np_array(join(base_dir, random_class, random_video, frame)) 
                       for frame in dataset_structure[random_class][random_video]])
        
        yield frames, class_idx
        
        
        
def _get_iterator(split_key):
    
    key_to_iterator = {
        'train': _random_infinite_iterator_rnn,
        'valid': _sequential_infinite_iterator_rnn
    }
    
    return key_to_iterator[split_key]
    
    

def frames_generator_rnn(dataset_dir, split_key, batch_size):
    
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
            videos, labels = map(np.array, zip(*data))
            
            # if batch_size == 1:
            #    videos = np.array([videos])
            #    labels = np.array([labels])
            # else:
            #    videos = np.array(videos)
            #    labels = np.array(labels)
            
            videos = preprocess_images_tf(videos)
            labels = to_categorical(labels, len(all_classes))

            yield videos, labels