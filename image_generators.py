import os
import numpy as np
import random
from os.path import join
from keras.utils import to_categorical
from utils import load_image_as_np_array, preprocess_images_tf


def get_class_to_idx_dict(classes):
    
    return { cl: idx for idx, cl in enumerate(sorted(classes)) }

    
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
            random_frame = random.choice(list(dataset_structure[random_class][random_video]))
            
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