import cv2
import numpy as np
import os
from os.path import join, isdir
import argparse
import pprint
from tqdm import tqdm
from im_utils import imresize
from collections import defaultdict


def get_n_frames_from_video(video_path, n_frames):
    
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        print 'Couldnt open file', video_path
        return None

    tot_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    if tot_frames < n_frames:
        print 'Too less frames', video_path, 'File is discarded'
        return None

    one_frame_every_n = tot_frames // n_frames
    returned_frames = 0
    
    frames = []
    frame_idx = 0

    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            break
    
        frame_idx += 1

        if frame_idx % one_frame_every_n == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) == n_frames:
                break

    vc.release()
    
    frames = np.array(frames)

    assert len(frames.shape) == 4 and frames.shape[0] == n_frames
    return frames    



def get_filenames_and_frames_from_subdir(subdir_path, n_frames):
    
    for filename in os.listdir(subdir_path):
        subdir_and_filename = join(subdir_path, filename)
        frames = get_n_frames_from_video(subdir_and_filename, n_frames)
        if frames is not None:
            yield subdir_and_filename, frames



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='UCF11, ...', default='UCF11')
parser.add_argument('--nb_frames', help='Number of frames per clip', type=int, default=50)
parser.add_argument('--valid_perc', help='Percentage of validation set', type=float, default=0.2)
parser.add_argument('--test_perc', help='Percentage of test set', type=float, default=0.2)
parser.add_argument('--out_height', help='Ouptut height for each frame', type=int, default=240)
parser.add_argument('--out_width', help='Output width for each frame', type=int, default=320)

args = vars(parser.parse_args())

print('*' * 40)
print('Input parameters:')
pprint.pprint(args)
print('*' * 40)

base_dataset_path = join('datasets', args['dataset'])
classes = [d for d in os.listdir(base_dataset_path) if isdir(join(base_dataset_path, d))]

print(classes)

train_perc = 1. - args['valid_perc'] - args['test_perc']

data = defaultdict(lambda: defaultdict(list))


for label_idx, class_dir in enumerate(classes):
    
    base_class_path = join(base_dataset_path, class_dir)
    subdirs = [d for d in os.listdir(base_class_path) if isdir(join(base_class_path, d)) and d != 'Annotation']
    subdirs.sort()

    num_subdirs = len(subdirs)
    idx_subdirs_train = int(round(train_perc * num_subdirs)) 
    idx_subdirs_valid = int(round((train_perc + args['valid_perc']) * num_subdirs))
    
    subdirs_split = {key : subdirs[range_min:range_max] for key, (range_min, range_max) in zip(('train', 'valid', 'test'), ((0, idx_subdirs_train), (idx_subdirs_train, idx_subdirs_valid), (idx_subdirs_valid, num_subdirs))) }
    

    
    print
    print class_dir
    print subdirs_split
    print
    
    for key in subdirs_split:
        subdirs_subset = subdirs_split[key]

        for subdir in tqdm(subdirs_subset, desc=class_dir + ' ' + key):
            for filename, frames in get_filenames_and_frames_from_subdir(join(base_class_path, subdir), args['nb_frames']):
            # print filename, frames.shape
                if frames.shape[1] != args['out_height'] or frames.shape[2] != args['out_width']:
                    frames = imresize(frames, (args['out_height'], args['out_width']), mode='RGB')

                data[key]['frames'].append(frames)
                data[key]['labels'].append(label_idx)
                data[key]['filenames'].append(filename)





