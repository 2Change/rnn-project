import cv2
import numpy as np
import os
from os.path import join, isdir
import argparse
import pprint
from tqdm import tqdm
from im_utils import imresize
from collections import defaultdict
from utils import preprocess_images_tf
from PIL import Image


def get_n_frames_from_video(video_path, n_frames):
    
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        print('Couldnt open file ' + video_path)
        return None

    # WARNING: this value is not always accurate! Sometimes it could be an estimation
    # of the number of frames. 
    # See https://stackoverflow.com/questions/31472155/python-opencv-cv2-cv-cv-cap-prop-frame-count-get-wrong-numbers
    # We can still use this for estimating if the number of video frames is lower than the number required by us.
    # (or we could get rid of that)
    tot_frames_cv = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if tot_frames_cv < n_frames:
        print('Too less frames: ' + video_path + '. File is discarded')
        return None
    
    # Since we cannot rely on the exact number returned by vc.get(cv2.CAP_PROP_FRAME_COUNT),
    # we are going to save temporarily all frames into the frames variable, so we know
    # the exact number of frames.
    
    frames = []

    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            break
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    vc.release()
    
    tot_frames_exact = len(frames)
    one_frame_every_n = tot_frames_exact // n_frames
    
    if one_frame_every_n == 0:   # I don't know why this happens
        return None

    frames = frames[::one_frame_every_n]
    
    # By using the above the line, we usually take more frames than the number required.
    # Here we make sure to take exactly (the first) n_frames.
    frames = frames[:n_frames]
    frames = np.array(frames)

    if len(frames.shape) == 4 and frames.shape[0] == n_frames:
        return frames
    else:
        print('Problem with', video_path, 'frames.shape', frames.shape, 'tot_frames is', tot_frames_exact)   


def get_filenames_and_frames_from_subdir(subdir_path, n_frames):
    
    for filename in os.listdir(subdir_path):
        subdir_and_filename = join(subdir_path, filename)
        frames = get_n_frames_from_video(subdir_and_filename, n_frames)
        if frames is not None:
            yield filename, frames



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', help='UCF11, ...', default='UCF11')
parser.add_argument('--out_format', help='Choose output format', choices=['images', 'inception'], default='images')
parser.add_argument('--nb_frames', help='Number of frames per clip', type=int, default=50)
parser.add_argument('--train_perc', help='Percentage of training set', type=float, default=0.7)
parser.add_argument('--out_height', help='Ouptut height for each frame', type=int, default=240)
parser.add_argument('--out_width', help='Output width for each frame', type=int, default=320)

args = vars(parser.parse_args())

print('*' * 40)
print('Input parameters:')
pprint.pprint(args)
print('*' * 40)

base_dataset_path_without_video = join('datasets', args['dataset'])
base_dataset_path = join(base_dataset_path_without_video, 'video')
classes = [d for d in os.listdir(base_dataset_path) if isdir(join(base_dataset_path, d))]

print(classes)

data = defaultdict(lambda: defaultdict(list))

separate_files_out_dir = 'separate_frames_{}_h_{}_w_{}_{}'.format(args['nb_frames'], args['out_height'], args['out_width'], args['out_format'])

if args['out_format'] == 'inception':
    
    from keras.applications.inception_v3 import InceptionV3
    from utils import preprocess_images_tf
    from keras_utils import set_keras_session

    set_keras_session()
    
    inception = InceptionV3(include_top=False, pooling='avg')

# for every class
for label_idx, class_dir in enumerate(sorted(classes)):
    
    base_class_path = join(base_dataset_path, class_dir)
    subdirs = [d for d in os.listdir(base_class_path) if isdir(join(base_class_path, d)) and d != 'Annotation']
    subdirs.sort()

    num_subdirs = len(subdirs)
    idx_subdirs_train = int(round(args['train_perc'] * num_subdirs)) 
    
    # split the video collection in train test
    subdirs_split = {key: subdirs[range_min:range_max] for key, (range_min, range_max) in zip(('train', 'valid'), ((0, idx_subdirs_train), (idx_subdirs_train, num_subdirs)))}

    print(' ')
    print(class_dir)
    print(subdirs_split)
    print(' ')
    
    for key in subdirs_split:
        subdirs_subset = subdirs_split[key]
        separate_split_out_dir = join(base_dataset_path_without_video, separate_files_out_dir, key, class_dir)
        if not os.path.exists(separate_split_out_dir):
            os.makedirs(separate_split_out_dir)

        for subdir in tqdm(subdirs_subset, desc=class_dir + ' ' + key):
         
            for filename, frames in get_filenames_and_frames_from_subdir(join(base_class_path, subdir), args['nb_frames']):
                # print filename, frames.shape
                if frames.shape[1] != args['out_height'] or frames.shape[2] != args['out_width']:
                    frames = imresize(frames, (args['out_height'], args['out_width']), mode='RGB')
                
                video_dir = join(separate_split_out_dir, filename[:filename.rfind('.')])
                
                if args['out_format'] == 'images':
                    
                    os.makedirs(video_dir)

                    for frame_idx, frame in enumerate(frames):
                        im = Image.fromarray(frame)
                        im.save(join(video_dir, "frame_{:02}.jpg".format(frame_idx)))
                else:
                    
                    frames = preprocess_images_tf(frames)
                    inception_output = inception.predict(frames)
                    
                    np.save(video_dir, inception_output)
                        

print('Script ended.')
