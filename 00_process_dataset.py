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


def get_n_frames_from_video(video_path, n_frames, min_frames):
    
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        print('Couldnt open file ' + video_path)
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
    if args['out_format'] == 'optical':
        # we need 2 frame per optical flow
        min_frames += 1
        
    tot_frames_exact = len(frames)

    if tot_frames_exact < min_frames:
        print('Too less frames: ' + video_path + '. File is discarded')
        return None
    elif tot_frames_exact < n_frames:
        return np.array(frames)
    
    if args['out_format'] == 'optical':
        one_frame_every_n = (tot_frames_exact-1)// n_frames
        if one_frame_every_n == 0:   # I don't know why this happens
            return None
        
        hsv = np.zeros_like(frames[0])
        hsv[...,1] = 255
        optical = []
        for i in range(1, len(frames), one_frame_every_n):
                prev = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 25, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                optical.append(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB))
                if len(optical) == n_frames:
                    return np.array(optical)
                

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


def get_filenames_and_frames_from_subdir(subdir_path, n_frames, min_frames=None):

    if min_frames == None:
        min_frames = n_frames 
    
    for filename in os.listdir(subdir_path):
        subdir_and_filename = join(subdir_path, filename)
        frames = get_n_frames_from_video(subdir_and_filename, n_frames, min_frames)
        if frames is not None:
            yield filename, frames



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', help='UCF11, ...', default='UCF11')
parser.add_argument('--out_format', help='Choose output format', choices=['images', 'inception', 'optical', 'yolo'], default='images')
parser.add_argument('--nb_frames', help='Number of frames per clip', type=int, default=50)
parser.add_argument('--min_frames', help='Minimum number of frames per clip', type=int, default=0)
parser.add_argument('--padding', help='pad sequences with less than nb_frames', action='store_true')
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

separate_files_out_dir = 'separate_frames_{}_h_{}_w_{}_{}_padding_{}'.format(args['nb_frames'], args['out_height'], args['out_width'], args['out_format'], args['padding'])

if args['out_format'] == 'inception':
    
    from keras.applications.inception_v3 import InceptionV3
    from utils import preprocess_images_tf
    
    inception = InceptionV3(include_top=False, pooling='avg')

    
if args['out_format'] == 'yolo':
    
    import darkflow_utils
    
    tfnet = darkflow_utils.tfnet_load()
    
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

            if args['padding']:
                frames_generator = get_filenames_and_frames_from_subdir(join(base_class_path, subdir), args['nb_frames'], args['min_frames'])
            else:
                frames_generator = get_filenames_and_frames_from_subdir(join(base_class_path, subdir), args['nb_frames']) 
         
            for filename, frames in frames_generator:
                # print filename, frames.shape
                if frames.shape[1] != args['out_height'] or frames.shape[2] != args['out_width']:
                    frames = imresize(frames, (args['out_height'], args['out_width']), mode='RGB')
                
                video_dir = join(separate_split_out_dir, filename[:filename.rfind('.')])
                
                if args['out_format'] == 'images' or args['out_format'] == 'optical':
                    
                    os.makedirs(video_dir)

                    for frame_idx, frame in enumerate(frames):
                        im = Image.fromarray(frame)
                        im.save(join(video_dir, "frame_{:02}.jpg".format(frame_idx)))
                
                elif args['out_format'] == 'inception':
                    
                    frames = preprocess_images_tf(frames)
                    inception_output = inception.predict(frames)
                    inception_frames = inception_output.shape[0]

                    if inception_frames < args['nb_frames']:
                        padding = np.zeros((args['nb_frames'] - inception_frames, inception_output.shape[1]))
                        inception_output = np.concatenate([padding, inception_output])

                    np.save(video_dir, inception_output)
                    
                elif args['out_format'] == 'yolo':
                    
                    yolo_features = darkflow_utils.tfnet_predict(tfnet, frames)
                    np.save(video_dir, yolo_features)
                    

print('Script ended.')

