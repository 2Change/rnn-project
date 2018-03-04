import cv2
import os
from os.path import join
from tqdm import tqdm


def get_number_frames_from_video(video_path):
    
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        raise ValueError('Couldnt open file ' + video_path)

    # WARNING: this value is not always accurate! Sometimes it could be an estimation
    # of the number of frames. 
    # See https://stackoverflow.com/questions/31472155/python-opencv-cv2-cv-cv-cap-prop-frame-count-get-wrong-numbers
    # We can still use this for estimating if the number of video frames is lower than the number required by us.
    # (or we could get rid of that)
    tot_frames_cv = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    
    vc.release()
    
    return tot_frames_cv
   


def get_number_frames_from_dataset(dataset, disable_progress_bar=False):
    
    dataset_path = join('datasets', dataset, 'video')
    for cl in tqdm(os.listdir(dataset_path), disable=disable_progress_bar, desc='Counting...'):
        class_path = join(dataset_path, cl)
        
        for video_folder in os.listdir(class_path):
            
            if video_folder == 'Annotation':
                continue
                
            video_folder_path = join(class_path, video_folder)
            
            for video in os.listdir(video_folder_path):
                
                video_path = join(video_folder_path, video)
                yield get_number_frames_from_video(video_path)