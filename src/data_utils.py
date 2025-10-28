import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

def extract_frames(video_path, num_frames=16):
    """
    Extract frames from a video file.

    Args:
    - video_path (str): Path to the video file.
    - num_frames (int): Number of frames to extract (default is 16).

    Returns:
    - frames (np.array): Array of extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(np.zeros((112, 112, 3), np.uint8))
    
    return np.array(frames)

def load_data(labels, video_dir, num_classes, num_frames=16):
    """
    Load and preprocess video data for training or testing.
    """
    X = []
    y = []
    
    for idx, row in tqdm(labels.iterrows(), total=labels.shape[0]):
        video_path = os.path.join(video_dir, row['video_name'])
        frames = extract_frames(video_path, num_frames)
        if len(frames) == num_frames:
            X.append(frames)
            y.append(row['tag'])
    
    X = np.array(X)
    y = to_categorical(pd.factorize(y)[0], num_classes)
    
    return X, y
