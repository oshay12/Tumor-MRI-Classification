# isort was ran
import os

import cv2
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm as tqdm_iterator


# function to format data for the various CNN models
def load_data():
    labels_named = [
        "glioma_tumor",
        "meningioma_tumor",
        "no_tumor",
        "pituitary_tumor",
    ]
    data = []
    labels = []
    # reshaping images to 224x224px so shape is homogenous for all images.
    # 224 is chosen as it is the minimum for many CNNs that I have imported.
    image_size = 224

    # getting paths for this file and the data
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."),
    )
    data_path = os.path.join(project_root, "data")

    # loop through files
    for i, label in enumerate(labels_named):
        # get path to data within whichever folder is named the current label
        train_folder_path = os.path.join(data_path, "Training", label)
        # loop through folder, get images
        for image_file in tqdm_iterator(os.listdir(train_folder_path)):
            # read image
            img = cv2.imread(os.path.join(train_folder_path, image_file))
            # reshape image to (num_samples, 128, 128)
            img = cv2.resize(img, (image_size, image_size))
            # append image and label to arrays
            data.append(img)
            labels.append(i)

    for i, label in enumerate(labels_named):
        # get path to data within whichever folder is named the current label
        test_folder_path = os.path.join(data_path, "Testing", label)
        # loop through folder, get images
        for image_file in tqdm_iterator(os.listdir(test_folder_path)):
            # read image
            img = cv2.imread(os.path.join(test_folder_path, image_file))
            # reshape image to (num_samples, 128, 128)
            img = cv2.resize(img, (image_size, image_size))
            # append image and label to arrays
            data.append(img)
            labels.append(i)

    # change lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # one-hot encode labels from single labels to matricies,
    # ex. label '1' => [0, 1, 0, 0]
    labels = to_categorical(labels, 4)

    return data, labels
