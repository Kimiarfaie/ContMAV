import os
import glob
import numpy as np
import cv2
from ..dataset_base import DatasetBase

class LIACiDataset(DatasetBase):
    SPLITS = ["train", "val", "test"]  # Define your splits as needed

    def __init__(self, data_dir=None, n_classes=8, split="train", with_input_orig=False, overfit=False, classes=8):
        super(LIACiDataset, self).__init__()
        assert split in self.SPLITS, f"Split {split} not recognized. Available splits: {self.SPLITS}"
        self._n_classes = n_classes
        self._split = split
        self._with_input_orig = with_input_orig
        self.overfit = overfit
        self._cameras = ['camera1'] # dummy :D we dont have camera info

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            # Paths to images and labels
            self.images_path = os.path.join(data_dir, split, "images")
            self.labels_path = os.path.join(data_dir, split, "labels")

            # Load the image and label file paths
            self.images = sorted(glob.glob(os.path.join(self.images_path, "*.jpg"))) 
            self.labels = sorted(glob.glob(os.path.join(self.labels_path, "*.png")))  

        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_colors_without_void(self):
        return self.class_colors[1:]

    @property
    def class_names_without_void(self):
        return self.class_names[1:]

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def class_names(self):
        return ["void", "see_chest_grating", "paint_peel", "overboard_valves", "defect", "corrosion", "propeller", "Anod", "shiphull"] 

    @property
    def class_colors(self):
        return [(0, 0, 0), (255, 255, 255), (255, 0, 0), (64, 224, 208), (254, 193, 203), (255, 255, 0), (128, 0, 128), (0, 255, 255), (0, 0, 255)]
    

    @property
    def n_classes(self):
        return self._n_classes + 1 

    # added
    @n_classes.setter
    def n_classes(self, value):
        self._n_classes = value

    @property
    def split(self):
        return self._split

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, filename):
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_name(self, idx):
        return self.images[idx]

    def load_image(self, idx):
        image = self._load(self.images[idx])
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        return image

    def load_label(self, idx):
        label = self._load(self.labels[idx])
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        #print(f"Label range for sample {idx}: min={label.min()}, max={label.max()}, unigue:{np.unique(label)}")
        return label

    def __len__(self):
        if self.overfit:
            return 2
        return len(self.images)
