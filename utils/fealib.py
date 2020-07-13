import tensorflow as tf
from tensorflow.train import Feature, Features, Example

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report 

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

import os
import sys
from IPython.display import display 
from io import BytesIO
import time
import datetime
import warnings

from PIL import Image
from skimage import io
import logging
import random


class TFRECParser:
    """Class to parse the TFRecord files that come with the original dataset.
    """
    __TFREC_DESCRIPTOR = {"image_name": tf.io.FixedLenFeature([], tf.string),
                          "image": tf.io.FixedLenFeature([], tf.string)}

    def __init__(self) -> None:
        self.__dataset = None # tf.data.TFRecordDataset

    @tf.autograph.experimental.do_not_convert
    def load(self, path_tfrec: str = 'tfrecords/train*.tfrec') -> None:
        def parser(serialized_example: Example) -> Example:
            example = tf.io.parse_single_example(serialized_example,
                                                 features=TFRECParser.__TFREC_DESCRIPTOR)
            return example
        self.__dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path_tfrec))
        self.__dataset = self.__dataset.map(parser)

    def get_dataset(self) -> tf.data.Dataset:
        return self.__dataset

    def get_image_arr(self, image_name: str) -> np.ndarray:
        record_dataset = self.__dataset.filter(lambda example: tf.equal(example["image_name"], image_name))
        example = next(iter(record_dataset))
        arr = tf.image.decode_jpeg(example['image'], channels=3).numpy()
        return arr

    def get_image(self, image_name: str) -> Image:
        return Image.fromarray(self.get_image_arr(image_name))

    def plot_image(self, image_name: str, figsize: list = [5, 5]) -> Image:
        img_arr = self.get_image_arr(image_name)
        img = Image.fromarray(img_arr)

        # prep title
        title = "{}, {}x{}, {:.2f}MB".format(image_name,
                                             img.size[0],
                                             img.size[1],
                                             sys.getsizeof(img_arr)/1024/1024)

        # render plot
        plt.figure(figsize=figsize)
        io.imshow(img_arr)
        plt.title(title)
        plt.show()

class Logger:
    """Class to provide basic functions for producing log files.
    """
    __loggers = {}
    __path = 'log.txt'

    @staticmethod
    def set_log_file_path(path: str) -> None:
        Logger.__path = path

    @staticmethod
    def get_logger(name: str) -> logging:
        # return instance if already exist
        if Logger.__loggers.get(name):
            return Logger.__loggers.get(name)

        # create a new isstance
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # build handler
        f_handler = logging.FileHandler(Logger.__path)
        formatter = logging.Formatter("[%(asctime)s] %(message)s",
                                    "%Y-%m-%d %H:%M:%S")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

        # store logger to static arr
        Logger.__loggers[name] = logger

        return logger

class StdLogger:
    """Class to output stdouts to both console and log file.
    """
    
    def __init__(self) -> None:
        self.terminal_original = None
        self.terminal = None

    def open(self, log_file: str, flush_log_file: bool = False) -> None:
        self.terminal_original = sys.stdout
        if flush_log_file:
            self.terminal = open(log_file, "w")
        else:
            self.terminal = open(log_file, "a")
        sys.stdout = self.terminal

    def close(self) -> None:
        self.terminal.close()
        sys.stdout = self.terminal_original

# Generator[YieldType, SendType, ReturnType]
from typing import Generator, List, Dict, Tuple
from itertools import cycle
import math

class MelDSManager:
    """Data manager for Kaggle's SIIM-ISIC Melanoma Classification competition. The
    class is designed to perform the following cases:
        - Load & parse csv.
        - Load & transform train/test images.
        - Allow users create mini batches with configurable params including:
            - Forcing a certain ratio of positive/negative examples in each batch.
            - Applying custom image random transformation.
            - Customizing the batch size.
            - Limiting the number of loops through dataset.
            - Shuffling the order of examples.
            - Peform k-fold and hold-out splits.
            - etc.
    """
    def __init__(self) -> None:
        self.__df_train = None # pd DataFrame
        self.__df_test = None # pd DataFrame

        self.__dir_train = ''
        self.__dir_test = ''
        self.__img_fmt = 'jpg'
        self.__img_tranf_func = None # Callable
        
        self.__resize_w = 1024

    @property
    def train_size(self) -> int:
        if self.__df_train is None:
            return 0
        return self.__df_train.shape[0]

    @property
    def train_df(self) -> pd.DataFrame:
        return self.__df_train

    @property
    def train_labels(self) -> np.ndarray:
        return self.__df_train.loc[:, "target"].to_numpy()

    @property
    def train_labels_one_hot(self) -> np.ndarray:
        return to_categorical(self.train_labels, num_classes=2)

    def get_train_img_arr(self, load_img: bool = False) -> np.ndarray:
        """This method is designed to be used on hold-outs that come with a reasonable 
        amount of examples (say, a few hundreds). Do not use this method if you need 
        to load a large amount of examples; use get_train_batch_generator() instead.
        """
        if not load_img:
            return self.__df_train.loc[:, 'image_name'].to_numpy()
        
        if self.train_size > 1000:
            raise TypeError("Error: Training dataset exceeds 1,000 examples...")
        
        # pull all images in one batch through generator
        gen = self.get_train_batch_generator(loop=1,
                                             load_img_arr=load_img,
                                             remove_404=False,
                                             print_404=True,
                                             tranform_img=False,
                                             one_hot_encoding=False)
        image_arr = next(gen)['images']
        images = []
        for img in image_arr:
            img = Image.fromarray(img)
            img = self.__resize_image(img)
            images.append(np.array(img))
        
        return np.array(images)
    
    @property
    def test_df(self) -> pd.DataFrame:
        return self.__df_test
    
    @property
    def test_labels(self) -> np.ndarray:
        return self.__df_test.loc[:, "target"].to_numpy()

    @property
    def test_labels_one_hot(self) -> np.ndarray:
        return to_categorical(self.__df_test, num_classes=2)
    
    @property
    def test_size(self) -> int:
        if self.__df_test is None:
            return 0
        return self.__df_test.shape[0]

    def enable_resize(self, w: int) -> None:
        self.__resize_w = w
    
    def set_img_transform_func(self, func: callable) -> None:
        self.__img_tranf_func = func

    def set_img_format(self, fmt: str = 'jpg') -> bool:
        """The str will be suffixed to the image names (e.g. from IP_7887363 to 
        IP_7887363.jpg ) in csv to load the actual images.
        the actual images.
        """
        self.__img_fmt = fmt

    def set_img_dirs(self, dir_train: str, dir_test: str) -> bool:
        """The paths will be prefixed to the image names (e.g. IP_7887363) in csv 
        to load the actual images.
        Returns
            bool: True is both dir_train and dir_test exist; false otherwise.
        """
        if not (os.path.isdir(dir_train) and os.path.isdir(dir_test)):
            return False

        self.__dir_train = dir_train
        self.__dir_test = dir_test
        return True

    def set_dfs(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        self.__df_train = train_df
        self.__df_test = test_df
    
    def load_csv(self, train_csv_path: str, 
                 test_csv_path: str) -> List[pd.DataFrame]:
        """Method to load training and test csv files.
        Returns:
            list: [0] DF for training. [1] DF for testing.
        """
        self.__df_train = pd.read_csv(train_csv_path)
        self.__df_test = pd.read_csv(test_csv_path)
    
    def hold_out(self, 
                 n_hold: int, 
                 n_positive: int,
                 shuffle: bool = True, 
                 cls: 'class constructor' = None) -> Dict[str, 'MelDSManager']:
        """Hold out a specified number of examples for testing. Note that due to the 
        limited number of positive examples included in the dataset, the positive examples
        in 'hold' but will NOT be removed from 'train'.

        Args:
            n_hold (int): Number of examples to hold out for testing.
            shuffle (bool): Set true to shuffle the dataset.
            n_positive (int): Number of positive examples to be included.
            cls (class constructor): Child classes can pass in its class constructor; 
                                     otherwise, the hold-outs will be initiated as MelDSManager.
        
        Returns:
            List[MelDSManager]: A dictionary with two keys, 'train' and 'hold', 
                                each is a MelDSManager instance.
        """
        if cls is None:
            cls = MelDSManager

        # pull df
        train_positive_df = self.__df_train.loc[self.__df_train['target'] == 1, :]
        train_negative_df = self.__df_train.loc[self.__df_train['target'] == 0, :]

        # shuffle if required
        if shuffle:
            train_positive_df = train_positive_df.sample(frac=1).reset_index(drop=True)
            train_negative_df = train_negative_df.sample(frac=1).reset_index(drop=True)

        # create hold df
        n_negative = n_hold - n_positive
        hold_positive_df = train_positive_df.iloc[:n_positive, :]
        hold_negitve_df = train_negative_df.iloc[:n_negative, :]
        hold_df = pd.concat([hold_positive_df, hold_negitve_df])
        hold_df = hold_df.sample(frac=1).reset_index(drop=True) # shuffle

        # create train df
        # all positive examples will be kept for training
        train_positive_df = train_positive_df.iloc[:, :]
        train_negative_df = train_negative_df.iloc[n_negative:, :]
        train_df = pd.concat([train_positive_df, train_negative_df])
        train_df = hold_df.sample(frac=1).reset_index(drop=True) # shuffle

        dict_ = {}

        # create hold batch, a MelDSManager or child class instance
        dsm = cls()
        dsm.set_img_dirs(self.__dir_train, self.__dir_test)
        dsm.set_img_format(self.__img_fmt)
        dsm.set_img_transform_func(self.__img_tranf_func)
        dsm.set_dfs(hold_df, self.__df_test.iloc[:, :])
        dsm.enable_resize(self.__resize_w)
        dict_['hold'] = dsm

        # create train batch, a MelDSManager or child class instance
        dsm = cls()
        dsm.set_img_dirs(self.__dir_train, self.__dir_test)
        dsm.set_img_format(self.__img_fmt)
        dsm.set_img_transform_func(self.__img_tranf_func)
        dsm.set_dfs(train_df, self.__df_test.iloc[:, :])
        dsm.enable_resize(self.__resize_w)
        dict_['train'] = dsm

        return dict_

    def k_fold_split(self, 
                     k: int, 
                     shuffle: bool = True, 
                     cls: 'class' = None) -> List['MelDSManager']:
        """Split training dataset into k folds.

        Args:
            k (int): Number of folds.
            shuffle (bool): Set true to shuffle the dataset.
            cls (class constructor): Child classes can pass in its class constructor; 
                                     otherwise, the hold-outs will be initiated as MelDSManager.

        Returns:
            List[MelDSManager]: A list of k MelDSManager instances, each comes with 
                                the same test but different training sets.
        """
        if cls is None:
            cls = MelDSManager

        # copy and shuffle train
        train_df = self.__df_train.iloc[:, :]
        if shuffle:
            train_df = train_df.sample(frac=1).reset_index(drop=True)
        
        # split training data
        fold_size = self.__df_train.shape[0] // k
        train_df_splits = []
        for i in range(k):
            # calculate start/end index
            i_start = i * fold_size
            i_end = i_start + fold_size
            if i_end > self.__df_train.shape[0]:
                i_end = self.__df_train.shape[0]

            # construct fold
            fold_i = self.__df_train.iloc[i_start:i_end, :]
            train_df_splits.append(fold_i)
        
        # build folds
        folds = []
        for df in train_df_splits:
            dsm = cls()
            dsm.set_img_dirs(self.__dir_train, self.__dir_test)
            dsm.set_img_format(self.__img_fmt)
            dsm.set_img_transform_func(self.__img_tranf_func)
            dsm.set_dfs(df, self.__df_test.iloc[:, :])
            dsm.enable_resize(self.__resize_w)
            
            folds.append(dsm)

        return folds
        
    def shuffle_train(self) -> None:
        """Shuffle the training dataset. 
        
        When calling get_train_batch_generator(), examples will always be read in 
        the order of the rows as they are stored in csv. Without shuffling, you 
        would always end up with the same cases in each of the batches.
        """
        self.__df_train = self.__df_train.sample(frac=1).reset_index(drop=True)
    
    def shuffle_test(self) -> None:
        """Shuffle the test dataset. 
        """
        self.__df_test = self.__df_test.sample(frac=1).reset_index(drop=True)
    
    def get_test_batch_generator(self, 
                                 batch_size: int = -1,
                                 max_batch_num: int = -1,
                                 load_img_arr: bool = False,
                                 flatten_img_arr: bool = False,
                                 remove_404: bool = False,
                                 print_404: bool = False,
                                 tranform_img: bool = True,
                                 one_hot_encoding: bool = False) -> Generator[dict, None, None]:
        # get the "real" batch size
        if batch_size <= 0:
            batch_size = self.__df_test.shape[0]

        # get generator
        yield from self.__loop_df(df=self.__df_test,
                                  loop=1, 
                                  batch_size=batch_size, 
                                  max_batch_num=max_batch_num,
                                  positive_case_pct=-1,
                                  load_img_arr=load_img_arr,
                                  flatten_img_arr=flatten_img_arr,
                                  remove_404=remove_404,
                                  print_404=print_404,
                                  tranform_img=tranform_img,
                                  one_hot_encoding=one_hot_encoding)
        
    def get_train_batch_generator(self, 
                                  loop: int = -1, 
                                  batch_size: int = -1,
                                  max_batch_num: int = -1,
                                  positive_case_pct: float = -1,
                                  load_img_arr: bool = False,
                                  flatten_img_arr: bool = False,
                                  remove_404: bool = False,
                                  print_404: bool = False,
                                  tranform_img: bool = True,
                                  one_hot_encoding: bool = False) -> Generator[dict, None, None]:
        """
        Args:
            load_img_arr (bool): Set true to load the JPEG and convert to np.ndarray.
            loop (int): Param to set how many loops returned generator will go 
                through the training dataset.

            batch_size (int): Size of the mini-batch; must be larger than 1; set 
                to -1 for the full batch.
            
            max_batch_num (int): Maximum # of batches the returned generator will 
                produce. Iteration will end if max_batch_num is reached before the 
                end of the loop (and vice versa). Set to -1 to remove the max limit.
            
            positive_case_pct (float): Force a percentage of the examples included 
                in the batch to be positive cases (i.e. target=1). Set to -1 to 
                remove the restriction. 
            flatten_img_arr (bool): Convert the 3d image array to 1d.
            remove_404 (bool): Set true to remove images that fail to load from the batch.
            print_404 (bool): Print out the paths to images that fail to load.
            tranform_img (bool): Apply random transformation to the images loaded.
        Returns:
            Generator[dict, None, None]: A dictionary of np.ndarray. Generator will 
                continue until end of looping.
                'images':   Array of JPEGs in np.ndarray; this field will be filled 
                            only if load_img_arr is set to True.
                'urls':     Array of image URLs; image_name column in csv, prefixed 
                            with the path to the image director.
                'labels':   Array of labels; target column in csv.
                'others':   Array of pd.Series that contains the rest of the columns 
                            in CSV.

                Note that the last batch will loop back to the beginning of the 
                training data set if the training size is not an exact multiple of 
                the batch size.
        """
        # get the "real" batch size
        if batch_size <= 0:
            batch_size = self.__df_train.shape[0]

        # get generator
        if positive_case_pct == -1:
            yield from self.__loop_df(df=self.__df_train,
                                      loop=loop, 
                                      batch_size=batch_size, 
                                      max_batch_num=max_batch_num,
                                      load_img_arr=load_img_arr,
                                      flatten_img_arr=flatten_img_arr,
                                      remove_404=remove_404,
                                      print_404=print_404,
                                      tranform_img=tranform_img,
                                      one_hot_encoding=one_hot_encoding)
        else:
            # split the +ve and -ve cases
            df_positive = self.__df_train.loc[self.__df_train['target'] == 1, :]
            df_negative = self.__df_train.loc[self.__df_train['target'] == 0, :]

            # input validation
            if positive_case_pct > 1:
                positive_case_pct = 1

            # calculate the number of positive & negative cases to be included 
            # in the batch
            n_positive = math.floor(batch_size * positive_case_pct)
            n_negative = batch_size - n_positive

            # get the generators
            # Note that gen_positive is set to infinite looping. Looping for 
            # gen_positivewill only stop when gen_negative stops.
            gen_negative = self.__loop_df(df=df_negative,
                                          loop=loop, 
                                          batch_size=n_negative, 
                                          max_batch_num=max_batch_num,
                                          load_img_arr=load_img_arr,
                                          flatten_img_arr=flatten_img_arr,
                                          remove_404=remove_404,
                                          print_404=print_404,
                                          tranform_img=tranform_img,
                                          one_hot_encoding=one_hot_encoding)
            gen_positive = self.__loop_df(df=df_positive,
                                          loop=-1, 
                                          batch_size=n_positive, 
                                          load_img_arr=load_img_arr,
                                          flatten_img_arr=flatten_img_arr,
                                          remove_404=remove_404,
                                          print_404=print_404,
                                          tranform_img=tranform_img,
                                          one_hot_encoding=one_hot_encoding)
            
            for examples_negative in gen_negative:
                examples_positive = next(gen_positive)

                images = np.append(examples_negative['images'],
                                   examples_positive['images'],
                                   axis=0)

                urls = np.append(examples_negative['urls'],
                                 examples_positive['urls'],
                                 axis=0)
                labels = np.append(examples_negative['labels'],
                                   examples_positive['labels'],
                                   axis=0)
                others = pd.concat([examples_negative['others'],
                                    examples_positive['others']]).reset_index(drop=True)

                yield {'images': images, 'urls': urls, 'labels': labels, 'others': others}

    
    def __loop_df(self, 
                  df: pd.DataFrame,
                  loop: int = -1, 
                  batch_size: int = -1,
                  max_batch_num: int = -1,
                  positive_case_pct: float = -1,
                  load_img_arr: bool = False,
                  flatten_img_arr: bool = False,
                  remove_404: bool = False,
                  print_404: bool = False, 
                  tranform_img: bool = True,
                  one_hot_encoding: bool = False) -> Generator[dict, None, None]:
        """Method to loop through the df given.
        """
        index = 0
        # If loop starts as a positive number, while will stop when loop drops to 0.
        # If loop starts as a negative number, while becomes infinite.
        while (loop != 0):
            if max_batch_num == 0:
                break;
            
            # extract appropriate rows from df to batch_df
            index_end = index + batch_size
            batch_df = None
            if index_end < df.shape[0]:
                batch_df = df.iloc[index: index_end, :]
                index = index_end
            else:
                # retrieve the indices for the rows to be included
                rows = []
                i_list = list(range(index, df.shape[0])) + list(range(0, index))
                n = batch_size
                for i in cycle(i_list):
                    rows += [i]
                    n -= 1
                    if n == 0:
                        break

                batch_df = df.iloc[rows, :]
                index = rows[len(rows) - 1]
                loop -= 1
            
            max_batch_num -= 1
            yield self.__parse_csv_df(df=batch_df,
                                      load_img_arr=load_img_arr, 
                                      flatten_img_arr=flatten_img_arr, 
                                      remove_404=remove_404,
                                      print_404=print_404,
                                      tranform_img=tranform_img,
                                      one_hot_encoding=one_hot_encoding)

    def __parse_csv_df(self, 
                       df: pd.DataFrame,
                       load_img_arr: bool = False,
                       flatten_img_arr: bool = False,
                       remove_404: bool = False,
                       print_404: bool = False, 
                       tranform_img: bool = True,
                       one_hot_encoding: bool = False) -> dict:
        """Method to parse the df created out of the csv.
        Returns: 
            dict: A dictionary of np.ndarray. Generator will continue until end of looping.

                'images':   Array of JPEGs in np.ndarray; this field will be filled 
                            only if load_img_arr is set to True.
                'urls':     Array of image URLs; image_name column in csv, prefixed 
                            with the path to the image director.
                'labels':   Array of labels; target column in csv.
                'others':   Array of pd.Series that contains the rest of the columns 
                            in CSV.
        """

        urls = self.__dir_train + df.loc[:, 'image_name'] + '.' + self.__img_fmt
        urls = urls.to_numpy()
        labels = df.loc[:, 'target'].to_numpy()
        others = df.loc[:, 
                        (df.columns != 'image_name') & 
                        (df.columns != 'target')].reset_index(drop=True)
            
        images = []
        load_success_arr = []
        if load_img_arr:
            for url in urls.tolist():
                try:
                    img = np.array(Image.open(url))
                    load_success_arr.append(True)
                except:
                    img = None
                    load_success_arr.append(False)
                    if print_404:
                        print('404:', url)
                images.append(img)

            # transform img if required
            # Note: images is a list at this point...
            if tranform_img:
                images = list(map(self.__img_tranf_func, images))
            images = np.array(images)

            if remove_404:
                images = images[load_success_arr]
                urls = urls[load_success_arr]
                labels = labels[load_success_arr]
                others = others.iloc[load_success_arr, :]
            
            if one_hot_encoding:
                labels = to_categorical(labels, num_classes=2)
                
        return {'images': images, 'urls': urls, 'labels': labels, 'others': others}

    __ANGLES= [0, 90, 180, 270]
    __FLIPS= [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    @property
    def default_img_transform_func(self) -> callable:
        """Return the default image transform function.
        """
        def transform(img_arr: np.ndarray) -> np.ndarray:
            if img_arr is None:
                return

            # config constants
            # PIL doc: https://tinyurl.com/yceqdcxj
            ROTATION = 5
            ZOOM_RATIO = [0.95, 1.05]
            V_FLIP = random.randint(0, 1)
            H_FLIP = random.randint(0, 1)

            # calculate transformations
            angle =  MelDSManager.__ANGLES[random.randint(0, len(MelDSManager.__ANGLES) - 1)]
            angle += random.randint(-ROTATION, ROTATION)
            flip = MelDSManager.__FLIPS[random.randint(0, len(MelDSManager.__FLIPS) - 1)]
            zoom_w = round(1024 * random.uniform(ZOOM_RATIO[0], ZOOM_RATIO[1]))
            left = top = (zoom_w - 1024)/2
            right = bottom = (zoom_w + 1024)/2

            img = Image.fromarray(img_arr)

            # rotate & flip
            img = img.rotate(angle)
            img = img.transpose(flip)  

            # zooming
            img = img.resize((zoom_w, zoom_w))
            img = img.crop((left, top, right, bottom))
            
            # resize
            img = self.__resize_image(img)

            return np.array(img)

        return transform
    
    def __resize_image(self, img:Image) -> Image:
        return img.resize((self.__resize_w, self.__resize_w))
            
class KerasDSManager(MelDSManager):
    def __init__(self) -> None:
        super().__init__()
        super().set_img_transform_func(super().default_img_transform_func)

        # track down the first and latest batches generated by self.get_cnn_tbg()
        self._last_batch = None
        self._first_batch = None

    @property
    def last_batch(self) -> tuple:
        """Return the last batch generated by self.get_cnn_tbg().
        
        Returns:
            tuple(MelDSManager, MelDSManager): A tuple with the first item being 
                an 'images' array and second item being a 'labels' array.

                Description of the items in tuple:                 
                [0], i.e. 'images': Array of JPEGs in np.ndarray; this field 
                                    will be filled only if load_img_arr is set 
                                    to True.
                [1], i.e. 'labels': Array of labels; target column in csv.
        """
        return self._last_batch

    @property
    def first_batch(self) -> tuple:
        """Return the first batch generated by self.get_cnn_tbg(). Refer to last_batch.
        """
        return self._first_batch
    
    def get_cnn_tbg(self, *args, **kwargs) -> Generator[tuple, None, None]:
        """TBG stands for "Training Batch Generator." This method is exactly the 
        same as MelDSManager.get_train_batch_generator(). The only difference is the 
        returned format.

        Returns:
            tuple(MelDSManager, MelDSManager): A tuple with the first item being 
                an 'images' array and second item being a 'labels' array.

                Description of the items in tuple:                 
                [0], i.e. 'images': Array of JPEGs in np.ndarray; this field 
                                    will be filled only if load_img_arr is set 
                                    to True.
                [1], i.e. 'labels': Array of labels; target column in csv.
        """
        gen = super().get_train_batch_generator(*args, **kwargs)
        for batch in gen:
            self._last_batch = (batch['images'], batch['labels'])
            if self._first_batch is None:
                self._first_batch = self._last_batch
            
            yield self._last_batch

    def hold_out(self, *args, **kwargs) -> Dict[str, 'MelDSManager']:
        return super().hold_out(cls=KerasDSManager, *args, **kwargs)

    def k_fold_split(self, *args, **kwargs) -> List['MelDSManager']:
        return super().k_fold_split(cls=KerasDSManager, *args, **kwargs)