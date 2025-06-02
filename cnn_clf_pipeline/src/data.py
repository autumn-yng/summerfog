import tensorflow as tf
from typing import List, Tuple

def load_dataset(data_dir: str,
                 class_names: List[str],
                 batch_size: int,
                 image_size: Tuple[int, int] = (256, 256),
                 shuffle: bool = True,
                 seed: int = 42) -> tf.data.Dataset:
    """
    Loads images from directory and returns a batched dataset. 
    
    I used a Keras utility, ```image_dataset_from_directory()```, to read in and pre-process the data. This utility generates a ```tf.data.Dataset```, from image files in the directory we set the path for above. This Dataset allows us to **apply pre-processing** steps (like normalizing pixel values, resizing images, etc.) to the dataset **batch by batch**.
    """
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=class_names,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed
    )


def scale_dataset(dataset: tf.data.Dataset, factor: float = 255.0) -> tf.data.Dataset:
    """
    Scales pixel values in dataset by a given factor (e.g., 255.0 to normalize to [0,1]).
    """
    return dataset.map(lambda x, y: (x / factor, y))


def split_dataset(dataset: tf.data.Dataset,
                  train_frac: float = 0.7,
                  val_frac: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Splits a tf.data.Dataset into train, validation, and test subsets.
    Assumes dataset is already shuffled.
    """
    total_size = len(dataset)
    train_size = int(total_size * train_frac)
    val_size = int(total_size * val_frac)
    test_size = total_size - train_size - val_size

	# Take the number of train batches among the total number of batches to be the training set
    train_ds = dataset.take(train_size) 
    # Skip the first number of batches (that have already been in the training set) and take the next batches for validation set.
    val_ds = dataset.skip(train_size).take(val_size) 
    # Skip the batches already taken for train & validation, and take the next batches for the test set.
    test_ds = dataset.skip(train_size + val_size).take(test_size) 

    return train_ds, val_ds, test_ds