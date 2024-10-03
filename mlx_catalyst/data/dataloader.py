"""Data loader class for loading data in the MLX model trainer"""
import mlx.core as mx
import random
from typing import Optional
import mlx.data.datasets as dx
from mlx.data._c import Buffer

def get_mnist(batch_size, root=None):
    tr = load_mnist(root=root)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    
    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    test = load_mnist(root=root, train=False)
    test_iter = test.to_stream().key_transform("image", normalize).batch(batch_size)

    return tr_iter, test_iter

def normalize(x):
    x = x.astype("float32") / 255.0
    return (x - mean) / std

def DatasetLoader(
    dataset: Buffer,
    train: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[Iterable] = None,
    batch_sampler: Optional[Iterable] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = False,
    timeout: float = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    )->Buffer:

    # Apply shuffling if needed
    if shuffle:
        dataset = dataset.shuffle()
    
    dataset = dataset.to_stream()

    # Apply batching
    dataset = dataset.batch(batch_size)
    return dataset   
