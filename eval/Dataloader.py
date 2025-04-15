import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
class CustomDataLoader:
    """
    A custom DataLoader that works with the provided Dataset class.
    Supports batching, shuffling, and multi-threaded data loading.
    """

    def __init__(
        self,
        dataset,
        split_name,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False
    ):
        """
        Initializes the CustomDataLoader.

        Args:
            dataset (Dataset): An instance of the custom Dataset class.
            split_name (str): The name of the data split to load (e.g., 'train', 'test').
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at every epoch.
            num_workers (int): Number of threads to use for data loading.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        self.dataset = dataset
        self.split_name = split_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

        # Get the total number of samples
        self.n_samples = self.dataset.get_feature(
            split_name=self.split_name,
            feature_name='<default_input>'
        ).shape[0]

        # Calculate the number of batches
        self.num_batches = math.ceil(self.n_samples / self.batch_size)
        if self.drop_last and self.n_samples % self.batch_size != 0:
            self.num_batches = math.floor(self.n_samples / self.batch_size)

        # Prepare the indices
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """
        Yields batches of data.
        """
        # Shuffle indices at the start of each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Generate batch indices
        batch_indices = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            if end > self.n_samples:
                if self.drop_last:
                    break
                end = self.n_samples
            batch = self.indices[start:end]
            batch_indices.append(batch)

        # Use ThreadPoolExecutor for multi-threaded data loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batch loading tasks
            futures = {
                executor.submit(self.load_batch, batch_idx): batch_idx
                for batch_idx in batch_indices
            }

            # Yield batches as they are completed
            for future in as_completed(futures):
                batch = future.result()
                yield batch

    def load_batch(self, batch_indices):
        """
        Loads a single batch of data.

        Args:
            batch_indices (np.ndarray): Array of indices for the batch.

        Returns:
            tuple: (inputs, targets) for the batch.
        """
        inputs = self.dataset.get_feature(
            split_name=self.split_name,
            feature_name='<default_input>',
            indices=batch_indices.tolist()
        )
        targets = self.dataset.get_feature(
            split_name=self.split_name,
            feature_name='<default_output>',
            indices=batch_indices.tolist()
        )
        return torch.Tensor(inputs), torch.Tensor(targets)

