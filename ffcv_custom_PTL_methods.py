import os
from typing import Any

import torch
import clip
import numpy as np
from PIL import Image
from pytorch_lightning.utilities.fetching import AbstractDataFetcher
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, BytesField, NDArrayField


class FFCVCLDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, captions, max_len=2000):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.captions = captions
        self.max_len = max_len
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        labels = np.uint8(self.labels[i])
        text = self.captions[i][-self.max_len:]  # delete first instead of last characters as the last sentences usually contain the finding
        # pad
        padded_text = text + (" " * (self.max_len - len(text)))
        padded_text = np.frombuffer(padded_text.encode("ascii", errors="replace").replace(b"?", b" "), dtype='uint8')
        return img, labels, padded_text


class UintArrToText(torch.nn.Module):
    def forward(self, x):
        return x.tobytes().decode('ascii').strip()

    
class Tokenize(torch.nn.Module):
    def forward(self, x):
        return clip.tokenize(x, truncate=True)

    
def ffcv_convert_dataset(dataset, write_path):
    
    os.makedirs("/".join(write_path.split("/")[:-1]), exist_ok=True)

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(max_resolution=256, jpeg_quality=90),
        'labels': NDArrayField(np.dtype('uint8'), (len(dataset.labels[0]),)),
        'text': NDArrayField(np.dtype('uint8'), (dataset.max_len,)),
        #'tokens', NDArrayField(np.int32, (77,)),
    },
                          num_workers=0)

    # Write dataset
    writer.from_indexed_dataset(dataset)



def on_run_start(self, data_fetcher: AbstractDataFetcher, **kwargs: Any) -> None:
    self.trainer.logger_connector.on_epoch_start()
    self.trainer.call_hook("on_epoch_start")
    self.trainer.call_hook("on_train_epoch_start")
    self.trainer.fit_loop.epoch_progress.increment_started()
    
    self._dataloader_iter = enumerate(iter(data_fetcher.dataloader.loaders))

    
def advance(self, *args: Any, **kwargs: Any) -> None:
    """Runs a single training batch.

    Args:
        dataloader_iter: the iterator over the dataloader producing the new batch

    Raises:
        StopIteration: When the epoch is canceled by the user returning -1
    """
    if self.restarting and self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch):
        # skip training and run validation in `on_advance_end`
        return

    try:
        batch_idx, (batch) = next(self._dataloader_iter)
        self.batch_progress.is_last_batch = False
    except StopIteration:
        self.batch_progress.is_last_batch = True
        raise StopIteration

    if not self.trainer._data_connector.train_data_fetcher.store_on_device:
        with self.trainer.profiler.profile("training_batch_to_device"):
            batch = self.trainer.accelerator.batch_to_device(batch)

    self.batch_progress.increment_ready()

    self.trainer.logger_connector.on_batch_start(batch_idx, batch)

    if batch is None:
        self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
        batch_output = []
    else:
        # hook
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            self.batch_progress.increment_processed()
            raise StopIteration

        # TODO: Update this in v1.7 (deprecation: #9816)
        model_fx = self.trainer.lightning_module.on_train_batch_start
        extra_kwargs = (
            {"dataloader_idx": 0}
            if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
            else {}
        )

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, **extra_kwargs)
        if response == -1:
            self.batch_progress.increment_processed()
            raise StopIteration

        self.batch_progress.increment_started()

        with self.trainer.profiler.profile("run_training_batch"):
            batch_output = self.batch_loop.run(batch, batch_idx)

    self.batch_progress.increment_processed()

    # update non-plateau LR schedulers
    # update epoch-interval ones only when we are at the end of training epoch
    self.update_lr_schedulers("step", update_plateau_schedulers=False)
    if self._num_ready_batches_reached():
        self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

    batch_end_outputs = self._prepare_outputs_training_batch_end(
        batch_output,
        automatic=self.trainer.lightning_module.trainer.lightning_module.automatic_optimization,
        num_optimizers=len(self.trainer.optimizers),
    )

    # TODO: Update this in v1.7 (deprecation: #9816)
    model_fx = self.trainer.lightning_module.on_train_batch_end
    extra_kwargs = (
        {"dataloader_idx": 0}
        if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
        else {}
    )
    self.trainer.call_hook("on_train_batch_end", batch_end_outputs, batch, batch_idx, **extra_kwargs)
    self.trainer.call_hook("on_batch_end")
    self.trainer.logger_connector.on_batch_end()

    self.batch_progress.increment_completed()

    if is_overridden("training_epoch_end", self.trainer.lightning_module):
        self._outputs.append(batch_output)

    # -----------------------------------------
    # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
    # -----------------------------------------
    self.trainer.logger_connector.update_train_step_metrics()