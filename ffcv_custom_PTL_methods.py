import os
from typing import Any, List

import torch
import torchvision.transforms as TF
import clip
from clip.simple_tokenizer import SimpleTokenizer
import numpy as np
import numba
from PIL import Image
from pytorch_lightning.utilities.fetching import AbstractDataFetcher
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, BytesField, NDArrayField
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from dataclasses import replace


class SubsampleTokens(Operation):
    def __init__(self, shuffle, context_length=77):
        self.shuffle = shuffle
        self.context_length = context_length
        
        self.dot_token = 126
        self.end_token = 49407
        
    # Return the code to run this operation
    def generate_code(self):
        #@numba.njit()
        def sample_tokens_until_size(tokens: np.ndarray) -> np.ndarray:
            end_idx = np.where(tokens == 49407)[0][0]
            real_tokens = tokens[1: end_idx]
            # find where we have a dot==end of sentence
            dot_idcs = list(np.where(real_tokens == 269)[0])

            # divide tokens into list of token sentences
            #real_tokens = list(real_tokens)

            last = 0
            sents = []
            for i in dot_idcs:
                i = i + 1
                sents.append(real_tokens[last:i].copy())
                last = i

            # sample random sentences until we have enough tokens or no more sentences left
            possible_idcs = list(range(len(sents)))

            token_count = 0
            sent_idcs = []
            while token_count < 75 and len(possible_idcs) > 0:
                rand_idx = np.random.randint(0, len(possible_idcs))
                rand_sent_idx = possible_idcs[rand_idx]
                sent_idcs.append(rand_sent_idx)
                token_count += len(sents[rand_sent_idx])
                possible_idcs.remove(rand_sent_idx)

            # assemble sentences
            final_tokens = np.zeros((token_count,), dtype=np.int32)
            count = 0
            for i in sorted(sent_idcs):
                sent = sents[i]
                final_tokens[count:count + len(sent)] = sent
                count += len(sent)
                #final_tokens.extend(sents[i])

            # add start and end token
            shortened_real_tokens = final_tokens[:77 - 2]
            # store in empty array
            final_tokens = np.zeros((77,), dtype=np.int32)
            final_tokens[0] = 49406 #tokens[0]
            final_tokens[1:1 + len(shortened_real_tokens)] = shortened_real_tokens
            final_tokens[1 + len(shortened_real_tokens)] = 49407
            return final_tokens
        
        if self.shuffle:
            #@numba.njit()
            def subsample(tokens, dst):
                tokens = tokens.reshape(-1)
                #end_idx = np.where(tokens == 49407)[0][0]
                #real_tokens = tokens[1: end_idx]
                shortened = sample_tokens_until_size(tokens)#, end_idx=end_idx)
                return shortened
        else:
            def subsample(tokens, dst):
                tokens = tokens.reshape(-1)
                end_idx = np.where(tokens == 49407)[0][0]
                real_tokens = tokens[1: end_idx][:77 - 2]
                
                shortened = np.zeros((77,), dtype=np.int32)
                shortened[0] = 49406
                shortened[1: 1 + len(real_tokens)] = real_tokens
                shortened[1 + len(real_tokens)] = 49407
                
                return shortened
            
        return subsample

    def declare_state_and_memory(self, previous_state):
        new_state = replace(previous_state, shape=(77,), dtype=np.int32, jit_mode=True)
        mem_allocation = AllocationQuery((77,), np.int32)
        return (new_state, mem_allocation)


class UintArrToToken(Operation):
    def __init__(self, sent_frac):
        self.sent_frac = sent_frac
        
    # Return the code to run this operation
    def generate_code(self):
        tokenize = Tokenizer()
        
        def uintarr_to_text(x):
            return x.tobytes().decode('ascii').strip()
        
        if self.sent_frac > 0:
            subsample = SubsampleSentsNp(self.sent_frac)
            def uint_to_token(x, dst):
                text = uintarr_to_text(x)
                text = subsample(text)
                tokens = tokenize(text)
                return tokens
        else:
            def uint_to_token(x, dst):
                text = uintarr_to_text(x)
                tokens = tokenize(text)
                return tokens
            
        return uint_to_token

    def declare_state_and_memory(self, previous_state):
        new_state = replace(previous_state, shape=(77,), dtype=np.int32, jit_mode=False)
        mem_allocation = AllocationQuery((77,), np.int32)
        return (new_state, mem_allocation)

    
class SubsampleSentsNp:
    def __init__(self, frac, min_sent_len=5):
        super().__init__()
        self.frac = frac
        self.min_sent_len = min_sent_len
        
    def __call__(self, x):
        sents = [sent.strip() for sent in x.split(".") if len(sent) > self.min_sent_len]
        num_sents_used = round(len(sents) * self.frac)
        sents_used = np.random.choice(sents, num_sents_used, replace=False)
        text = ". ".join(sents_used)
        return text

    
class Tokenizer:#(torch.nn.Module):
    def __init__(self, truncate=True, context_length=77):
        self.tokenizer = SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self.tokenizer.encoder["<|endoftext|>"]
        self.truncate = truncate
        self.context_length = context_length

    def __call__(self, x):
        return self.tokenize_np(x, truncate=self.truncate, context_length=self.context_length)
    
    def tokenize_np(self, text, dst=None, context_length=77, truncate=True):
        if truncate:
            if dst is None:
                dst = np.empty((context_length, ), dtype=np.int32)
                #tokens = np.array([self.sot_token] + self.tokenizer.encode(text)[:context_length - 2] + [self.eot_token], dtype=np.int32)
                #return tokens
            #else:
            dst[0] = self.sot_token
            encoded = self.tokenizer.encode(text)[:context_length - 2]
            dst[1:1 + len(encoded)] = encoded
            dst[1 + len(encoded)] = self.eot_token
            dst[2 + len(encoded):] = 0
            return dst
        else:
            tokens = np.array([self.sot_token] + self.tokenizer.encode(text) + [self.eot_token], dtype=np.int32)
            return tokens


class UintArrToTextSlow:#(torch.nn.Module):
    def __call__(self, x):
        return x.tobytes().decode('ascii').strip()


class FFCVCLDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, captions, max_len=2000, max_tokens=400, max_res=256):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.captions = captions
        self.max_len = max_len
        self.max_tokens = max_tokens
        self.tokenizer = Tokenizer(truncate=True, context_length=max_tokens)
        self.max_res = max_res
        
        self.center_crop = TF.CenterCrop((max_res, max_res))
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        pil_img = Image.open(self.paths[i]).convert("RGB")
        pil_img = self.center_crop(pil_img)
        img = np.uint8(np.array(pil_img))
        labels = np.uint8(self.labels[i])
        text = self.captions[i][-self.max_len:]  # delete first instead of last characters as the last sentences usually contain the finding
        # pad
        #padded_text = text + (" " * (self.max_len - len(text)))
        #padded_text = np.frombuffer(padded_text.encode("ascii", errors="replace").replace(b"?", b" "), dtype='uint8')
        # tokenize to len max_token
        padded_tokens = self.tokenizer(text).astype(np.int32)
        return img, labels, padded_tokens

    
def ffcv_convert_dataset(dataset, write_path):
    os.makedirs("/".join(write_path.split("/")[:-1]), exist_ok=True)

    # Pass a type for each data field
    writer = DatasetWriter(write_path, 
                           {
                               # Tune options to optimize dataset size, throughput at train-time
                               'image': RGBImageField(max_resolution=256, jpeg_quality=90),
                               'labels': NDArrayField(np.dtype('uint8'), (len(dataset.labels[0]),)),
                               #'text': NDArrayField(np.dtype('uint8'), (dataset.max_len,)),
                               'tokens': NDArrayField(np.dtype('int32'), (dataset.max_tokens,)),
                            }, num_workers=8)
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