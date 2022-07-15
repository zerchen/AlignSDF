#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :loader.py
#@Date        :2021/10/16 21:06:28
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch


class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input_iter, next_label_iter, next_meta_iter in self.loader:
            with torch.cuda.stream(stream):
                for key in next_input_iter:
                    next_input_iter[key] = next_input_iter[key].cuda(non_blocking=True)

                for key in next_label_iter:
                    next_label_iter[key] = next_label_iter[key].cuda(non_blocking=True)

                for key in next_meta_iter:
                    if isinstance(next_meta_iter[key], torch.Tensor):
                        next_meta_iter[key] = next_meta_iter[key].cuda(non_blocking=True)

            if not first:
                yield input_iter, label_iter, meta_iter
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input_iter = next_input_iter
            label_iter = next_label_iter
            meta_iter = next_meta_iter

        yield input_iter, label_iter, meta_iter
    
    def __len__(self):
        return len(self.loader)
    
    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(dataset, batch_size, num_workers, is_training=False, use_prefetcher=True, pin_memory=True, distributed=False, persistent_workers=False, collate_fn=None):
    if distributed and is_training:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    
    loader_default = torch.utils.data.DataLoader

    loader_args = dict(batch_size=batch_size, shuffle=(sampler is None and is_training), num_workers=num_workers, sampler=sampler, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=is_training, persistent_workers=persistent_workers)

    loader = loader_default(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(loader)
    
    return loader