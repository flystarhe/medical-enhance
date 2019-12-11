import re
import torch
from torch.utils.data import DataLoader
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = 'batch must contain tensors, numbers, dicts or lists; found {}'


def default_collate(batch):
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ == 'ndarray':
        elem_type_ = batch[0].dtype
        if np_str_obj_array_pattern.search(elem_type_.str) is not None:
            raise TypeError(error_msg_fmt.format(elem_type_))

        return default_collate([torch.from_numpy(b) for b in batch])
    elif isinstance(batch[0], float):
        return torch.tensor(batch)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        if batch[0].get('cpu_only', False):
            return {key: [d[key] for d in batch] for key in batch[0]}
        else:
            return {key: default_collate([d[key] for d in batch]) for key in batch[0]}

    raise TypeError(error_msg_fmt.format(type(batch[0])))


def build_dataloader(dataset, imgs_per_gpu, workers_per_gpu, num_gpus=1, shuffle=True, **kwargs):
    batch_size = num_gpus * imgs_per_gpu
    num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=default_collate,
                             pin_memory=False,
                             **kwargs)

    return data_loader
