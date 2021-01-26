import torch


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)

def get_dataloaders(opt, distributed_data_parallel = False):
    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders."+dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    if distributed_data_parallel:
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opt.batch_size//opt.num_gpus,
            sampler=data_sampler(dataset_train, shuffle=True, distributed=opt.distributed),
            drop_last=True,
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=opt.batch_size//opt.num_gpus,
            sampler=data_sampler(dataset_val, shuffle=True, distributed=opt.distributed),
            drop_last=True,
        )
    else:
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False)

    return dataloader_train, dataloader_val
