import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import re
import numpy as np
from PIL import Image
from build_vocab import parse_code

#    data_loader = get_loader(args.image_dir,  vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers) 
def make_dataset(dir):
    folders = []
    for root, dirs, files in os.walk(os.path.abspath(dir)):
        for file in files:
            if file.endswith(".txt"):
                folders.append(root)
    return folders

def validation_split(dataset, val_share=0.1):
    val_offset = int(len(dataset)*(1-val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset)-val_offset)
class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[i+self.offset]

class ProcessingDataset(data.Dataset):
    """Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root,  vocab, transform=None,length=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.folders = make_dataset(root)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        path = self.folders[index]
        vocab = self.vocab
        with open(os.path.join(path,"code.txt"), 'r') as f:
            code = str(f.read())
        image = Image.open(os.path.join(path, "image.tiff")).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = parse_code(code)
        code = []
        code.append(vocab('<start>'))
        code.extend([vocab(token) for token in tokens])
        code.append(vocab('<end>'))
        target = torch.Tensor(code)
        return image, target

    def __len__(self):
        return len(self.folders)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging code (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by code length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge code (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def get_loader(root,  vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom processing dataset."""
    processing = ProcessingDataset(root=root,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for processing dataset
    # This will return (images, code, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=processing,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
