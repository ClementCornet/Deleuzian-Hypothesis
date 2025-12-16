from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from PIL import Image
from enum import Enum
import pandas as pd
from rich import print
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class Modalities(Enum):
    IMAGE = 0
    TEXT = 1
    AUDIO = 2

def get_dataset(dataset: str, split='train', transform=None, **kwargs):
    if dataset == 'ImNet': return ImNetDataset(split, transform, **kwargs), Modalities.IMAGE
    if dataset == 'WikiArt': return WikiArt(split, transform, **kwargs), Modalities.IMAGE
    if dataset == 'IMDB': return IMDB(split, **kwargs), Modalities.TEXT
    if dataset == 'AudioSet': return Audioset(split, transform, **kwargs), Modalities.AUDIO

class ImNetDataset(torch.utils.data.Dataset):
    """Subset of 50k+5k images from ImageNet-1k, 100 classes"""
    def __init__(self, split, transform, **kwargs):
        super().__init__()
        url = 'timm/mini-imagenet'
        self.ds = load_dataset(url)[split] # Has both training and testing splits
        self.labels = pd.DataFrame(pd.Series(self.ds['label'], name='Class'))
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index):
        im = self.ds[index]['image']
        if self.transform is not None:
            try:
                im = self.transform(im)
            except:
                im = self.transform(Image.new('RGB', (400,400)))
        
        return im
    

class WikiArt(torch.utils.data.Dataset):
    """80k painting, hald used for train and test. 3 attributes per painting : artist, style and genre"""
    def __init__(self, split, transform, **kwargs):
        super().__init__()
        url = 'huggan/wikiart'
        self.ds = load_dataset(url)['train']
        self.split = split
        self.transform = transform

        columns = self.ds.column_names
        columns.remove('image')

        self.labels = pd.DataFrame()
        for label_type in columns:
            labels_mapping = self.ds.features[label_type].names
            self.labels[label_type] = [labels_mapping[i] for i in self.ds[label_type]]

        self.offset = 0 if split=='train' else 1

        self.labels = self.labels[self.offset::2]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        im = self.ds[self.offset + 2*index]['image']
        if self.transform is not None:
            try:
                im = self.transform(im)
            except:
                print("Detaults to a blank image")
                im = self.transform(Image.new('RGB', (400,400)))
        #print(im)
        return im
    

        
class IMDB(torch.utils.data.Dataset):
    """Binary classification dataset on film reviews"""
    def __init__(self, split, **kwargs):
        super().__init__()
        url = "stanfordnlp/imdb"
        self.ds = load_dataset(url)[split] # Has both train/test

        self.labels = pd.DataFrame(
            pd.Series(
                [['neg', 'pos'][i] for i in self.ds['label']],
                name='label'
            )
        )

    def __len__(self): return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]['text']

class Audioset(Dataset):
    """Multi-Classification Dataset : 527 binary attributes, one per class"""
    def __init__(self, split='train', transform=None, **kwargs):
        super().__init__()
        self.ds = load_dataset('agkphysics/AudioSet', name="balanced", trust_remote_code=True)[split]
        self.transform = transform
        alllabels = self.ds['human_labels'] #[d['human_labels'] for d in self.ds]
        labels_l = []
        for ll in alllabels:
            for lll in ll: labels_l.append(lll)

        mlb = MultiLabelBinarizer()
        self.labels = pd.DataFrame(mlb.fit_transform(alllabels), columns=mlb.classes_)
        

    def __len__(self): return len(self.ds)

    def __getitem__(self, index):
        try:
            return torch.tensor(self.transform(self.ds[index]['audio']['array'], sampling_rate=16000)['input_values'][0])
        except:
            return torch.tensor(self.transform(torch.zeros((1000,)), sampling_rate=16000)['input_values'][0])


if __name__ == '__main__':
    imnet_train = ImNetDataset(split='train', transform=None)
    print(imnet_train.labels.head())