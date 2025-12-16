import utils
import os
from rich import print
import wrappedmodels
import torch
from tqdm import tqdm
import data

def record(
    d: dict
):
    
    """Record activations from a model on a dataset specified in the config passed has a parameter"""

    model = d['model']
    model_size = d['model_size']
    layer = d['layer']
    batch_size = d['batch_size']

    cache_name = utils.sanitize_cache_name(d)
    
    if not os.path.isdir('activations'): os.mkdir('activations')
    os.makedirs(cache_name, exist_ok=True)

    wm = getattr(wrappedmodels, model)(layers=[layer], model_size=model_size).to('cuda' if torch.cuda.is_available() else 'cpu')

    ds_train, mod_train = data.get_dataset(**d, split='train', transform=wm.transform)
    
    dl_train = torch.utils.data.DataLoader(
        dataset=ds_train, batch_size=batch_size, drop_last=False, num_workers=0, persistent_workers=False, shuffle=False, pin_memory=False
    )

    ds_test, mod_test = data.get_dataset(**d, split='test', transform=wm.transform)
    dl_test = torch.utils.data.DataLoader(
        dataset=ds_test, batch_size=batch_size, drop_last=False, num_workers=0, persistent_workers=False, shuffle=False, pin_memory=False
    )
    A_train = torch.zeros((len(ds_train), wm.d_vit(layer)))
    _start_idx = 0
    for batch in tqdm(dl_train, total=len(dl_train), desc='Recording Train Activations'):
        if mod_train == data.Modalities.TEXT: batch = wm.transform(batch)
        try:
            wm(batch.to('cuda' if torch.cuda.is_available() else 'cpu'))
        except: continue
        _end_idx = min(_start_idx + batch_size, A_train.shape[0])
        try:
            A_train[_start_idx:_end_idx] = wm.activations[layer][:,0,:] # Index 0 for first patch = CLS
        except: print('Batch Failed') # Should not happen
        _start_idx += batch_size
        del batch
    torch.save(A_train, f'{cache_name}/train.pt')
    del A_train

    A_test = torch.zeros((len(ds_test), wm.d_vit(layer)))
    _start_idx = 0
    for batch in tqdm(dl_test, total=len(dl_test), desc='Recording Test Activations'):
        if mod_test == data.Modalities.TEXT: batch = wm.transform(batch)
        try:
            wm(batch.to('cuda' if torch.cuda.is_available() else 'cpu'))
        except: continue
        _end_idx = min(_start_idx + batch_size, A_test.shape[0])
        try:
            A_test[_start_idx:_end_idx] = wm.activations[layer][:,0,:] # Index 0 for first patch = CLS
        except: print('Batch failed')
        _start_idx += batch_size
        del batch
    torch.save(A_test, f'{cache_name}/test.pt')
    del A_test