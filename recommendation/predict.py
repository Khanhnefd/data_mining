import numpy as np
import pandas as pd
import torch
from util import set_device


device = set_device()

def _preddict(loader, model):
    model.eval()
    j = 1
    with torch.no_grad():
      for seq, target, lens in loader:
        seq = seq.to(device)
        target = target.to(device)
        outputs = model(seq, lens)
        logits = F.softmax(outputs, dim = 1)
        _, indices = torch.topk(logits, 20, -1)
        print('Is next clicked item in top 20 suggestions: ', (target in indices))
        print('Top 20 next item indices suggested: ')
    return indices