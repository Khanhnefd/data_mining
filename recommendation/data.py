import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Các token default
PAD_token = 0  # token padding cho câu ngắn ==> index = 0

# tạo Vocabulary for index input

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.item2index = {}
        self.item2count = {}
        self.index2item = {PAD_token: "PAD"}
        self.num_items = 1  # số lượng mặc định ban đầu là 1 ứng với PAD_token

    def addSenquence(self, data):
        for sequence in data:
          for item in sequence:
              self.addItem(item)

    # Thêm một item vào hệ thống
    def addItem(self, item):
        if item not in self.item2index:
            self.item2index[item] = self.num_items
            self.item2count[item] = 1
            self.index2item[self.num_items] = item
            self.num_items += 1
        else:
            self.item2count[item] += 1

    # Loại các item dưới ngưỡng xuất hiện min_count
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_items = []

        for k, v in self.item2count.items():
            if v >= min_count:
                keep_items.append(k)

        print('keep_items {} / {} = {:.4f}'.format(
            len(keep_items), len(self.item2index), len(keep_items) / len(self.item2index)
        ))

        # Khởi tạo lại từ điển
        self.item2index = {}
        self.item2count = {}
        self.index2item = {PAD_token: "PAD"}
        self.num_items = 1

        # Thêm các items vào từ điển
        for item in keep_items:
            self.addItem(item)

    # Hàm convert sequence về chuỗi các indices
    def _seqItem2seqIndex(self, x):
        return [self.item2index[item] if item in self.item2index else 0 for item in x]

class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """
    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)

    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])