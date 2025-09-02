import os
import torch
from torch.utils.data.dataset import Dataset


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        word = word.lower()
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
        self._build_vocab(DATA_DIR, filenames)

    def _build_vocab(self, DATA_DIR, filenames):
        for filename in filenames:
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word.lower())


class TxtDatasetProcessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):
        self.txt_path = os.path.join(data_path, txt_path)

        # 加载文件名
        with open(os.path.join(data_path, txt_filename), 'r', encoding='utf-8') as f:
            self.txt_filename = [x.strip() for x in f]

        # 加载标签
        with open(os.path.join(data_path, label_filename), 'r', encoding='utf-8') as f:
            self.label = [int(x.strip()) for x in f]

        self.sen_len = sen_len
        self.corpus = corpus

    def __getitem__(self, index):
        filename = os.path.join(self.txt_path, self.txt_filename[index])
        txt_tensor = torch.zeros(self.sen_len, dtype=torch.long)

        count = 0
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    word = word.lower()
                    if word in self.corpus.dictionary.word2idx:
                        txt_tensor[count] = self.corpus.dictionary.word2idx[word]
                        count += 1
                        if count >= self.sen_len:
                            break
                if count >= self.sen_len:
                    break

        label_tensor = torch.tensor(self.label[index], dtype=torch.long)
        return txt_tensor, label_tensor

    def __len__(self):
        return len(self.txt_filename)
