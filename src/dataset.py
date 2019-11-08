from torch.utils.data import Dataset
import torch


class BertDataset(Dataset):
    def __init__(self, data, max_len, test):
        self.data = data
        self.max_len = max_len
        self.test = test

    def __len__(self):
        if self.test:
            return len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        reverse = False
        if index >= len(self.data):
            index -= len(self.data)
            reverse = True
        return {
            'reverse': reverse,
            'bert_data': self.data[index],
        }

    def collate_fn(self, datas):
        # get max length in this batch
        max_len = max([min(len(data['bert_data']['tokens']), self.max_len) for data in datas])
        batch_tokens = []
        batch_segments = []
        batch_masks = []
        batch_label = []
        for data in datas:
            abstract_len = len(data['bert_data']['tokens'])
            if not data['reverse']:
                # padding abstract to make them in same length
                if abstract_len > max_len:
                    batch_tokens.append(data['bert_data']['tokens'][:max_len])
                    batch_segments.append([0] * max_len)
                    batch_masks.append([1] * max_len)
                else:
                    batch_tokens.append(data['bert_data']['tokens'] + [0] * (max_len - abstract_len))
                    batch_segments.append([0] * max_len)
                    batch_masks.append([1] * abstract_len + [0] * (max_len - abstract_len))
            else:
                if abstract_len > max_len:
                    batch_tokens.append(data['bert_data']['tokens'][:-max_len-1:-1])
                    batch_segments.append([0] * max_len)
                    batch_masks.append([1] * max_len)
                else:
                    batch_tokens.append(data['bert_data']['tokens'][::-1] + [0] * (max_len - abstract_len))
                    batch_segments.append([0] * max_len)
                    batch_masks.append([1] * abstract_len + [0] * (max_len - abstract_len))
            # gather labels
            if 'Label' in data['bert_data']:
                batch_label.append(data['bert_data']['Label'])
        return torch.LongTensor(batch_tokens), torch.LongTensor(batch_segments), torch.LongTensor(
            batch_masks), torch.FloatTensor(batch_label)
