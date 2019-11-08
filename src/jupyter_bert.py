
import torch
torch.manual_seed(58)
import pandas as pd

dataset = pd.read_csv('../data/task2_trainset.csv', dtype=str)
dataset.head()

dataset.drop('Title',axis=1,inplace=True)
dataset.drop('Categories',axis=1,inplace=True)
dataset.drop('Created Date',axis=1, inplace=True)
dataset.drop('Authors',axis=1,inplace=True)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)

trainset.to_csv('trainset.csv', index=False)
validset.to_csv('validset.csv', index=False)

dataset = pd.read_csv('../data/task2_public_testset.csv', dtype=str)
dataset.drop('Title',axis=1,inplace=True)
dataset.drop('Categories',axis=1,inplace=True)
dataset.drop('Created Date',axis=1, inplace=True)
dataset.drop('Authors',axis=1,inplace=True)
dataset.to_csv('testset.csv',index=False)

from transformers import BertTokenizer, BertForMaskedLM

PRETRAINED_MODEL_NAME = 'bert-base-cased'

NUM_LABLES = 4

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

vocab = tokenizer.vocab
print("字典大小：", len(vocab))

from tqdm import tqdm
from multiprocessing import Pool


def label_to_onehot(labels):
    """ Convert label to onehot .
        Args:
            labels (string): sentence's labels.
        Return:
            outputs (onehot list): sentence's onehot label.
    """
    label_dict = {'THEORETICAL': 0, 'ENGINEERING': 1, 'EMPIRICAL': 2, 'OTHERS': 3}
    onehot = [0, 0, 0, 0]
    for l in labels.split():
        onehot[label_dict[l]] = 1
    return onehot


def sentence_to_indices(sent, tokenizer):
    """ Convert sentence to its word indices.
    Args:
        sentence (str): One string.
    Return:
        indices (list of int): List of word indices.
    """
    return [tokenizer.convert_tokens_to_ids(word) for word in tokenizer.tokenize(sent)]


def get_dataset(data_path, tokenizer, n_workers=4):
    """ Load data and return dataset for training and validating.

    Args:
        data_path (str): Path to the data.
    """
    dataset = pd.read_csv(data_path, dtype=str)

    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(dataset) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(dataset)
            else:
                batch_end = (len(dataset) // n_workers) * (i + 1)

            batch = dataset[batch_start: batch_end]
            results[i] = pool.apply_async(preprocess_samples, args=(batch, tokenizer))

        pool.close()
        pool.join()

    processed = []
    for result in results:
        processed += result.get()
    return processed


def preprocess_samples(dataset, tokenizer):
    """ Worker function.

    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset)):
        processed.append(preprocess_sample(sample[1], tokenizer))

    return processed


def preprocess_sample(data, tokenizer):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    processed = {}
    processed['tokens'] = [sentence_to_indices(sent, tokenizer)
                           for sent in data['Abstract'].split('$$$')]
    processed['tokens'] = sum(processed['tokens'], [])
    processed['tokens'] = [tokenizer.convert_tokens_to_ids('[CLS]')] + processed['tokens'] + [
        tokenizer.convert_tokens_to_ids('[SEP]')]
    processed['segments'] = [0] * len(processed['tokens'])

    if 'Task 2' in data:
        processed['Label'] = label_to_onehot(data['Task 2'])

    return processed

print('[INFO] Start processing trainset...')
train = get_dataset('trainset.csv', tokenizer, n_workers=8)
print('[INFO] Start processing validset...')
valid = get_dataset('validset.csv', tokenizer, n_workers=8)
print('[INFO] Start processing testset...')
test = get_dataset('testset.csv', tokenizer, n_workers=8)

from torch.utils.data import Dataset
import torch


class BertDataset(Dataset):
    def __init__(self, data, max_len=512):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        # get max length in this batch
        max_len = max([min(len(data['tokens']), self.max_len) for data in datas])
        batch_tokens = []
        batch_segments = []
        batch_masks = []
        batch_label = []
        for data in datas:
            # padding abstract to make them in same length
            abstract_len = len(data['tokens'])
            if abstract_len > max_len:
                batch_tokens.append(data['tokens'][:max_len])
                batch_segments.append(data['segments'][:max_len])
                batch_masks.append([1] * max_len)
            else:
                batch_tokens.append(data['tokens'] + [0] * (max_len - abstract_len))
                batch_segments.append(data['segments'] + [0] * (max_len - abstract_len))
                batch_masks.append([1] * abstract_len + [0] * (max_len - abstract_len))
            # gather labels
            if 'Label' in data:
                batch_label.append(data['Label'])
        return torch.LongTensor(batch_tokens), torch.LongTensor(batch_segments), torch.LongTensor(
            batch_masks), torch.FloatTensor(batch_label)

trainData = BertDataset(train)
validData = BertDataset(valid)
testData = BertDataset(test)

from transformers import BertModel, BertPreTrainedModel


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=None, head_mask=None)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[1:]

        return outputs[0]

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

model = BertForMultiLabelSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels = 4)
# model.freeze_bert_encoder()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class F1():
    def __init__(self):
        self.threshold = 0.0
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        predicts = predicts > self.threshold
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth.type(torch.bool) * predicts).data.item()

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)

BATCH_SIZE = 2
EPOCHS = 6
import os


def _run_epoch(epoch, training):
    model.train(training)
    if training:
        description = 'Train'
        BATCH_SIZE = 2
        dataset = trainData
        shuffle = True
    else:
        description = 'Valid'
        BATCH_SIZE = 2
        dataset = validData
        shuffle = False
    dataloader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=shuffle,
                            collate_fn=dataset.collate_fn,
                            num_workers=4)

    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
    loss = 0.0
    f1_score = F1()
    for i, (tokens, segments, masks, labels) in trange:
        o_labels, batch_loss = _run_iter(tokens, segments, masks, labels)

        if training:
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

        loss += batch_loss.item()
        f1_score.update(o_labels.cpu(), labels)

        trange.set_postfix(
            loss=loss / (i + 1), f1=f1_score.print_score())
    if training:
        history['train'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})
    else:
        history['valid'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})


def _run_iter(tokens, segments, masks, labels):
    tokens = tokens.to(device)
    segments = segments.to(device)
    masks = masks.to(device)
    labels = labels.to(device)
    outputs = model(tokens, token_type_ids=segments, attention_mask=masks)
    l_loss = criteria(outputs, labels)
    return outputs, l_loss


def save(epoch):
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(model.state_dict(), 'model/model-1.pkl.' + str(epoch))
    with open('model/history-1.json', 'w') as f:
        json.dump(history, f, indent=4)
from torch.utils.data import DataLoader
from tqdm import trange
import json

opt = torch.optim.AdamW(model.parameters(), lr=1e-5, eps = 1e-8)

criteria = torch.nn.BCEWithLogitsLoss()
model.to(device)
history = {'train':[],'valid':[]}
for epoch in range(EPOCHS):
    print('Epoch: {}'.format(epoch))
    if epoch > 1:
        model.freeze_bert_encoder()
    _run_epoch(epoch, True)
    _run_epoch(epoch, False)
    save(epoch)
