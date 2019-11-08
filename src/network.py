import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
import ipdb


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
                            position_ids=None, head_mask=None)[1]
        # outputs = torch.sum(outputs[0], 1)

        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)

        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class SimpleNet(nn.Module):
    def __init__(self, embedder):
        super(SimpleNet, self).__init__()
        self.hidden_dim = 256
        self.sent_rnn = nn.GRU(embedder.get_dim(),
                               self.hidden_dim,
                               bidirectional=True,
                               batch_first=True)
        self.l1 = nn.Linear(self.hidden_dim, 4)

        self.embedding = nn.Embedding(embedder.get_vocabulary_size(), embedder.get_dim())
        self.embedding.weight = nn.Parameter(embedder.vectors)

        self.dropout = nn.Dropout(0.5)
        self.layerNorm = nn.LayerNorm([self.hidden_dim * 2])

    def forward(self, x):
        x = self.embedding(x)
        b, s, w, e = x.shape
        x = x.view(b, s*w, e)
        x, __ = self.sent_rnn(x)
        # x = self.layerNorm(x)
        x = self.dropout(x)
        x = x.view(b, s, w, -1)
        x = torch.max(x, dim=2)[0]
        x = x[:, :, :self.hidden_dim] + x[:, :, self.hidden_dim:]
        x = torch.max(x, dim=1)[0]
        x = self.l1(F.relu(x))
        x = torch.sigmoid(x)
        return x
