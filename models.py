import torch.nn as nn
import torch
import torch.nn.functional as F


class WordEmbedding(nn.Module):
    '''
    In : (N, sentence_len)
    Out: (N, sentence_len, embd_size)
    '''
    def __init__(self, vocab_size, embd_size, pre_embd_w=None, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        if pre_embd_w is not None:
            print('pre embedding weight is set')
            self.embedding.weight = nn.Parameter(pre_embd_w, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)


class HybridCodeNetwork(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, action_size, pre_embd_w=None, use_ctx=False, use_embd=False, use_prev=False, use_mask=False):
        super(HybridCodeNetwork, self).__init__()
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.embedding = WordEmbedding(vocab_size, embd_size, pre_embd_w)

        lstm_in_dim = vocab_size # default is bow vector
        if use_ctx: # context feature
            lstm_in_dim += 4 # n of context
        if use_embd: # embedding size
            lstm_in_dim += embd_size
        if use_prev: # previous action
            lstm_in_dim += action_size
        if use_mask: # action filter
            lstm_in_dim += action_size
        self.lstm = nn.LSTM(lstm_in_dim, hidden_size, batch_first=True)
        self.use_ctx = use_ctx
        self.use_embd = use_embd
        self.use_prev = use_prev
        self.use_mask = use_mask

        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, uttr, context, bow, prev, act_filter):
        # uttr       : (bs, dialog_len, sentence_len)
        # context    : (bs, dialog_len, context_dim)
        # bow        : (bs, dialog_len, vocab_size)
        # prev       : (bs, dialog_len, action_size)
        # act_filter : (bs, dialog_len, action_size)
        bs = uttr.size(0)
        dlg_len = uttr.size(1)
        sent_len = uttr.size(2)

        x = bow # (bs, dialog_len, embd+context_dim)
        if self.use_ctx:
            x = torch.cat((x, context), 2)
        if self.use_embd:
            embd = self.embedding(uttr.view(bs, -1)) # (bs, dialog_len*sentence_len, embd)
            embd = embd.view(bs, dlg_len, sent_len, -1) # (bs, dialog_len, sentence_len, embd)
            embd = torch.mean(embd, 2) # (bs, dialog_len, embd)
            x = torch.cat((x, embd), 2)
        if self.use_prev:
            x = torch.cat((x, prev), 2)
        if self.use_mask:
            x = torch.cat((x, act_filter), 2)
        x, (h, c) = self.lstm(x) # (bs, dialog_len, hid), ((1, bs, hid), (1, bs, hid))
        y = self.fc(F.tanh(x)) # (bs, dialog_len, action_size)
        y = F.softmax(y, -1) # (bs, dialog_len, action_size)
        if self.use_mask:
            y = y * act_filter
        return y
