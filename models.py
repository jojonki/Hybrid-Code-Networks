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
    def __init__(self, vocab_size, embd_size, hidden_size, action_size, pre_embd_w=None):
        super(HybridCodeNetwork, self).__init__()
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        # lstm_in_dim = embd_size + vocab_size + action_size + 4 # 4 (context size)
        # lstm_in_dim = embd_size  # 4 (context size)
        lstm_in_dim = vocab_size# 4 (context size)
        self.embedding = WordEmbedding(vocab_size, embd_size, pre_embd_w)
        self.lstm = nn.LSTM(lstm_in_dim, hidden_size, batch_first=True)
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

        embd = self.embedding(uttr.view(bs, -1)) # (bs, dialog_len*sentence_len, embd)
        embd = embd.view(bs, dlg_len, sent_len, -1) # (bs, dialog_len, sentence_len, embd)
        embd = torch.mean(embd, 2) # (bs, dialog_len, embd)
        # x = torch.cat((embd, context, bow, prev), 2) # (bs, dialog_len, embd+context_dim)
        # x = torch.cat((embd), 2) # (bs, dialog_len, embd+context_dim)
        # x = embd # (bs, dialog_len, embd+context_dim)
        x = bow # (bs, dialog_len, embd+context_dim)
        x, (h, c) = self.lstm(x) # (bs, dialog_len, hid), ((1, bs, hid), (1, bs, hid))
        y = self.fc(F.tanh(x)) # (bs, dialog_len, action_size)
        y = F.softmax(y, -1) # (bs, dialog_len, action_size)
        # y = y * act_filter
        return y
