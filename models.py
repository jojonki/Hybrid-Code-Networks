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
        self.embedding = WordEmbedding(vocab_size, embd_size, pre_embd_w)
        self.lstm = nn.LSTM(307, hidden_size, batch_first=True) # TODO input dim
        self.fc = nn.Linear(hidden_size, action_size)

#     def forward(self, uttr, context, act_mask, bow, last_act):
    def forward(self, uttr, context):
        # uttr: (bs, dialog_len, sentence_len)
        # uttr: (bs, dialog_len, context_dim)
        bs = uttr.size(0)
        dlg_len = uttr.size(1)
        sent_len = uttr.size(2)

        embd = self.embedding(uttr.view(bs, -1)) # (bs, dialog_len*sentence_len, embd)
        embd = embd.view(bs, dlg_len, sent_len, -1) # (bs, dialog_len, sentence_len, embd)
        embd = torch.mean(embd, 2) # (bs, dialog_len, embd)
        x = torch.cat((embd, context), 2) # (bs, dialog_len, embd+context_dim)
        x, (h, c) = self.lstm(x) # (bs, seq, hid), ((1, bs, hid), (1, bs, hid))
        y = F.log_softmax(self.fc(x), -1) # (bs, seq, action_size)
        return y
