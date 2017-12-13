import random
import copy
from gensim.models.keyedvectors import KeyedVectors
import torch
import torch.nn.functional as F

from utils import get_entities, load_data, make_word_vector, to_var
from models import HybridCodeNetwork

# model_path = './data/GoogleNews-vectors-negative300.bin'
# word2vec = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
# model.wv['computer']

entities = get_entities('dialog-bAbI-tasks/dialog-babi-kb-all.txt')

for idx, (ent_name, ent_vals) in enumerate(entities.items()):
    print(idx, ent_name, ent_vals[0] )

# create training dataset
SILENT = '<SILENT>'
system_acts = [SILENT]
vocab = []
fpath_train = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt'
fpath_test = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-tst-OOV.txt'
train_data, vocab, system_acts = load_data(fpath_train, entities, vocab, system_acts)
test_data, vocab, system_acts = load_data(fpath_test, entities, vocab, system_acts)
print('vocab size:', len(vocab))
print('action size:', len(system_acts))

max_turn_train = max([len(d[0]) for d in train_data])
max_turn_test = max([len(d[0]) for d in test_data])
max_turn = max(max_turn_train, max_turn_test)
print('max turn:', max_turn)
w2i = dict((w, i) for i, w in enumerate(vocab))
i2w = dict((i, w) for i, w in enumerate(vocab))
act2i = dict((act, i) for i, act in enumerate(system_acts))


embd_size = 300
hidden_size = 100
print('action_size:', len(system_acts))
model = HybridCodeNetwork(len(vocab), embd_size, hidden_size, len(system_acts))
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))


def train(model, data, optimizer, w2i, act2i, n_epochs=10, batch_size=64):
    for epoch in range(n_epochs):
        print('Epoch', epoch)
        random.shuffle(data)
        data = copy.copy(data)
        acc, total = 0, 0
        for i in range(0, len(data)-batch_size, batch_size):
            batch = data[i:i+batch_size]
            uttrs_list = [d[0] for d in batch]
            dialog_maxlen = max([len(uttrs) for uttrs in uttrs_list])
            uttr_maxlen = max([len(u) for uttrs in uttrs_list for u in uttrs])
#             print('dialog_maxlen', dialog_maxlen, ', uttr_maxlen', uttr_maxlen)
            uttr_var = make_word_vector(uttrs_list, w2i, dialog_maxlen, uttr_maxlen)
            batch_labels = [d[1] for d in batch]
            labels_var = []
            for labels in batch_labels:
                vec_labels = [act2i[l] for l in labels]
                pad_len = dialog_maxlen - len(labels)
#                 print('b vec_labels', len(vec_labels))
                for _ in range(pad_len):
                    vec_labels.append(act2i[SILENT])
#                 print('vec_labels', len(vec_labels))
                labels_var.append(torch.LongTensor(vec_labels))
            labels_var = to_var(torch.stack(labels_var, 0))
            context = copy.deepcopy([d[2] for d in batch])
            for i, c in enumerate(context):
                pad_len = dialog_maxlen - len(c)
                for _ in range(pad_len):
                    context[i].append([1] * len(entities.keys()))
            context = to_var(torch.FloatTensor(context))
            pred = model(uttr_var, context)
            action_size = pred.size(-1)
            pred = pred.view(-1, action_size)
            labels_var = labels_var.view(-1)
            loss = F.nll_loss(pred, labels_var)
            acc += torch.sum(labels_var == torch.max(pred, 1)[1]).data[0]
            total += labels_var.size(0)
            print('Acc: {:.3f}% ({}/{})'.format(100 * acc/total, acc, total))
            # print('loss', loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
train(model, train_data, optimizer, w2i, act2i)
