import random
import copy
from gensim.models.keyedvectors import KeyedVectors
import torch
import torch.nn.functional as F

from utils import save_pickle, load_pickle, load_vocab, load_embd_weights, get_entities, load_data, make_word_vector, to_var
from models import HybridCodeNetwork

embd_size = 300
hidden_size = 128


entities = get_entities('dialog-bAbI-tasks/dialog-babi-kb-all.txt')

for idx, (ent_name, ent_vals) in enumerate(entities.items()):
    print('entities', idx, ent_name, ent_vals[0] )

# create training dataset
SILENT = '<SILENT>'
UNK = '<UNK>'
system_acts = [SILENT]
fpath_train = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt'
fpath_test = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-tst-OOV.txt'
vocab = []
vocab = load_vocab(fpath_train, vocab)
# vocab = load_vocab(fpath_test, vocab)
print(vocab)
w2i = dict((w, i) for i, w in enumerate(vocab, 1))
i2w = dict((i, w) for i, w in enumerate(vocab, 1))
w2i[UNK] = 0
i2w[0] = UNK
train_data, system_acts = load_data(fpath_train, entities, w2i, system_acts)
test_data, system_acts = load_data(fpath_test, entities, w2i, system_acts)
print('vocab size:', len(vocab))
print('action size:', len(system_acts))

max_turn_train = max([len(d[0]) for d in train_data])
max_turn_test = max([len(d[0]) for d in test_data])
max_turn = max(max_turn_train, max_turn_test)
print('max turn:', max_turn)
act2i = dict((act, i) for i, act in enumerate(system_acts))
print('action_size:', len(system_acts))

# print('loading a word2vec binary...')
# model_path = './data/GoogleNews-vectors-negative300.bin'
# word2vec = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
# print('done')
# pre_embd_w = load_embd_weights(word2vec, len(vocab), embd_size, w2i)
# save_pickle(pre_embd_w, 'pre_embd_w.pickle')
pre_embd_w = load_pickle('pre_embd_w.pickle')

model = HybridCodeNetwork(len(vocab), embd_size, hidden_size, len(system_acts), pre_embd_w)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))


def get_data_from_batch(batch, w2i, act2i):
    uttrs_list = [d[0] for d in batch]
    dialog_maxlen = max([len(uttrs) for uttrs in uttrs_list])
    uttr_maxlen = max([len(u) for uttrs in uttrs_list for u in uttrs])
    # print('dialog_maxlen', dialog_maxlen, ', uttr_maxlen', uttr_maxlen)
    uttr_var = make_word_vector(uttrs_list, w2i, dialog_maxlen, uttr_maxlen)

    batch_labels = [d[1] for d in batch]
    labels_var = []
    for labels in batch_labels:
        vec_labels = [act2i[l] for l in labels]
        pad_len = dialog_maxlen - len(labels)
        for _ in range(pad_len):
            vec_labels.append(act2i[SILENT])
        labels_var.append(torch.LongTensor(vec_labels))
    labels_var = to_var(torch.stack(labels_var, 0))

    context = copy.deepcopy([d[2] for d in batch])
    for i, c in enumerate(context):
        pad_len = dialog_maxlen - len(c)
        for _ in range(pad_len):
            context[i].append([1] * len(entities.keys()))
    context = to_var(torch.FloatTensor(context))

    bow = copy.deepcopy([d[3] for d in batch])
    for i, b in enumerate(bow):
        pad_len = dialog_maxlen - len(b)
        for _ in range(pad_len):
            bow[i].append([0] * len(w2i))
    bow = to_var(torch.FloatTensor(bow))

    return uttr_var, labels_var, context, bow


def train(model, data, optimizer, w2i, act2i, n_epochs=5, batch_size=1):
    print('----Train---')
    for epoch in range(n_epochs):
        print('Epoch', epoch)
        data = copy.copy(data)
        random.shuffle(data)
        acc, total = 0, 0
        for batch_idx in range(0, len(data)-batch_size, batch_size):
            batch = data[batch_idx:batch_idx+batch_size]
            uttrs, labels, contexts, bows = get_data_from_batch(batch, w2i, act2i)

            preds = model(uttrs, contexts, bows)
            action_size = preds.size(-1)
            preds = preds.view(-1, action_size)
            labels = labels.view(-1)
            loss = F.nll_loss(preds, labels)
            acc += torch.sum(labels == torch.max(preds, 1)[1]).data[0]
            total += labels.size(0)
            if batch_idx % (100 * batch_size) == 0:
                print('Acc: {:.3f}% ({}/{})'.format(100 * acc/total, acc, total))
                print('loss', loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, data, w2i, act2i, batch_size=1):
    print('----Test---')
    model.eval()
    acc, total = 0, 0
    for batch_idx in range(0, len(data)-batch_size, batch_size):
        batch = data[batch_idx:batch_idx+batch_size]
        uttrs, labels, contexts, bows = get_data_from_batch(batch, w2i, act2i)

        preds = model(uttrs, contexts, bows)
        action_size = preds.size(-1)
        preds = preds.view(-1, action_size)
        labels = labels.view(-1)
        loss = F.nll_loss(preds, labels)
        acc += torch.sum(labels == torch.max(preds, 1)[1]).data[0]
        total += labels.size(0)
    print('Test Acc: {:.3f}% ({}/{})'.format(100 * acc/total, acc, total))
    # print('loss', loss.data[0])

train(model, train_data, optimizer, w2i, act2i)
test(model, test_data, w2i, act2i)
