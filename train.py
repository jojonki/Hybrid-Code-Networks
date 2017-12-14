import random
import copy
import argparse
from gensim.models.keyedvectors import KeyedVectors
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import save_pickle, load_pickle, preload, load_embd_weights, load_data, to_var
from utils import get_entities, make_word_vector
from models import HybridCodeNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='each dialog formed one minibatch')
parser.add_argument('--embd_size', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for LSTM')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--resume', default='./checkpoints/model_best.tar', type=str, metavar='PATH', help='path saved params')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)


entities = get_entities('dialog-bAbI-tasks/dialog-babi-kb-all.txt')
for idx, (ent_name, ent_vals) in enumerate(entities.items()):
    print('entities', idx, ent_name, ent_vals[0] )

SILENT = '<SILENT>'
UNK = '<UNK>'
system_acts = [SILENT]
fpath_train = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt'
fpath_test = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-tst-OOV.txt'
vocab = []
vocab, system_acts = preload(fpath_train, vocab, system_acts) # only read training for vocab because OOV vocabrary should not know.
# print(vocab)
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
for act, i in act2i.items():
    print('act', i, act)

print('loading a word2vec binary...')
model_path = './data/GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
print('done')
pre_embd_w = load_embd_weights(word2vec, len(vocab), embd_size, w2i)
save_pickle(pre_embd_w, 'pre_embd_w.pickle')
pre_embd_w = load_pickle('pre_embd_w.pickle')

model = HybridCodeNetwork(len(vocab), args.embd_size, args.hidden_size, len(system_acts), pre_embd_w)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))


def padding(data, default_val, maxlen):
    for i, d in enumerate(data):
        pad_len = maxlen - len(d)
        for _ in range(pad_len):
            data[i].append([default_val] * len(entities.keys()))
    return to_var(torch.FloatTensor(data))


def get_data_from_batch(batch, w2i, act2i):
    uttrs_list = [d[0] for d in batch]
    dialog_maxlen = max([len(uttrs) for uttrs in uttrs_list])
    uttr_maxlen = max([len(u) for uttrs in uttrs_list for u in uttrs])
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

    batch_prev_acts = [d[4] for d in batch]
    prev_var = []
    for prev_acts in batch_prev_acts:
        vec_prev_acts = []
        for act in prev_acts:
            tmp = [0] * len(act2i)
            tmp[act2i[act]] = 1
            vec_prev_acts.append(tmp)
        pad_len = dialog_maxlen - len(prev_acts)
        for _ in range(pad_len):
            vec_prev_acts.append([0] * len(act2i))
        prev_var.append(torch.FloatTensor(vec_prev_acts))
    prev_var = to_var(torch.stack(prev_var, 0))

    context = copy.deepcopy([d[2] for d in batch])
    context = padding(context, 1, dialog_maxlen)

    bow = copy.deepcopy([d[3] for d in batch])
    bow = padding(bow, 0, dialog_maxlen)

    act_filter = copy.deepcopy([d[5] for d in batch])
    act_filter = padding(act_filter, 0, dialog_maxlen)

    return uttr_var, labels_var, context, bow, prev_var, act_filter


def categorical_cross_entropy(preds, labels):
    loss = Variable(torch.zeros(1))
    for p, label in zip(preds, labels):
        loss -= torch.log(p[label] + 1.e-7).cpu()
    loss /= preds.size(0)
    return loss


def train(model, data, optimizer, w2i, act2i, n_epochs=5, batch_size=1):
    print('----Train---')
    for epoch in range(n_epochs):
        print('Epoch', epoch)
        data = copy.copy(data)
        random.shuffle(data)
        acc, total = 0, 0
        for batch_idx in range(0, len(data)-batch_size, batch_size):
            batch = data[batch_idx:batch_idx+batch_size]
            uttrs, labels, contexts, bows, prevs, act_fils = get_data_from_batch(batch, w2i, act2i)

            preds = model(uttrs, contexts, bows, prevs, act_fils)
            action_size = preds.size(-1)
            preds = preds.view(-1, action_size)
            labels = labels.view(-1)
            # loss = F.nll_loss(preds, labels)
            loss = categorical_cross_entropy(preds, labels)
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
        uttrs, labels, contexts, bows, prevs, act_fils = get_data_from_batch(batch, w2i, act2i)

        preds = model(uttrs, contexts, bows, prevs, act_fils)
        action_size = preds.size(-1)
        preds = preds.view(-1, action_size)
        labels = labels.view(-1)
        # loss = F.nll_loss(preds, labels)
        acc += torch.sum(labels == torch.max(preds, 1)[1]).data[0]
        total += labels.size(0)
    print('Test Acc: {:.3f}% ({}/{})'.format(100 * acc/total, acc, total))

train(model, train_data, optimizer, w2i, act2i)
test(model, test_data, w2i, act2i)
