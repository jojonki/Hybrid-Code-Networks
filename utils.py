import re
import copy
import pickle
import numpy as np
import torch
from torch.autograd import Variable

SILENT = '<SILENT>' # TODO hard code
UNK = '<UNK>'


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)


def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)


def get_entities(fpath):
    # entities = {}
    entities = {'R_cuisine': [], 'R_location': [], 'R_price': [], 'R_number': []}
    with open(fpath, 'r') as file:
        lines = file.readlines()
        for l in lines:
            wds = l.rstrip().split(' ')[2].split('\t')
            slot_type = wds[0] # ex) R_price
            slot_val = wds[1] # ex) cheap
            # if slot_type not in entities:
            #     entities[slot_type] = []
            if slot_type in entities:
                if slot_val not in entities[slot_type]:
                    entities[slot_type].append(slot_val)
    return entities


def load_embd_weights(word2vec, vocab_size, embd_size, w2i):
    embedding_matrix = np.zeros((vocab_size, embd_size))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_ct = 0
    for word, idx in w2i.items():
        # words not found in embedding index will be all-zeros.
        if word in word2vec.wv:
            embedding_matrix[idx] = word2vec.wv[word]
            found_ct += 1
    print(found_ct, 'words are found in word2vec. vocab_size is', vocab_size)
    return torch.from_numpy(embedding_matrix).type(torch.FloatTensor)


def load_vocab(fpath, vocab):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            l = l.rstrip()
            if l != '':
                ls = l.split("\t")
                t_u = ls[0].split(' ', 1)
                # turn = t_u[0]
                uttr = t_u[1].split(' ')
                if len(ls) == 2: # includes user and system utterance
                    for w in uttr:
                        if w not in vocab:
                            vocab.append(w)
    vocab = sorted(vocab)
    return vocab


def load_data(fpath, entities, w2i, system_acts):
    data = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        # x: user uttr, y: sys act, c: context, b: BoW, p: previous sys act
        x, y, c, b, p = [], [], [], [], []
        context = [0] * len(entities.keys())
        for idx, l in enumerate(lines):
            l = l.rstrip()
            if l == '':
                data.append((x, y, c, b, p))
                # reset
                x, y, c, b, p = [], [], [], [], []
                context = [0] * len(entities.keys())
            else:
                ls = l.split("\t")
                t_u = ls[0].split(' ', 1)
                # turn = t_u[0]
                uttr = t_u[1].split(' ')
                update_context(context, uttr, entities)
                bow = get_bow(uttr, w2i)
                sys_act = SILENT
                if len(ls) == 2: # includes user and system utterance
                    sys_act = ls[1]
                    sys_act = re.sub(r'resto_\S+', '', sys_act)
                    if sys_act.startswith('api_call'): sys_act = 'api_call'
                    if sys_act not in system_acts: system_acts.append(sys_act)
                else:
                    continue # TODO

                x.append(uttr)
                if len(y) == 0:
                    p.append(SILENT)
                else:
                    p.append(y[-1])
                y.append(sys_act)
                c.append(copy.copy((context)))
                b.append(bow)
    return data, system_acts


def update_context(context, sentence, entities):
    for idx, (ent_key, ent_vals) in enumerate(entities.items()):
        for w in sentence:
            if w in ent_vals:
                context[idx] = 1


def get_bow(sentence, w2i):
    bow = [0] * len(w2i)
    for word in sentence:
        if word in w2i:
            bow[w2i[word]] += 1
    return bow


def add_padding(data, seq_len):
    pad_len = max(0, seq_len - len(data))
    data += [0] * pad_len
    data = data[:seq_len]
    return data


def make_word_vector(uttrs_list, w2i, dialog_maxlen, uttr_maxlen):
    dialog_list = []
    for uttrs in uttrs_list:
        dialog = []
        for sentence in uttrs:
            sent_vec = [w2i[w] if w in w2i else w2i[UNK] for w in sentence]
            sent_vec = add_padding(sent_vec, uttr_maxlen)
            dialog.append(sent_vec)
        for _ in range(dialog_maxlen - len(dialog)):
            dialog.append([0] * uttr_maxlen)
        dialog = torch.LongTensor(dialog[:dialog_maxlen])
        dialog_list.append(dialog)
    return to_var(torch.stack(dialog_list, 0))


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
