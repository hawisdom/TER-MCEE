from collections import Counter, defaultdict
import numpy as np
import torch
import os
import json
import logging
import pickle,joblib
from torch.utils.data import Dataset
from tree_Operate import *
from data_process import event_roles
from bert_serving.client import BertClient

logger = logging.getLogger(__name__)

def load_datasets_and_vocabs(args):
    train_example_file = os.path.join(args.cache_dir, 'train_example.pkl')
    dev_example_file = os.path.join(args.cache_dir, 'dev_example.pkl')
    test_example_file = os.path.join(args.cache_dir, 'test_example.pkl')
    train_weight_file = os.path.join(args.cache_dir, 'train_weight_cache.txt')
    dev_weight_file = os.path.join(args.cache_dir, 'dev_weight_cache.txt')
    test_weight_file = os.path.join(args.cache_dir, 'test_weight_cache.txt')

    if os.path.exists(train_example_file) and os.path.exists(dev_example_file) and os.path.exists(test_example_file):
        logger.info('Loading train_example from %s', train_example_file)
        with open(train_example_file, 'rb') as f:
            train_examples = joblib.load(f)
        logger.info('Loading dev_example from %s', dev_example_file)
        with open(dev_example_file, 'rb') as f:
            dev_examples = joblib.load(f)
        logger.info('Loading test_example from %s', test_example_file)
        with open(test_example_file, 'rb') as f:
            test_examples = joblib.load(f)

        with open(train_weight_file, 'rb') as f:
            train_labels_weight = json.load(f)
        with open(dev_weight_file, 'rb') as f:
            dev_labels_weight = json.load(f)
        with open(test_weight_file, 'rb') as f:
            test_labels_weight = json.load(f)
    else:
        train_tree_file = os.path.join(args.dataset_path,'train.pkl')
        dev_tree_file = os.path.join(args.dataset_path,'dev.pkl')
        test_tree_file = os.path.join(args.dataset_path,'tests.pkl')

        # get examples of data
        logger.info('Loading train trees')
        with open(train_tree_file, 'rb') as f:
            train_trees = pickle.load(f)
        train_examples, train_labels_weight = create_example(train_trees, args.train_event_num,train_tree_file)
        with open(train_weight_file, 'w') as wf:
            json.dump(train_labels_weight, wf)
        logger.info('Creating train examples')
        with open(train_example_file, 'wb') as f:
            joblib.dump(train_examples, f)

        logger.info('Loading test trees')
        with open(test_tree_file, 'rb') as f:
            test_trees = pickle.load(f)
        test_examples,test_labels_weight = create_example(test_trees,args.test_event_num,test_tree_file)
        with open(test_weight_file,'w') as wf:
            json.dump(test_labels_weight,wf)
        logger.info('Creating test examples')
        with open(test_example_file,'wb') as f:
            joblib.dump(test_examples,f)

        logger.info('Loading dev trees')
        with open(dev_tree_file, 'rb') as f:
            dev_trees = pickle.load(f)
        dev_examples, dev_labels_weight = create_example(dev_trees, args.dev_event_num,dev_tree_file)
        with open(dev_weight_file, 'w') as wf:
            json.dump(dev_labels_weight, wf)
        logger.info('Creating dev examples')
        with open(dev_example_file, 'wb') as f:
            joblib.dump(dev_examples, f)


    logger.info('Train set size: %s', len(train_examples))
    logger.info('Dev set size: %s', len(dev_examples))
    logger.info('Test set size: %s,', len(test_examples))

    # Build word vocabulary(dep_tag, part of speech) and save pickles.
    word_vecs, word_vocab, pos_tag_vocab, dep_tag_vocab, sen_pos_tag_vocab, word_pos_tag_vocab, event_id_tag_vocab = load_and_cache_vocabs(train_examples+dev_examples+test_examples, args)

    embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
    args.token_embedding = embedding

    train_dataset = EE_Dataset(train_examples,args,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab)
    dev_dataset = EE_Dataset(dev_examples,args,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab)
    test_dataset = EE_Dataset(test_examples,args,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab)

    return train_dataset,train_labels_weight,dev_dataset,dev_labels_weight,test_dataset,test_labels_weight,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab, word_pos_tag_vocab,event_id_tag_vocab

def create_example(docs,event_num,file_name):
    examples = []
    type_labels_ids = {}
    type_labels = {}
    for doc_id,doc in enumerate(docs):
        example = {'tokens':[],'pos':[],'deps':[],'sen_pos':[],'word_pos':[],'parents':[],'ppos':[],'pdeps':[],'e_ids':[]}
        nodes = doc.dp_tree.all_nodes()
        nodes.sort(key=doc.node_sort)
        for node in nodes:
            if node.identifier == DROOT:
                continue
            for i in range(event_num):
                example['tokens'].append(node.tag)
                example['pos'].append(node.data.pos)
                example['deps'].append(node.data.dep)
                example['sen_pos'].append(node.data.sen_pos)
                example['word_pos'].append(node.data.token_id)
                for e_id,etype in enumerate(node.data.token_labels.keys()):
                    if etype not in type_labels_ids.keys():
                        type_labels_ids[etype] = []
                    if etype not in type_labels.keys():
                        type_labels[etype] = []
                    type_labels[etype].append(int(node.data.token_labels[etype][i]))
                    type_labels_ids[etype].append(int(node.data.token_labels[etype][i]))

                pnode = doc.dp_tree.parent(node.identifier)
                example['parents'].append(pnode.tag)
                example['ppos'].append(pnode.data.pos)
                example['pdeps'].append(pnode.data.dep)
                example['e_ids'].append(i)

        example['type_labels'] = list(type_labels.values())
        type_labels.clear()

        examples.append(example)

    labels_weight = {}
    for etype in event_roles.keys():
        labels_weight[etype] = get_labels_weight(type_labels_ids[etype],event_roles[etype])

    return examples,labels_weight

def get_labels_weight(label_ids,labels_lookup):
    nums_labels = Counter(label_ids)
    nums_labels = [(l,k) for k, l in sorted([(j, i) for i, j in nums_labels.items()], reverse=True)]
    size = len(nums_labels)
    if size % 2 == 0:
        median = (nums_labels[size // 2][1] + nums_labels[size//2-1][1])/2
    else:
        median = nums_labels[(size - 1) // 2][1]

    weight_list = []
    for value_id in labels_lookup.values():
        if value_id not in label_ids:
            weight_list.append(0)
        else:
            for label in nums_labels:
                if label[0] == value_id:
                    weight_list.append(median/label[1])
                    break
    return weight_list

def load_and_cache_vocabs(examples,args):
    embedding_cache_path = os.path.join(args.cache_dir, 'embedding')
    if not os.path.exists(embedding_cache_path):
        os.makedirs(embedding_cache_path)

    # Build or load word vocab and word2vec embeddings.
    cached_word_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_word_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vocab_file):
        logger.info('Loading word vocab from %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'rb') as f:
            word_vocab = pickle.load(f)
    else:
        logger.info('Creating word vocab from dataset %s',args.dataset_name)
        word_vocab = build_text_vocab(examples)
        logger.info('Word vocab size: %s', word_vocab['len'])
        logging.info('Saving word vocab to %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'wb') as f:
            pickle.dump(word_vocab, f, -1)

    cached_word_vecs_file = os.path.join(embedding_cache_path, 'cached_{}_word_vecs.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vecs_file):
        logger.info('Loading word vecs from %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'rb') as f:
            word_vecs = pickle.load(f)
    else:
        logger.info('Creating word vecs from BERT')
        word_vecs = load_bert_embedding(word_vocab['itos'])
        logger.info('Saving word vecs to %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'wb') as f:
            pickle.dump(word_vecs, f, -1)

    # Build vocab of dep tags.
    cached_dep_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dep tags from %s', cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dep tags.')
        dep_tag_vocab = build_dep_tag_vocab(examples, min_freq=0)
        logger.info('Saving dep tags  vocab, size: %s, to file %s', dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of pos tags from %s',cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of pos tags.')
        pos_tag_vocab = build_pos_tag_vocab(examples, min_freq=0)
        logger.info('Saving pos tags  vocab, size: %s, to file %s',pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    # Build vocab of sentence position tags.
    cached_sen_pos_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_sen_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_sen_pos_tag_vocab_file):
        logger.info('Loading vocab of sentence pos tags from %s', cached_sen_pos_tag_vocab_file)
        with open(cached_sen_pos_tag_vocab_file, 'rb') as f:
            sen_pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of sentence pos tags.')
        sen_pos_tag_vocab = build_sen_pos_tag_vocab(examples, min_freq=0)
        logger.info('Saving sentence pos tags  vocab, size: %s, to file %s', sen_pos_tag_vocab['len'],
                    cached_sen_pos_tag_vocab_file)
        with open(cached_sen_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(sen_pos_tag_vocab, f, -1)

    # Build vocab of word position tags.
    cached_word_pos_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_word_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_pos_tag_vocab_file):
        logger.info('Loading vocab of word pos tags from %s', cached_word_pos_tag_vocab_file)
        with open(cached_word_pos_tag_vocab_file, 'rb') as f:
            word_pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of word pos tags.')
        word_pos_tag_vocab = build_word_pos_tag_vocab(examples, min_freq=0)
        logger.info('Saving word pos tags  vocab, size: %s, to file %s', word_pos_tag_vocab['len'],
                    cached_word_pos_tag_vocab_file)
        with open(cached_word_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(word_pos_tag_vocab, f, -1)

    # Build vocab of event ids tags.
    cached_event_id_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_event_id_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_event_id_tag_vocab_file):
        logger.info('Loading vocab of event id tags from %s', cached_event_id_tag_vocab_file)
        with open(cached_event_id_tag_vocab_file, 'rb') as f:
            event_id_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of event id tags.')
        event_id_tag_vocab = build_event_id_tag_vocab(examples, min_freq=0)
        logger.info('Saving event id tags  vocab, size: %s, to file %s', event_id_tag_vocab['len'],
                    cached_event_id_tag_vocab_file)
        with open(cached_event_id_tag_vocab_file, 'wb') as f:
            pickle.dump(event_id_tag_vocab, f, -1)

    return word_vecs,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab

def load_bert_embedding(word_list):
    word_vectors = []
    bc = BertClient()
    for word in word_list:
        if word == 'pad':
            word_vectors.append(np.zeros(768, dtype=np.float32))
        else:
            word_vectors.append(bc.encode([word])[0])
    return word_vectors

def _default_unk_index():
    return 1

def build_text_vocab(examples, vocab_size=1000000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['tokens']+example['parents'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_dep_tag_vocab(examples, vocab_size=1000, min_freq=0):
    """
    dependency tags vocab.
    """
    counter = Counter()
    for example in examples:
        counter.update(example['deps']+example['pdeps'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_pos_tag_vocab(examples, vocab_size=1000, min_freq=0):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for example in examples:
        counter.update(example['pos']+example['ppos'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_sen_pos_tag_vocab(examples, vocab_size=1000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['sen_pos'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_word_pos_tag_vocab(examples, vocab_size=10000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['word_pos'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_event_id_tag_vocab(examples, vocab_size=10000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['e_ids'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

class EE_Dataset(Dataset):
    def __init__(self, examples,args,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab):
        self.examples = examples
        self.args = args
        self.word_vocab = word_vocab
        self.pos_tag_vocab = pos_tag_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.sen_pos_tag_vocab = sen_pos_tag_vocab
        self.word_pos_tag_vocab = word_pos_tag_vocab
        self.event_id_tag_vocab = event_id_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        items = e['token_ids'],e['pos_ids'],e['dep_ids'],e['sen_pos_ids'],e['word_pos_ids'],e['parent_ids'],e['ppos_ids'],e['pdep_ids'],e['event_ids'],\
                e['type_labels']

        return items

    def convert_features(self):
        for i in range(len(self.examples)):
            self.examples[i]['token_ids'] = [self.word_vocab['stoi'][w] for w in self.examples[i]['tokens']]
            self.examples[i]['parent_ids'] = [self.word_vocab['stoi'][w] for w in self.examples[i]['parents']]
            self.examples[i]['pos_ids'] = [self.pos_tag_vocab['stoi'][p] for p in self.examples[i]['pos']]
            self.examples[i]['ppos_ids'] = [self.pos_tag_vocab['stoi'][p] for p in self.examples[i]['ppos']]
            self.examples[i]['dep_ids'] = [self.dep_tag_vocab['stoi'][d] for d in self.examples[i]['deps']]
            self.examples[i]['pdep_ids'] = [self.dep_tag_vocab['stoi'][d] for d in self.examples[i]['pdeps']]
            self.examples[i]['sen_pos_ids'] = [self.sen_pos_tag_vocab['stoi'][sen_pos] for sen_pos in
                                               self.examples[i]['sen_pos']]
            self.examples[i]['word_pos_ids'] = [self.word_pos_tag_vocab['stoi'][word_pos] for word_pos in
                                                self.examples[i]['word_pos']]
            self.examples[i]['event_ids'] = [self.event_id_tag_vocab['stoi'][e] for e in self.examples[i]['e_ids']]


def my_collate(batch):
    token_ids,pos_ids,dep_ids,sen_pos_ids,word_pos_ids,parent_ids,ppos_ids,pdep_ids,event_ids,\
    type_labels = zip(
        *batch)

    token_ids = torch.tensor(token_ids[0])
    pos_ids = torch.tensor(pos_ids[0])
    dep_ids = torch.tensor(dep_ids[0])
    sen_pos_ids = torch.tensor(sen_pos_ids[0])
    word_pos_ids = torch.tensor(word_pos_ids[0])
    parent_ids = torch.tensor(parent_ids[0])
    ppos_ids = torch.tensor(ppos_ids[0])
    pdep_ids = torch.tensor(pdep_ids[0])
    event_ids = torch.tensor(event_ids[0])
    type_labels = torch.tensor(type_labels[0])


    return token_ids,pos_ids,dep_ids,sen_pos_ids,word_pos_ids,parent_ids,ppos_ids,pdep_ids,event_ids, \
           type_labels
