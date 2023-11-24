import logging
import json
from ltp import LTP
import pickle
import numpy as np
from tree_Operate import *
logger = logging.getLogger(__name__)

event_roles = {
    'EquityFreeze': {'O':0,'EquityHolder': 1, 'FrozeShares': 2, 'LegalInstitution': 3, 'TotalHoldingShares': 4, 'TotalHoldingRatio': 5, 'StartDate': 6, 'EndDate': 7, 'UnfrozeDate': 8},
    'EquityRepurchase': {'O':0,'CompanyName': 1, 'HighestTradingPrice': 2, 'LowestTradingPrice': 3, 'RepurchasedShares': 4, 'ClosingDate': 5, 'RepurchaseAmount': 6},
    'EquityUnderweight': {'O':0,'EquityHolder': 1,'TradedShares': 2, 'StartDate': 3,'EndDate': 4, 'LaterHoldingShares': 5, 'AveragePrice': 6},
    'EquityOverweight': {'O':0,'EquityHolder': 1, 'TradedShares': 2, 'StartDate': 3, 'EndDate': 4,'LaterHoldingShares': 5, 'AveragePrice': 6},
    'EquityPledge': {'O':0,'Pledger': 1, 'PledgedShares': 2, 'Pledgee': 3, 'TotalHoldingShares': 4, 'TotalHoldingRatio': 5, 'TotalPledgedShares': 6, 'StartDate': 7, 'EndDate': 8, 'ReleasedDate': 9}}

train_event_nums = 34
dev_event_nums = 15
test_event_nums = 16

def parsing_document(file,tree_file,data_type):
    stopwords = []
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f_stopword:
        stopword_datas = f_stopword.readlines()
        for stopword in stopword_datas:
            stopwords.append(stopword.strip())

    ltp = LTP()
    with open(file, 'r', encoding='utf-8-sig') as fp:
        datas = json.load(fp)

    if data_type == 'train':
        event_num = train_event_nums
    elif data_type == 'dev':
        event_num = dev_event_nums
    elif data_type == 'test':
        event_num = test_event_nums
    else:
        event_num = None

    docs = []
    for doc in datas:
        sentences = doc[1]['sentences']
        events = doc[1]['recguid_eventname_eventdict_list']
        arguments = doc[1]['ann_mspan2dranges']
        doc_tree_obj = Doc_Tree()
        sen_id = 0
        word_id = 0
        for sentence in sentences:
            token_ids = []
            sentence = sentence.strip()
            sentence = sentence.strip('')
            sentence = sentence.strip('')
            if len(sentence) == 0:
                continue
            words, hidden = ltp.seg([sentence.strip()])
            words = words[0]
            word_pos = 0
            word_pos_list = []
            for i, word in enumerate(words):
                word_pos_list.append([sen_id, word_pos, word_pos + len(word)])
                word_pos = word_pos + len(word)
                token_ids.append(i)
            token_labels = get_token_labels(words, word_pos_list, arguments, events, event_num)
            pos = ltp.pos(hidden)[0]
            deps = ltp.dep(hidden)[0]
            dep_new = []
            for j,dep in enumerate(deps):
                if dep[1] == 0:
                    dep_new.append((dep[0] + word_id, dep[1], dep[2]))
                else:
                    dep_new.append((dep[0] + word_id, dep[1] + word_id, dep[2]))
            doc_tree_obj.build_dp_tree_ltp4(words, dep_new, pos, DROOT, sen_id, word_pos_list, token_ids,token_labels)
            doc_tree_obj.remove_stop_word_nodes_tree(stopwords)
            sen_id += 1
            word_id += len(words)
        docs.append(doc_tree_obj)

    with open(tree_file, 'wb') as f:
        pickle.dump(docs, f, -1)


def get_token_labels(words, word_pos_list, arguments, events, event_num):
    token_labels = []
    for i, token in enumerate(words):
        labels = init_role_event_labels(event_num)
        for argument, position_list in arguments.items():
            if token in argument:
                for position in position_list:
                    if word_pos_list[i][0] == position[0] and word_pos_list[i][1] >= position[1] and word_pos_list[i][2] <= position[2]:
                        update_token_role_event(token, events, labels, event_num)
        token_labels.append(labels)
    return token_labels


def init_role_event_labels(event_num):
    labels = {}
    for event_type in event_roles.keys():
        role_event = np.zeros([event_num])
        labels[event_type] = role_event
    return labels


def update_token_role_event(word, events, labels, event_num):
    event_ids = {'EquityFreeze': 0, 'EquityRepurchase': 0, 'EquityUnderweight': 0, 'EquityOverweight': 0, 'EquityPledge': 0}
    for i, event in enumerate(events):
        event_type = event[1]
        event_arguments = event[2]
        for role, value in event_arguments.items():
            if value is None:
                continue
            if event_ids[event_type] > event_num - 1:
                continue
            if word in value:
                role_id = event_roles[event_type][role]
                labels[event_type][event_ids[event_type]] = role_id
                event_ids[event_type] += 1


if __name__ == '__main__':
    parsing_document('./data/train.json','./data/train.pkl','train')
    parsing_document('./data/dev.json', './data/dev.pkl', 'dev')
    parsing_document('./data/test.json','./data/test.pkl','test')
