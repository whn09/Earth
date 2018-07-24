import os
import linecache

def build_vocab_dict(path):
    """
    Load Dataset from File
    """
    print('linecache.checkcache():', linecache.checkcache(path))
    label_to_int = {}
    vocab_to_int = {}
    int_to_label = {}
    int_to_vocab = {}
    count = 0
    label_cnt = 0
    vocab_cnt = 1
    f = open(path,'r')
    for line in f:
        #if count > 100:
        #    break
        count += 1
        line = line.strip().split(' ')
        for i in line:
            i = i.lower()
            if i.find('__label__') == 0:
                if i not in label_to_int:
                    label_to_int[i] = label_cnt
                    int_to_label[label_cnt] = i
                    label_cnt += 1
            else:
                if i not in vocab_to_int:
                    vocab_to_int[i] = vocab_cnt
                    int_to_vocab[vocab_cnt] = i
                    vocab_cnt += 1

    return label_to_int,vocab_to_int,int_to_label,int_to_vocab

def read_data_into_cache(path):
    return linecache.getlines(path)