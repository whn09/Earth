import os
import linecache

def build_vocab_dict(path):
    """
    Load Dataset from File
    """
    print('linecache.checkcache():', linecache.checkcache(path))
    label_to_int = {}
    vocab_to_int = {}
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
                    label_cnt += 1
            else:
                if i not in vocab_to_int:
                    vocab_to_int[i] = vocab_cnt
                    vocab_cnt += 1

    return label_to_int,vocab_to_int

def read_data_into_cache(path):
    return linecache.getlines(path)