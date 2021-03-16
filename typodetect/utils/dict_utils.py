# -*- coding:utf-8 -*-

import codecs
import os
import sys
from .io_utils import dump_pkl
from .text_utils import is_chinese_string

def load_line_set(filename):
    data = []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            data.append(w)
    return data

def load_fixed_pairs_dict_py(filename, sep = '\t'):
    word_dict = {'head':{},'tail':{}}
    lines = codecs.open(filename, 'r', 'utf8').readlines()
    for line in lines:
        line = line.strip().split(sep)
        etype,first,others = line[0], line[1], line[2:]
        word_dict[etype][first] = {}
        for other in others:
            other_list = other.split("|||")
            pinyin = other_list[0]
            cand = other_list[1:]
            word_dict[etype][first][pinyin] = cand
    return word_dict
   
def load_other_ners_dict(filename, sep = '\t'):
    word_dict = {'vdo':{},'loc':{},'org':{}}
    lines = codecs.open(filename,'r','utf8').readlines()
    for line in lines:
        line = line.strip().split(sep)
        dtype, pinyin, name = line[0], line[1], line[2:]
        if pinyin in word_dict[dtype]:
            word_dict[dtype][pinyin].extend(name)
        else:
            word_dict[dtype][pinyin] = name
    return word_dict 

def load_vdo_name_dict(filename,sep = '\t'):
    word_dict = {}
    lines = codecs.open(filename,'r','utf8').readlines()
    for line in lines:
        line = line.strip().split(sep)
        vdo_names,pinyins,names = line[0].split('|||'),line[1].split('|||'),line[2].split('|||')
        for vdo_name in vdo_names:
            if vdo_name not in word_dict:
                word_dict[vdo_name] = {}
                for p, n in zip(pinyins, names):
                    if p not in word_dict[vdo_name]:
                        word_dict[vdo_name][p] = [n]
                    else:
                        word_dict[vdo_name][p].append(n)
            else:
                for p, n in zip(pinyins, names):
                    if p not in word_dict[vdo_name]:
                        word_dict[vdo_name][p] = [n]
                    else:
                        word_dict[vdo_name][p].append(n)
    return word_dict

def load_key_values_dict(filename, sep1 = ':', sep2 = '\t'):
    data = dict()
    with codecs.open(filename, 'r', encoding = 'utf-8') as f:
        for line in f:
            parts = line.strip().split(sep1)
            key = parts[0].strip()
            if sep1 != sep2:
                values = parts[1].strip().strip("\t").split(sep2)
            else:
                values = parts[1:]
            data[key] = values
    return data

def load_key_values_info_dict(filename):
    data = dict()
    with codecs.open(filename, 'r', encoding = 'utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            key_info = parts[0].strip().split('\0')
            if int(key_info[1]) < 2:
                continue
            key = key_info[0]
            values = []
            for v in parts[1].strip().split():
                v_info = v.split('\0')
                if v_info[1] < 2:
                    continue
                values.append(v_info[0])
            if len(values) > 0:
                data[key] = values
    return data

def load_key_set(filename, lastnum = None, sep='\t'):
    word_dict = []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            w_v = line.strip().split(sep)
            w = w_v[0]
            word_dict.append(w)
    word_dict = set(word_dict[: lastnum])
    return word_dict

def load_key_value_dict(filename, sep='\t'):
    word_dict = dict()
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            w_v = line.strip().split(sep)
            w = w_v[0]
            v = w_v[1]
            #if is_chinese_string(w):
            word_dict[w] = v
    return word_dict

def load_keys_values_dict(path, sep='\t'):
    result = dict()
    if not os.path.exists(path):
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) > 1:
                for i, c in enumerate(parts):
                    if c not in result:
                        result[c] = set(list(parts[:i] + parts[i + 1:]))
                    else:
                        result[c] = result[c] | set(list(parts[:i] + parts[i + 1:]))
    return result

def load_multiple_keys_values_dict(path, sep='\t'):
    result = dict()
    if not os.path.exists(path):
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) == 2:
                keys = parts[0].split()
                values = parts[0].split()
                for key in keys:
                    result[key] = values

    return result

def load_jinyin_zi(path, sep='\t'):
    result = dict()
    if not os.path.exists(path):
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) == 2:
                tongyin_zi = parts[0]
                jinyin_zi = parts[1]
                for i, c in enumerate(tongyin_zi):
                    result[c] = set(list(jinyin_zi))
    return result

def load_jinyin_ci(path, sep = ';;', sep2 = ' '):
    result = dict()
    if not os.path.exists(path):
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) == 2 and len(parts[1]) > 0:
                tongyin_ci = parts[0].strip().split(sep2)
                jinyin_ci = parts[1].strip().split(sep2)
                for i, c in enumerate(tongyin_ci):
                    result[c] = set(list(jinyin_ci))
    return result

def load_confusion_set_ci_filter(path, sep=':'):
    result = dict()
    if not os.path.exists(path):
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) == 2:
                ci = parts[0]
                tj_parts = parts[1].strip().split(';;')
                if len(tj_parts) == 2 and len(tj_parts[1]) > 0:
                    jinyin_candlist = tj_parts[1].strip().split('\t')
                    cand = set(list(jinyin_candlist))
                    result[ci] = cand
    return result

def keephot_cand_dict(cand_dict):
    for w in cand_dict:
        if len(cand_dict[w]) > 10:
            w_dict = {}
            for cand in cand_dict[w]:
                if cand in word_freq:
                    w_dict[cand] = word_freq[cand]
                else:
                    w_dict[cand] = 0
            #sort
            sorted_dict = dict(sorted(w_dict.items(), key=lambda d: d[1], reverse=True))
            top_candidates = [key for key, val in sorted_dict.items()]
            if len(top_candidates) > 10:
                top_candidates = top_candidates[:10]
            cand_dict[w] = set(top_candidates)

if __name__ == '__main__':
    data_path = '/apdcephfs/private_rigorosyang/py3_grammarerror_baseline/data'
    top_dict = load_key_value_dict(os.path.join(data_path, 'top_word_40000'))
    w8 = '人间天堂'
    print (w8[0:2] in top_dict, w8[0:2])
    print (w8[-2:] in top_dict, w8[-2:])
    sys.exit()
    wrong_name = load_key_value_dict(os.path.join(data_path, 'name_dict/wrong_name.csv'), ',')                  
    vdo_name = load_vdo_name_dict(os.path.join(data_path, 'name_dict/vdo_name_pinyin'))                         
    all_name_pinyin = load_key_values_dict(os.path.join(data_path, 'name_dict/mingxing_names_pinyin'), '\t', '\t')     
    other_ners_pinyin = load_other_ners_dict(os.path.join(data_path, 'name_dict/other_ners_pinyin'))            
    loc_baidu_wrong = load_key_value_dict(os.path.join(data_path, 'name_dict/loc_baidu_wrong.csv'), ',')     
    sys.exit()
    #data = load_jinyin_ci('/data5/lyncao/grammarerror/data/build_dict/word2_kb_top8w_withcommon_similar_keyboard_near', sep='\t')
    #print(len(data))
    #dump_pkl(data, 'data/word2_kb_top8w_withcommon_similar_keyboard_near.pkl')
