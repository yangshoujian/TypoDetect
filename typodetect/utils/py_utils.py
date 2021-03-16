# -*- coding: utf-8 -*-

import re
import sys
import os
import codecs
import logging
from pypinyin import lazy_pinyin
from pypinyin import pinyin, TONE3, Style

SHENGMU = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']

CONFUSE_SM = {'s' : 'sh', 'sh' : 's', 
              'c' : 'ch', 'ch' : 'c',
              'z' : 'zh', 'zh' : 'z',
              'l' : 'n', 'n' : 'l',
              }
CONFUSE_YM = {'in' : ('ing',), 'ing' : ('in',),
              'en' : ('eng',),  'eng' : ('en',),
              'ong' : ('on',), 'on' : ('ong',),
              'an' : ('ang',), 'ang' : ('an',)}
CONFUSE_YIN = {'xuan' : 'xian', 'xian' : 'xuan',
               'qian' : 'quan', 'quan' : 'qian',
               'xu' : 'xi', 'xi' : 'xu',
               'qu' : 'qi', 'qi' : 'qu',
               'yu' : 'yi', 'yi' : 'yu',
               'ju' : 'ji', 'ji' : 'ju',
               'me' : 'mo', 'mo' : 'me',
               'nv' : 'lv', 'lv' : 'nv',
               'yong': 'you', 'nan' : 'lan', 'lan' : 'nan',
                'hu' : 'fu', 'fu' : 'hu'}
'''
CONFUSE_SM = {'s' : 'sh', 'sh' : 's', 
              'c' : 'ch', 'ch' : 'c',
              'z' : 'zh', 'zh' : 'z',
              'h' : 'f', 'f' : 'h',
              'l' : 'n',  'n' : 'l',
              'r' : 'l',  'l' : 'r'}

CONFUSE_YM = {'in' : ('ing',), 'ing' : ('in',),
              'en' : ('eng', 'e'),  'eng' : ('en',),
              'ong' : ('on',), 'on' : ('ong',),
              'e' : ('en','er'), 'ian' : ('in',),
              'an' : ('ang', 'a'), 'ang' : ('an',)}
CONFUSE_YIN = {'xuan' : 'xian', 'xian' : 'xuan',
               'qian' : 'quan', 'quan' : 'qian',
               'she' : 'shen', 'shen' : 'she',
               'xu' : 'xi', 'xi' : 'xu',
               'qu' : 'qi', 'qi' : 'qu',
               'yu' : 'yi', 'yi' : 'yu',
               'ju' : 'ji', 'ji' : 'ju',
               'me' : 'mo', 'mo' : 'me',
               'ju' : 'jue', 'lou' : 'lu', 'lu' : 'lou',
               'yong': 'you', 'nan' : 'lan', 'lan' : 'nan',
                'hu' : 'fu', 'fu' : 'hu'}
'''
def load_char_multi_py(filename):
    #load 多音字典，返回相似音字典、多音字字典    
    data, char_pys = {}, {}           
    with codecs.open(filename, 'r', encoding='utf-8') as f:
    #for line in open(filename, 'r'):  
        for line in f:
            parts = line.strip().split(':')
            single_word = parts[0].strip()
            values = parts[1].strip().split('\t')
            if len(values) > 1:
                char_pys[single_word] = values
                for i, c in enumerate(values):
                    if c in data:
                        data[c].extend(values[:i] + values[i + 1:])
                    else:
                        data[c] = values[:i] + values[i + 1:]
    similar_pys = {}
    for k, v in data.items():
        similar_pys[k] = set(v)
    return similar_pys, char_pys
#load 多音字读音pair
MULTI_PYS, CHAR_MULTI_PYS = load_char_multi_py('data/hanyu_single_muliti_py')

def is_py_similar(word1_py, word2_py):
    len_word1_py, len_word2_py = len(word1_py), len(word2_py)
    if abs(len_word1_py - len_word2_py) > 1:
        return False
    str_small, str_big = word1_py, word2_py
    len_small, len_big = len_word1_py, len_word2_py
    if len_word2_py < len_word1_py:
        str_small, str_big = word2_py, word1_py
        len_small, len_big = len_word2_py, len_word1_py
    i, j, diff_num = 0, 0, 0
    while i < len_small and j < len_big:
        if  str_small[i] == str_big[j]:
            i += 1;
            j += 1;
            continue
        diff_num += 1
        if len_small == len_big:
            i += 1
            j += 1
        else:
            j += 1
    while j < len_big:
        j += 1
        diff_num += 1

    if i == len_small and j == len_big and diff_num <= 1:
        return True
    return False

def get_char_pronunciation_type(char1_u8, char2_u8):
    ret = lazy_pinyin(char1_u8, style=TONE3)
    if len(ret) < 1: return -1
    char1_py = ret[0].encode('utf8')
    ret = lazy_pinyin(char2_u8, style=TONE3)
    if len(ret) < 1: return -1
    char2_py = ret[0].encode('utf8')
    if char1_py == char2_py:
        return 0
    char1_py = char1_py[0 : -1]
    char2_py = char2_py[0 : -1]
    if (char1_py == char2_py or char1_py in CHAR_MULTI_PYS.get(char2_u8, [])
            or char2_py in CHAR_MULTI_PYS.get(char1_u8, [])):
        return 1
    if is_py_similar(char1_py, char2_py):
        return 2
    return 3

def is_wordpy_similar(in_word1_uni, in_word2_uni, strip_same_word = True):
    #连续汉字字符串拼音是否相似
    if len(in_word1_uni) != len(in_word2_uni):
        return False

    if strip_same_word:
        begin, end = 0, len(in_word1_uni) - 1
        while begin < end and in_word1_uni[begin] == in_word2_uni[begin]:
            begin += 1
        while begin < end and in_word1_uni[end] == in_word2_uni[end]:
            end -= 1
        if begin >= len(in_word1_uni) or end < 0:
            return False
        word1_uni = in_word1_uni[begin : end + 1]
        word2_uni = in_word2_uni[begin : end + 1]
    else:
        word1_uni = in_word1_uni
        word2_uni = in_word2_uni

    word1_py_list = []
    word2_py_list = []
    for i in range(len(word1_uni)):
        py_list1 = CHAR_MULTI_PYS.get(word1_uni[i], []) 
        word_py = get_word_py(word1_uni[i])
        if word_py[0] not in py_list1:
            py_list1.append(word_py[0])

        py_list2 = CHAR_MULTI_PYS.get(word2_uni[i], []) 
        word_py = get_word_py(word2_uni[i])
        if word_py[0] not in py_list2:
            py_list2.append(word_py[0])
        is_similar = False
        for py1 in py_list1:
            for py2 in py_list2:
                if is_py_similar(py1, py2) is True:
                    is_similar = True
                    break

        if is_similar is False:
            return False
    return True 

 
def get_word_py(word_unicode):
    # 返回拼音列表，每个元素对应字符串中每个汉字拼音
    ret = pinyin(word_unicode,  style=Style.NORMAL)
    word_py = []
    for i in ret:
        word_py.append(i[0])
    return word_py

def is_words_same_py(fst_uni, sec_uni):
    try:
        fst_py = get_word_py(fst_uni)
        sec_py = get_word_py(sec_uni)
    except Exception as e:
        return False
    return ''.join(fst_py) == ''.join(sec_py)

def is_py_same_sm_ym(py1, py2):
    init1, final1 = get_split_py(py1)
    init2, final2 = get_split_py(py2)
    sm, ym = False, False
    if init1 == init2 and init1 != '':
        sm = True
    if final1 == final2 and final1 != "":
        ym = True
    return sm, ym

def get_split_py(char_py):
    # 识别单字拼音中声母和韵母部分
    init, final = '', ''
    if len(char_py) == 1:
        final = char_py 
        return init, final
    if char_py[0 : 2] in SHENGMU:
        init = char_py[0 : 2]
        final = char_py[2 :]
    elif char_py[0] in SHENGMU:
        init = char_py[0]
        final = char_py[1 :]
    else:
        final = char_py
    return init, final

def merge_py_list_ele(lista, listb, is_strict=True):
    listc = []
    if len(lista) == 0 and len(listb) == 0:
        return listc
    elif len(lista) == 0:
        return listb
    elif len(listb) == 0:
        return lista
    for a in lista:
        listc.append("%s %s" %(a, listb[0]))
    for i, b in enumerate(listb):
        if i == 0: continue
        listc.append("%s %s" %(lista[0], b))
    return listc

#字符串相似音
def get_word_similar_py(words_unicode):
    char_py_list = get_word_py(words_unicode)
    if len(char_py_list) != len(words_unicode):
        return []
    py_list = []
    #单字相似音
    for i, char_py in enumerate(char_py_list):
        py_similar = get_char_similar_py(char_py)
        py_list.append(py_similar)

        multi_py = CHAR_MULTI_PYS.get(words_unicode[i], [])
        for t_py in multi_py:
            if t_py in py_similar:
                continue
            py_list[-1].append(t_py)

    #相似音组合
    new_list = []
    for i in range(len(char_py_list)):
        new_list = merge_py_list_ele(new_list, py_list[i])

    ret_list = []
    raw_py_str = ' '.join(char_py_list)
    for item in new_list:
        if item == raw_py_str:
            continue
        ret_list.append(item)

    return ret_list

def get_char_similar_py(char_py, is_strict = True):
    #单字相似音
    init, final = get_split_py(char_py)
    init_cand = CONFUSE_SM.get(init, '')
    final_cand = CONFUSE_YM.get(final, '')
    py_list = [init + final]
    if init_cand != "":
        py_list.append(init_cand + final)
    if final_cand != "":
        py_list.extend([init + i for  i in final_cand])
    if is_strict is False and init_cand != "" and final_cand != "":
        py_list.extend([init_cand + i for i in final_cand])

    other_py = CONFUSE_YIN.get(char_py, None)
    if other_py is not None and other_py not in py_list:
        py_list.append(other_py)
    return py_list

def char_py_diff(char_uni1, char_uni2):
    #单字与单字音diff
    ret = pinyin(char_uni1,  style=Style.NORMAL)
    char_py1 = ret[0][0].encode('utf-8')

    ret = pinyin(char_uni2,  style=Style.NORMAL)
    char_py2 = ret[0][0].encode('utf-8')

    init1, final1 = get_split_py(char_py1)
    init2, final2 = get_split_py(char_py2)
    sm_same = init1 == init2
    ym_same = final1 == final2
    py_simi = is_py_similar(char_py1, char_py2)
    return py_simi, sm_same, ym_same

if __name__ == "__main__":
    char1 = '像'
    char2 = '细'
    print (get_char_pronunciation_type(char1, char2))

    item = '强求'
    item2 = '抢救'
    print (is_wordpy_similar(item, item2))

    item = '地瞥'
    item_py_list = get_word_py(item)
    print (item_py_list)

    item = '衫'
    item_py_list2 = get_word_py(item)
    similar_py_list2 = get_word_similar_py(item)
    print (item_py_list2)
    print (similar_py_list2)
    sys.exit()
