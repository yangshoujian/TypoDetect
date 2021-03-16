# -*- coding:utf-8 -*-

import os
from utils.dict_utils import load_keys_values_dict

similar_stroke_dict = load_keys_values_dict('data/same_stroke.txt', ',')
def is_stroke_similar(word1, word2):
    return word2 in similar_stroke_dict.get(word1, [])
