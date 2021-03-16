# -*- coding:utf-8

import logging
import heapq
from enum import Enum, unique
from ..utils.py_utils import get_word_py, get_word_similar_py

MAX_DOC_LENGTH = 2000
MAX_SENTENCE_LENGTH = 150
MAX_SENTENCE_COUNT = 70


class Topk():
    def __init__(self, k, key=lambda x: x[0]):
        self.k = k
        self.data = []
        self.key = key
    '''
    def push(self, elem):
        heapq.heappush(self.data, (self.key(item), item))
    '''

    def push(self, elem):
        # pplscore, word_list, stack_item_info
        elem = (-elem[0], elem[1], elem[2])
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem) # 往堆中插入一条新的值
        else:
            top_small = self.data[0][0]
            if elem[0] > top_small:
                heapq.heapreplace(self.data, elem)

    def get_top_k(self):
        lst, ret = [], []
        for _ in range(len(self.data)):
            lst.append(heapq.heappop(self.data))
        for item in reversed(lst):
            ret.append((-item[0], item[1], item[2]))


class ClassifyResult():
    def __init__(self):
        # 存储NN模型识别结果
        self.is_judged = False
        self.is_sent_wrong = False
        self.sent_prob = 0.0  # 序列标注模型预测概率
        self.classify_prob = 0.0  # 分类模型预测概率：用于视频标题高准版


@unique
class ECType(Enum):
    CHENGYU_SAMEPY = 0
    SINGLECHAR_NONWORD = 1 # 单字非词——同音字
    SINGLECHAR_NONWORD_JY = 51 # 单字非词——近音字
    SINGLECHAR_WORD = 2 # 单字词——同音字
    SINGLECHAR_WORD_JY = 52 # 单字词——近音字
    WORD_SAMEPY = 3 # 同音词
    WORD_SIMILARPY = 4 # 近音词
    WORD_BERT = 5 # 词语判别模型
    SINGLECHAR_WORD_BERT = 6 # 单字判别模型
    NER_PERSON = 7 # 人名识别
    NER_PERSON_RECALL = 57
    MULTICHAR = 8  # 连续单字组成同音词——多对一
    NER_OTHER = 9 # 其他专名识别
    FIXED_ERROR = 10  # 词语固定搭配
    BEFORE_CHAR = 11  # 连续单字定一换一搭配
    MULTICHAR_BERT = 12 # 多对一判别模型
    BEFORE_BERT = 13 # 定一换一判别模型
    CHENGYU_SIMILAR = 15
    ONE_TO_MULTI = 19 # 一对多
    NNLM_1 = 17  # NNLM召回固定搭配
    SEQ_ADD = 20 # 序列标注
    NNLM_2 = 22  # NNLM扩大候选
    PRONOUN = 23  # 代词 他、她
    VTITLE_CLASSIFY = 24  # 视频标题分类高准版

    # 以下为多路径策略召回类型
    SINGLECHAR_NONWORD_SAMEFIX = 1107
    SINGLECHAR_WORD_XJ = 1108
    SUSPECT_SAME_PRESUFFIX = 1110
    SUSPECT_NGRAM_PY_WORD = 1111
    SUSPECT_NGRAM_PY_PHRASE = 1112
    SUSPECT_NGRAM_PY_FRAG = 1113
    SUSPECT_PY = 1114
    SUSPECT_SPLIT_CROSS_WORD = 1115   # AB C -> A BC
    SUSPECT_SPLIT_CROSS_FRAG = 1116   # AB C -> A B C
    SUSPECT_SPLIT_FRAG = 1117   # AB -> A B
    SUSPECT_SPLIT_CHAR = 1118   # 词中单字替换为另个同音单字
    SUSPECT_SPLIT_STROKECHAR = 1119   # 词中单字替换为另个形近单字
    SINGLECHAR_WORD_YJ = 1120
    COMMON_ERROR = 1121
    COOCCURRENCE = 1122


class Stack():
    """栈"""

    def __init__(self):
        self.items = []

    def is_empty(self):
        """判断是否为空"""
        return len(self.items) == 0

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items) - 1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)


class DocInfo():

    def __init__(self, doc_id):
        # 存储文章信息
        self.doc_id = doc_id
        self.ner_list = [[], [], [], []]  # all ner-list seq
        # [{name:cnt}, {location:cnt}, {org:name}, {vdo:name}]
        self.ner_cnt = [None, None, None, None] # 统计文章整体信息：如人名出现频次等
        self.ners = []  # unique list: [[name], [location], [org], [vdo]]
        self.ner_all = []  # [ner, ner2, ...]
        self.name_detect_res = {}


class FlyWeight():
    # 长句的信息
    def __init__(self, sentence_str, doc_info, sent_no=-1,
                 para_id=-1, sent_id=-1, doc_id=-1,
                 channel_id=0, data_type=0, bid=''):
        self.sentence_str = sentence_str
        # start from 0:title 1:first_paragraph
        self.paragraph_id = para_id
        # start from 0: split by '。！？?'
        self.sentence_index = sent_id
        # 与旧版兼容、全文从0开始
        self.sent_no = sent_no # 句子在原文中的编号
        self.doc_id = doc_id
        self.doc_info = doc_info
        # 一级分类ID
        self.channel_id = channel_id
        # 0-图文 1-短视频 2-小视频
        self.data_type = data_type
        self.bid = bid
        self.logid = "%s_%s_%d" % (bid, doc_id, sent_no)

        # default utf8
        self.basic_list = []
        self.basic_pos = []
        self.mix_list = []
        self.basic_list_unicode = []
        # example: 北京 大学  word_pos_idx: 0 2  check_flags: 1 1
        # 基本词粒度：每个词在原句中的字偏移
        self.word_pos_idx = []
        # 基本词粒度
        self.check_flags = []  # -4:<>  -2:标点非汉字 -1:专名 0:已改 1:待检测
        self.char_check_flags = []
        self.ners_unicode = []
        self.ners_info = []

        # 句子错误概率
        self.prob_wrong = 0.0

        # 短句SentInfo 列表
        self.shortsent_infos = []

        # DetectErro 对外接口
        self.error_infos = []

        # for debug
        self.correct_sent = ""
        self.wrong_frag = ""
        self.right_frag = ""
        self.has_cand = 0   # 有候选
        self.has_add = 0   # 加入召回
        self.is_debug = False
        self.debug_info = {}
        #

    # begin_idx: offset in wordlist
    # frag_len: correct fragment length
    def adjust_check_flags(self, begin_idx, wordlist, frag_len, check_flags):
        cur_idx = begin_idx
        tmp_len = len(wordlist[cur_idx])
        while tmp_len <= frag_len:
            check_flags[cur_idx] = 0
            cur_idx += 1
            tmp_len += len(wordlist[cur_idx])

    def __str__(self):
        pos_str = ' '.join([str(i) for i in self.word_pos_idx])
        check_str = ' '.join(str(int(i)) for i in self.check_flags)
        ner_str = ' '.join(self.ners_unicode)
        log_str = ('\nraw_str:%s para_id:%d sent_idx:%d  \n'
                   'basic_seg:%s \n mix_seg:%s \n word_pos_idx:%s'
                   '\n check_flags:%s \n ner_str:%s'
                   % (self.sentence_str, self.paragraph_id,
                      self.sentence_index,
                      ' '.join(self.basic_list), ' '.join(self.mix_list),
                      pos_str, check_str, ner_str))
        return log_str

    def get_error_infos(self):
        return self.error_infos

    def set_resource(self, resource):
        self.resource = resource

    def get_sentence_info(self):
        return self.shortsent_infos


class SentInfo():
    # 短句信息： 长句以逗号分隔的短句
    def __init__(self, basic_list_unicode, basic_words_score, basic_norm_unicode, check_flags, basic_pos, word_pos_idx, punc_list, ppl, ss_idx, debug_mode=0):
        # for debug
        self.correct_sent = ''.join(basic_list_unicode)
        # normed basic list
        self.basic_norm_unicode = basic_norm_unicode
        self.basic_norm_score = basic_words_score
        self.suspect_flag = []

        # 链接矩阵的形式存储原句及所有候选的图结构
        # cands_list[i][0] 为原句分词后第i个词信息
        # cands_list[i][i : k] 为第i个词的所有正确候选
        self.cands_list = []
        self.py_list = []

        # raw basic list
        self.basic_list_unicode = basic_list_unicode
        self.basic_pos = basic_pos
        # 词粒度: 每个词原长句中的flag
        self.check_flags = check_flags
        # 基本词粒度: 每个词在原长句中的字偏移
        self.word_pos_idx = word_pos_idx
        # 0:标点 1:词典汉字词 2:非汉字词数词等 3:oov
        self.punc_list = punc_list
        self.raw_ppl = ppl
        # 短句在长句中的位置
        self.short_idx = ss_idx
        self.raw_to_norm = {}
        self.norm_to_raw = {}

        # 纠错topk候选句
        self.topk_cand_list = []
        self.final_cand_rank = []

        # 与旧架构策略兼容，记录不同类型错别个数:
        # idx:0(成语、专名、固定搭配类错误）
        # idx:1(其他nnlm等识别个数)
        self.error_cnt = [0, 0]
        # 错误片段
        self.error_infos = []

        self.debug_info = {'add_cands': [], 'cands': {},
                           'final_sents': [], 'frag_cands': {}}
        self.debug_mode = debug_mode

        self.initialize()

    def initialize(self):
        j = 0
        for i, item in enumerate(self.basic_list_unicode):
            if self.punc_list[i] == 0:
                continue
            self.raw_to_norm[i] = j
            self.norm_to_raw[j] = i
            cand_item = CandItem(item, j, j)
            self.cands_list.append([cand_item])

            if self.punc_list[i] != 2:
                item_py_list = get_word_py(item)
                self.py_list.append([' '.join(item_py_list)])
                if len(item) < 2:
                    similar_py_list = get_word_similar_py(item)
                    self.py_list[j].extend(similar_py_list)
            else:
                # 非汉字
                self.py_list.append(['<upy>'])

            j += 1
        self.suspect_flag = [0] * len(self.basic_list_unicode)

    def add_cand(self, in_begin, in_end, correct_str_unicode, correct_type, ppl_ratio=0.0, is_norm=False, is_exempt=False):
        if is_exempt: # 旧架构兼容、豁免一些字词
            return
        if is_norm is False:
            (basic_begin, basic_end) = (self.raw_to_norm[in_begin], self.raw_to_norm[in_end])
        else:
            basic_begin, basic_end = in_begin, in_end

        # 归一化后的词粒度偏移
        for item in self.cands_list[basic_begin]:
            if (item.basic_end == basic_end and item.cand_str == correct_str_unicode):
                return
        cand_item = CandItem(correct_str_unicode, basic_begin, basic_end, correct_type.value, ppl_ratio)
        self.cands_list[basic_begin].append(cand_item)
        if self.debug_mode & 2 == 2:
            tmp_typ = -1
            if in_begin == in_end:
                if in_begin not in self.debug_info['cands']:
                    self.debug_info['cands'][in_begin] = []
                self.debug_info['cands'][in_begin].append((in_begin, in_end, correct_str_unicode, correct_type.value, ppl_ratio, 1))
                tmp_typ = 0
            else:
                if in_begin not in self.debug_info['frag_cands']:
                    self.debug_info['frag_cands'][(in_begin, in_end)] = []
                self.debug_info['frag_cands'][(in_begin, in_end)].append((in_begin, in_end, correct_str_unicode, correct_type.value, ppl_ratio, 1))
                tmp_typ = 1
            logging.info('add_debug %d-%d cand:%s type:%d ratio:%.4f'
                         ' isv:1 add_tp:%d'
                         % (in_begin, in_end, correct_str_unicode, correct_type.value, ppl_ratio, tmp_typ))
        # 记录类型错误个数
        if (correct_type in [ECType.CHENGYU_SAMEPY, ECType.FIXED_ERROR, ECType.NER_PERSON, ECType.NER_OTHER]): # 0-成语识别，10-词语固定搭配，7-人名识别，9-其他专名识别
            self.error_cnt[0] += 1
        else:
            self.error_cnt[1] += 1

    def set_check_flags(self, begin, end, value=0): # 设为已改
        for i in range(begin, end + 1):
            self.check_flags[i] = value


class CandItem(): # 纠错候选信息
    def __init__(self, cand_str, basic_begin, basic_end=0, cand_type=-1, ppl_ratio=0.0):
        # unicode编码
        self.cand_str = cand_str # 候选
        #
        self.basic_begin = basic_begin
        self.basic_end = basic_end
        # 候选类型: -1原词, 0成语  1单字非词  2单字 3同义词 4近义词
        self.cand_type = cand_type
        # for debug
        self.ppl_ratio = ppl_ratio


class ErrorInfo():

    # 短句内错误信息记录
    def __init__(self, wrong_frag_str, correct_frag_str, sent_idx,
                 in_sent_ci_pos_begin, in_sent_ci_pos_end, error_type,
                 prob=0.0, in_sent_char_pos_begin=-1,
                 in_sent_char_pos_end=-1, term_imp=0):

        self.wrong_frag = wrong_frag_str
        self.correct_frag = correct_frag_str
        self.sent_index = sent_idx
        # basic granularity pos
        self.sent_ci_pos_begin = in_sent_ci_pos_begin
        self.sent_ci_pos_end = in_sent_ci_pos_end
        # unicde char pos
        self.sent_char_pos_begin = in_sent_char_pos_begin
        self.sent_char_pos_end = in_sent_char_pos_end
        self.error_type = error_type
        self.prob = prob
        # 错字词重要性:[0:不重要,1:一般,2:重要]
        self.term_imp = term_imp


class DetectError():
    # 最终返回信息接口
    def __init__(self, doc_id, wrong_frag, correct_frag, short_index,
                 long_index, long_str, short_str, idx_begin, idx_end,
                 error_type, prob, uni_begin, uni_end):
        self.doc_id = doc_id
        # 长句在文章中偏移
        self.long_index = long_index
        # 短句在长句中偏移
        self.short_index = short_index
        # 归一化短句basic_norm 中偏移[begin, end]
        self.idx_begin = idx_begin
        self.idx_end = idx_end
        # 原始长句中的字偏移
        self.unichar_begin = uni_begin
        self.unichar_end = uni_end
        # error_str
        # 长句index
        # 长句原句
        # 短句原句
        self.wrong_frag = wrong_frag
        self.correct_frag = correct_frag
        self.error_type = error_type
        self.prob = prob
        self.long_str = long_str
        self.short_str = short_str

    def __str__(self):
        log_str = ('[detect_error_info] doc_id:%s[%d] basic_pos[%d-%d]'
                   ' error_type:[%d] prob:[%.4f] wfrag:[%s] cfrag:[%s] '
                   ' sstr:[%s] lstr:[%d-%d][%s]'
                   % (self.doc_id, self.long_index, self.idx_begin,
                      self.idx_end, self.error_type, self.prob,
                      self.wrong_frag, self.correct_frag, self.short_str,
                      self.unichar_begin, self.unichar_end, self.long_str))
        return log_str
