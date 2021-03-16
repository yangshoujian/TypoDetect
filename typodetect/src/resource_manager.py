# -*- coding:utf-8
import os
import logging
import logging.config
import configparser
import gensim
import kenlm
import fasttext
from pypinyin import pinyin, Style
from .wordseg import Segment
from ..utils.dict_utils import load_key_values_dict, load_key_set
from ..utils.dict_utils import load_key_value_dict, load_line_set
from ..utils.dict_utils import load_keys_values_dict
from ..utils.dict_utils import load_fixed_pairs_dict_py
from ..utils.dict_utils import load_other_ners_dict, load_vdo_name_dict
from ..utils.io_utils import load_pkl
# from ..utils.time_utils import timer


class ResourceManager():

    def __init__(self, cfg_path):
        self.cfg = configparser.ConfigParser() # 初始化。ConfigParser 是用来读取配置文件的包。配置文件的格式如下：中括号“[ ]”内包含的为section。section下面为类似于key-value的options内容。例如：[config] server_mode = 0
        self.cfg.read(cfg_path) # 读取配置文件名

        server_mode = self.cfg.getint('config', 'server_mode')
        data_path = self.cfg.get('config', 'dict_path') # 获得指定sections的option的信息
        bigdict_path = self.cfg.get('config', 'big_dict')

        guwen_path = os.path.join(bigdict_path, 'uniq_autotune_model_uniform_800M_v7.bin')
        logging.info('loaddict: guwen_model_path:%s' % (guwen_path))
        self.guwen_classifier = fasttext.load_model(guwen_path) # 加载现有的fastText训练模型，path现有模型的路径

        logging.info('loaddict: w2v_model')
        self.w2v_mode = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(data_path, 'w2v_model.bin'), binary=True, unicode_errors='ignore')

        lm_path = os.path.join(bigdict_path, 'ken_lm_120G.bin')
        logging.info('loaddict: language_model path:%s' % (lm_path))
        # lm_path = os.path.join(data_path, 'people_chars_lm.klm')
        self.ngram_lm = kenlm.LanguageModel(lm_path) # 使用一个kenlm的python包去训练一个语言模型，并对每个句子进行打分。

        # name dict
        self.wrong_name = load_key_value_dict(
            os.path.join(data_path, 'name_dict/wrong_name.csv'), ',')
        self.vdo_namepy_name = load_vdo_name_dict(
            os.path.join(data_path, 'name_dict/vdo_name_pinyin'))
        self.all_name_pinyin = load_key_values_dict(
            os.path.join(data_path, 'name_dict/mingxing_names_pinyin'),
            '\t', '\t')
        self.other_ners_pinyin = load_other_ners_dict(
            os.path.join(data_path, 'name_dict/other_ners_pinyin'))
        self.loc_baidu_wrong = load_key_value_dict(
            os.path.join(data_path, 'name_dict/loc_baidu_wrong.csv'), ',')
        self.name_gender = load_key_value_dict(
            os.path.join(data_path, 'name_dict/name_gender'))

        # default utf-8
        self.sig_char = load_key_set(os.path.join(
            data_path, 'single_char_score'), 100)
        self.word_freq = load_key_value_dict(
            os.path.join(data_path, 'uni_wordfreq_dict'), ':')  # uni_word_dict

        self.chengyu_pinyin_dict = load_key_values_dict(
            os.path.join(data_path, 'chengyu_pinyin'))
        self.prefix_suffix_chengyu = load_key_values_dict(
            os.path.join(data_path, 'same_presuffix_chengyu'))
        self.danzi_dict = load_key_values_dict(
            os.path.join(data_path, 'danzi_dict'))
        self.before_after_char_dict = load_key_values_dict(
            os.path.join(data_path, 'before_after_zi_dict'))
        # 形近
        self.similar_stroke_dict = load_keys_values_dict(
            os.path.join(data_path, 'same_stroke.txt'), ',')
        # 共现数据
        self.fixed_pairs = load_fixed_pairs_dict_py(
            os.path.join(data_path, 'head_tail_fixed_py'))
        self.top_dict = load_key_value_dict(
            os.path.join(data_path, 'top_word_40000'))
        self.pass_words = load_line_set(os.path.join(data_path, 'pass_words'))
        self.pass_chars = load_line_set(os.path.join(data_path, 'pass_chars'))
        self.corr_words = []
        self.corr_chars = []
        self.one_to_multi = load_key_values_dict(
            os.path.join(data_path, 'one_to_multi_dict'), '\t', ' ')
        if server_mode == 0:
            self.commonword_py_dict = load_key_values_dict(os.path.join(data_path, 'word_kb_top8w_withcommon_py'))
            self.same_pinyin_ci_zi = load_pkl(os.path.join(data_path, 'same_pinyin_ci_zi_filter2.pkl'))
        else:
            self.same_pinyin_ci_zi = load_pkl(os.path.join(data_path, 'same_pinyin_ci_zi.txt.filter.pkl'))
            self.commonword_py_dict = load_key_values_dict(os.path.join(data_path, 'wordge2_kb_py'))
        self.jinyin_zi = load_pkl(os.path.join(data_path, 'buchong_jinyin_zi_bidirection_new.pkl'))

        self.jinyin_ci_confu_filter = load_pkl(os.path.join(data_path, 'bc_conf_v2.0_bid.pkl'))

        self.term_imp = load_key_value_dict(os.path.join(data_path, 'term_imp.txt'))

        if server_mode == 0:
            logging.info('load singledict end')
            return
        # for multipath
        self.commonchar_py_dict = load_key_values_dict(
            os.path.join(data_path, 'char_common_all_py_withmlti'))
        self.single_char_multipy_dict = load_key_values_dict(
            os.path.join(data_path, 'hanyu_single_muliti_py'))
        self.commonphrase_py_dict = load_key_values_dict(
            os.path.join(data_path, 'phrase_common_py'))
        self.commonfrag_py_dict = load_key_values_dict(
            os.path.join(data_path, 'frag_common_with_top100_py'))
        self.commonerror_dict = load_key_values_dict(
            os.path.join(data_path, 'confusion_wrong_right_fromhumanlb'),
            ':', ' ')
        logging.info('load multipathdict end')
        # for multipath deprecated
        self.jinyin_ci = load_pkl(
            os.path.join(data_path,
                         'word2_kb_near.pkl'))
        self.cooc_dict = load_key_values_dict(
            os.path.join(data_path, 'cooc_dict'))
        self.prefix_suffix_dict = load_key_values_dict(
            os.path.join(data_path, 'same_presuffix_word2_kb_top8w'))
        # not used below
        self.tongyin_ci = load_pkl(os.path.join(data_path, 'tongyin_ci.pkl'))
        self.jinyin_ci_confu = load_pkl(
            os.path.join(data_path, 'confusion_set_ci_v1.0_filter.pkl'))

    def get_ppl_score(self, words_list):
        gbk_str = ' '.join(words_list).encode('gb18030')
        score = self.ngram_lm.perplexity(gbk_str)
        return score

    def get_one_to_multi_cand(self, word_unicode):
        return self.one_to_multi.get(word_unicode, [])

    def get_danzi_cand(self, char_unicode):
        return self.danzi_dict.get(char_unicode, [])

    def get_before_after_char_cand(self, char_unicode):
        return self.before_after_char_dict.get(char_unicode, [])

    def get_cooc_cand(self, wordstr_unicode, is_suffix=0):
        key = wordstr_unicode
        if is_suffix == 1:
            key = '_' + wordstr_unicode
        return self.cooc_dict.get(key, [])

    def get_common_error_cand(self, word_unicode):
        return self.commonerror_dict.get(word_unicode, [])

    def get_prefix_suffix_cand(self, words_unicode):
        cands_list = []
        prefix = words_unicode[0]
        suffix = "_" + words_unicode[-1]
        cands_list.extend(self.prefix_suffix_dict.get(prefix, []))
        cands_list.extend(self.prefix_suffix_dict.get(suffix, []))
        return cands_list

    def get_prefix_suffix_chengyu_cand(self, words_unicode):
        cands_list = []
        if len(words_unicode) <= 2:
            return cands_list
        prefix = words_unicode[0: 2]
        suffix = "_" + words_unicode[-2:]
        prefix_cands = self.prefix_suffix_chengyu.get(prefix, [])
        suffix_cands = self.prefix_suffix_chengyu.get(suffix, [])
        cands_list.extend(prefix_cands)
        cands_list.extend(suffix_cands)
        return cands_list

    def get_same_py_chengyu(self, pinyin_str):
        return self.chengyu_pinyin_dict.get(pinyin_str, [])

    def is_words_oov(self, words_gbk):
        ret = []
        for word in words_gbk:
            if word in self.ngram_lm:
                ret.append((word, 0))
            else:
                ret.append((word, 1))
        return ret

    def is_similar_word(self, word_fst, word_sec, threshold=0.5):
        if word_fst in self.w2v_mode.vocab and word_sec in self.w2v_mode.vocab:
            sim = self.w2v_mode.similarity(word_fst, word_sec) # word2vec的余弦相似度
            if sim > threshold: # 大于阈值才认为相似
                logging.info('word_similar: %s-%s' % (word_fst, word_sec))
                return True
        return False

    def get_same_pinyin_ci_zi(self, word):
        return self.same_pinyin_ci_zi.get(word, set())

    def get_similar_stroke(self, word):
        return self.similar_stroke_dict.get(word, [])

    def is_stroke_similar(self, word1, word2):
        return word2 in self.similar_stroke_dict.get(word1, [])

    def get_samepy_common_char(self, word_py):
        return self.commonchar_py_dict.get(word_py, [])

    def get_samepy_common_char_withchar(self, word_unicode):
        ret = pinyin(word_unicode, style=Style.NORMAL)
        word_py = ret[0][0]
        return self.commonchar_py_dict.get(word_py, [])

    def get_samepy_common_word(self, word_py):
        return self.commonword_py_dict.get(word_py, [])

    def get_samepy_common_frag(self, word_py):
        return self.commonfrag_py_dict.get(word_py, [])

    def get_singlechar_multi_py(self, word_unicode):
        return self.single_char_multipy_dict.get(word_unicode, [])

    def get_samepy_common_phrase(self, word_py):
        return self.commonphrase_py_dict.get(word_py, [])

    def get_tongyin_ci_zi(self, word):
        confusion_word_set = self.get_same_pinyin_ci_zi(word)
        if not confusion_word_set:
            confusion_word_set = set()
        return confusion_word_set

    def get_jinyin_zi_(self, word):
        return self.jinyin_zi.get(word, set())

    def get_jinyin_zi(self, word):
        confusion_word_set = self.get_jinyin_zi_(word)
        if not confusion_word_set:
            confusion_word_set = set()
        return confusion_word_set

    def get_jinyin_ci_(self, word):
        return self.jinyin_ci.get(word, set())

    def get_jinyin_ci(self, word):
        confusion_word_set = self.get_jinyin_ci_(word)
        if not confusion_word_set:
            confusion_word_set = set()
        return confusion_word_set

    def get_jinyin_ci_conf_(self, word):
        return self.jinyin_ci_confu.get(word, set())

    def get_jinyin_ci_conf(self, word):
        confusion_word_set = self.get_jinyin_ci_conf_(word)
        if not confusion_word_set:
            confusion_word_set = set()
        return confusion_word_set

    def get_jinyin_ci_conf_ft_(self, word):
        return self.jinyin_ci_confu_filter.get(word, set())

    def get_jinyin_ci_conf_ft(self, word):
        confusion_word_set = self.get_jinyin_ci_conf_ft_(word)
        if not confusion_word_set:
            confusion_word_set = set()
        return confusion_word_set

    def get_same_pinyin_ci_new(self, word):
        return self.tongyin_ci.get(word, set())

    def get_tongyin_ci_new(self, word):
        confusion_word_set = self.get_same_pinyin_ci_new(word)
        if not confusion_word_set:
            confusion_word_set = set()
        return confusion_word_set

    # reload dict if needed, else pass
    def reload_dict(self):
        return True


resource = ResourceManager('etc/config.ini')
# resource.load_dict()

# some object cann't be serialized in muliprocess
logging.info("dict_path:%s" % (resource.cfg.get('config', 'seg_dict')))
SEQ = Segment(resource.cfg.get('config', 'seg_dict'))


def seg_basic(content, qtype=1):
    handle = SEQ.segment(content)
    basic_result = SEQ.get_basic_words(handle)
    if qtype == 1:
        mix_result = SEQ.get_mix_words(handle)
        ner_result = SEQ.get_ner_names(handle)
    else:
        mix_result, ner_result = [], []
    return basic_result, mix_result, ner_result


logging.info('resourcemanager load_dict_end')
