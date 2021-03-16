# -*- coding:utf-8 -*-

import sys
from collections import Counter
from pypinyin import lazy_pinyin
from .flyweight import MAX_DOC_LENGTH, MAX_SENTENCE_LENGTH
from .flyweight import MAX_SENTENCE_COUNT
from .flyweight import FlyWeight, DocInfo
from ..utils.text_utils import split_para, split_sentence, clean
from ..utils.text_utils import sentence_is_valid, wordnum
from .resource_manager import seg_basic
from .resource_manager import resource as rs
from ..utils.time_utils import timer


class PrePostProcessor():

    def __init__(self):
        pass

    def pre_process(self, title, content, doc_id='', channel_id=0, data_type=0, bid=''):
        tms = timer(sys._getframe().f_code.co_name, doc_id)
        title = clean(title)
        content = clean(content)
        sentence_flyweights = []
        doc_info = DocInfo(doc_id)
        if title != "":
            sno = 0
            title = title.replace(" ", "").replace("　", "")
            title_fly = FlyWeight(title, doc_info, sno, 0, 0, doc_id, channel_id, data_type, bid)
            sno += 1
            is_suc = self.process_sentence(title_fly)
            if is_suc:
                sentence_flyweights.append(title_fly)
            else:
                pass
        if content != "":
            doc_len, sent_cnt, sno = 0, 0, 1
            content_paras = split_para(content)
            for i, para in enumerate(content_paras):
                para_sents = split_sentence(para)
                doc_len += wordnum(para)
                if doc_len > MAX_DOC_LENGTH: # MAX_DOC_LENGTH=2000
                    break
                for sent in para_sents:
                    sent = sent.replace(" ", '')
                    if sentence_is_valid(sent):
                        sent_cnt += 1
                        sent_fly = FlyWeight(sent, doc_info, sno, i + 1, sent_cnt, doc_id, channel_id, data_type, bid)
                        sno += 1
                        is_suc = self.process_sentence(sent_fly)
                        if is_suc:
                            sentence_flyweights.append(sent_fly)
                        else:
                            pass
                if sno > MAX_SENTENCE_COUNT:
                    break
        self.post_process(doc_info)
        self.post_process_name_gender(sentence_flyweights)
        del tms
        return sentence_flyweights

    def process_sentence(self, sent_fly):
        # 分词、识别NER，对ner字词打上处理豁免标记
        basic_result, _, ners = seg_basic(sent_fly.sentence_str, 1)
        if len(basic_result) == 0:
            return False
        basic_words, basic_pos, _ = zip(*basic_result)
        ners_uni = set([i[0] for i in ners])
        basic_words_unicode = [w for w in basic_words[0: MAX_SENTENCE_LENGTH]]
        word_pos_idx = []
        pos_i = 0
        for word in basic_words_unicode:
            word_pos_idx.append(pos_i)
            pos_i += len(word)
        total_char_count = pos_i
        sent_fly.basic_list = basic_words[0: MAX_SENTENCE_LENGTH]
        sent_fly.basic_list_unicode = basic_words_unicode
        sent_fly.basic_pos = basic_pos[0: MAX_SENTENCE_LENGTH]
        sent_fly.word_pos_idx = word_pos_idx
        sent_fly.ners_unicode = ners_uni
        sent_fly.ners_info = ners

        doc_info = sent_fly.doc_info
        for word, sidx, eidx, ner_type in ners:
            if ner_type < 4:
                doc_info.ner_list[ner_type].append(word)

        # set flag
        basic_word_cnt = len(basic_words_unicode)
        check_flags = [1] * basic_word_cnt
        char_check_flags = [1] * len(''.join(basic_words_unicode))

        for word, sidx, eidx, ner_type in ners:
            if sidx >= basic_word_cnt:
                continue
            char_sidx = word_pos_idx[sidx]
            if eidx >= basic_word_cnt:
                char_eidx = total_char_count
            else:
                char_eidx = word_pos_idx[eidx]
            for j in range(char_sidx, char_eidx):
                char_check_flags[j] = -1
            for i in range(sidx, eidx):
                if i >= MAX_SENTENCE_LENGTH:
                    break
                if ner_type == 0:
                    check_flags[i] = -2
                else:
                    check_flags[i] = -1

        sent_fly.check_flags = check_flags
        sent_fly.char_check_flags = char_check_flags
        return True

    def post_process(self, doc_info):
        # 记录文章层面信息
        ner_list = doc_info.ner_list # ner_list = [[], [], [], []]  # all ner-list seq
        ners = doc_info.ners # unique list: [[name], [location], [org], [vdo]]
        for idx, item in enumerate(ner_list):
            ners.append(list(set(item)))
            doc_info.ner_cnt[idx] = Counter(item) # 计数器
        other_detect_res = self.get_other_ner_err_dict(ners, ner_list)
        per_detect_res = self.get_name_err_dict(ners, ner_list[0])
        name_detect_res = {**other_detect_res, **per_detect_res}
        doc_info.name_detect_res = name_detect_res

        ner_all = []
        for item in ners:
            ner_all.extend(item)
        doc_info.ner_all = ner_all

    def post_process_name_gender(self, sentence_flyweights):
        # 人称代词识别策略: 统计句子中人名、泛性别词语信息，指定句子
        # 人称性别，若句子中存在与句子人称性别相冲突的代词，则修改
        idx_map = {1: '他', 2: '她'}
        pre_sent_gender = -1
        for fly_weight in sentence_flyweights:
            long_sent_list = fly_weight.basic_list_unicode
            long_sent_raw = ''.join(long_sent_list)
            check_flags = fly_weight.check_flags
            check_flags_len = len(check_flags)
            sent_no = fly_weight.sent_no
            # 统计句子性别相关信息:
            # 记录非人名性别信息、 人名数量
            extra_info, per_cnt = [0, 0], 0
            # 记录不同代词数量、去重人名信息
            idx_cnt, namedict, names_info = [0, 0], {}, []
            for i in range(len(long_sent_list)):
                cur_word = long_sent_list[i]
                if (i < check_flags_len and check_flags[i] == -2):
                    # 人名对应性别
                    if cur_word not in namedict:
                        per_cnt += 1
                        cand_gender = rs.name_gender.get(long_sent_list[i], -1)
                        if cand_gender != -1:
                            names_info.append(int(cand_gender))
                        namedict[cur_word] = 1
                elif len(cur_word) == 2:
                    # 非人名其他隐性性别
                    cand_gender = int(rs.name_gender.get(
                        long_sent_list[i], '-1'))
                    if cand_gender in [3, 4]:
                        extra_info[cand_gender % 3] += 1
                if cur_word == '他':
                    idx_cnt[0] += 1
                elif cur_word == '她':
                    idx_cnt[1] += 1

            if (len(names_info) > 1 or per_cnt > 1 or (len(names_info) == 0 and pre_sent_gender == -1) or (extra_info[0] > 0 and extra_info[1] > 0)):
                # 多个人名、多种隐性性别、正文句无人名无显性性别
                pre_sent_gender = -1
                continue
            if len(names_info) == 0:
                sent_gender = pre_sent_gender
                pre_sent_gender = -1
            else:
                sent_gender = names_info[0]
                pre_sent_gender = sent_gender
            sum_idx_cnt = sum(idx_cnt)

            # 正文句子全部为他/她不处理(标题除外)
            if (sum_idx_cnt == idx_cnt[0] or sum_idx_cnt == idx_cnt[1]):
                continue
            if ((sent_gender == 1 and extra_info[1] > 0)
                    or (sent_gender == 2 and extra_info[0] > 0)):
                # 含有其他隐含性别词且冲突
                pre_sent_gender = -1
                continue

            # 策略1. 标题中无句子代表性别、无隐形性别词，则一律改为"它"
            # 策略2: 含有他/她不只一种，2.句子有代表性别，3.无与句子代表性别冲突的隐形性别
            if sent_gender in [1, 2]:
                correct_frag = idx_map[sent_gender]
            else:
                continue

            word_pos_idx = fly_weight.word_pos_idx
            err_infos = []
            for j in range(len(long_sent_list)):
                word = long_sent_list[j]
                if word in ['他', '她'] and word != correct_frag:
                    word_pos = word_pos_idx[j]
                    if fly_weight.paragraph_id <= 1:
                        term_imp = 2
                    else:
                        term_imp = 1
                    err_infos.append((sent_no, word_pos, long_sent_raw,
                                      word, correct_frag, 23, 1,
                                      long_sent_raw, term_imp, 0))
            fly_weight.error_infos = err_infos

    def get_name_err_dict(self, ners, per_list):

        vdo_namepy_name = rs.vdo_namepy_name # akb调查    tutianhuangzhi|||kuchuanxiangnai|||zhiyuanlinai 土田晃之|||堀川香奈|||指原莉乃  2014    TV
        all_name_pinyin = rs.all_name_pinyin # wangjun  王珺  王军  汪俊  王俊
        wrong_name = rs.wrong_name # 张蔓玉 张曼玉
        pinyin_name_dict = {}  # {pinyin: {word : count}}
        for name in per_list:
            name_uni = name
            name_pinyin = ''.join(lazy_pinyin(name_uni))
            if name_pinyin not in pinyin_name_dict:
                pinyin_name_dict[name_pinyin] = {name_uni: 1} # 计为1
            else:
                if name_uni in pinyin_name_dict[name_pinyin]:
                    pinyin_name_dict[name_pinyin][name_uni] += 1 # 拼英 名字 计数
                else:
                    pinyin_name_dict[name_pinyin][name_uni] = 1

        final_pinyin_name_dict = {}
        for pinyin, names in pinyin_name_dict.items():
            if len(names) < 2:
                continue
            names_sort = sorted(names.items(), key=lambda x: x[1])
            final_pinyin_name_dict[pinyin] = names_sort

        per_res, _, _, vdo_res = ners[0], ners[1], ners[2], ners[3]
        vdo_cand_names = []
        vdo_cand_names_dict = {}  # {py : namelist}
        for vdo in vdo_res:
            vdo = vdo
            if vdo in vdo_namepy_name:
                cand_names = vdo_namepy_name[vdo]
                vdo_cand_names.append(cand_names)

        for vdict in vdo_cand_names:
            for tmp_py in vdict.keys():
                if tmp_py not in vdo_cand_names_dict:
                    vdo_cand_names_dict[tmp_py] = vdict[tmp_py]

        all_mx_name_list = []
        for name_list in all_name_pinyin.values():
            all_mx_name_list.extend(name_list)
        all_mx_name_set = set(all_mx_name_list)

        type_baidu = '7_baidu'
        type_vdo = '7_vdo'
        type_freq = '7_freq'
        detect_res = {}
        for pn_uni in per_res:
            if pn_uni in detect_res.keys():
                continue
            if pn_uni in wrong_name.keys() and pn_uni != wrong_name[pn_uni]:
                detect_res[pn_uni] = [type_baidu, wrong_name[pn_uni]]
            else:
                p_pinyin = ''.join(lazy_pinyin(pn_uni))
                if p_pinyin in vdo_cand_names_dict:
                    if pn_uni not in vdo_cand_names_dict[p_pinyin]:
                        detect_res[pn_uni] = [type_vdo, vdo_cand_names_dict[p_pinyin][0]]
                elif p_pinyin in final_pinyin_name_dict:
                    p_freq_cand = final_pinyin_name_dict[p_pinyin]
                    same_cand_freq = []
                    for tmp_k, tmp_v in p_freq_cand:
                        if tmp_k in all_mx_name_set:
                            same_cand_freq.append((tmp_k, tmp_v))
                    if len(same_cand_freq) > 0:
                        cand_res = same_cand_freq[-1][0]
                    else:
                        cand_res = p_freq_cand[-1][0]
                    if pn_uni != cand_res:
                        detect_res[pn_uni] = [type_freq, cand_res]
        return detect_res

    def get_other_ner_err_dict(self, ners, ner_list):

        loc_baidu_wrong = rs.loc_baidu_wrong
        other_ners_pinyin = rs.other_ners_pinyin
        detect_res = {}
        _, loc_res, _, vdo_res = ners[0], ners[1], ners[2], ners[3]

        vdo_dict = other_ners_pinyin['vdo']

        type_loc_baidu = '9_loc_baidu'
        type_vdo = '9_vdo'

        # ##### vdo detect #######
        for vdo_uni in vdo_res:
            if vdo_uni in detect_res.keys():
                continue
            vdo_pinyin = ' '.join(lazy_pinyin(vdo_uni))
            if vdo_pinyin in vdo_dict:
                if (vdo_uni not in vdo_dict[vdo_pinyin]
                        and len(vdo_uni) >= 4):
                    detect_res[vdo_uni] = [type_vdo, vdo_dict[vdo_pinyin][0]]
        for loc_uni in loc_res:
            if loc_uni in detect_res.keys():
                continue
            if (loc_uni in loc_baidu_wrong.keys()
                    and loc_uni != loc_baidu_wrong[loc_uni]):
                detect_res[loc_uni] = [
                    type_loc_baidu, loc_baidu_wrong[loc_uni]]

        return detect_res
