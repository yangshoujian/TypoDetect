# -*- coding:utf-8 -*-

import logging
import sys
import copy
import functools
from pypinyin import lazy_pinyin
from pypinyin import TONE3
from ..utils.text_utils import uniform, is_chinese_string
from ..utils.text_utils import PUNCTUATION_LIST, uniform_punc
from ..utils.py_utils import is_words_same_py
from .bert_pred_seq import bert_long_pred
from .flyweight import ErrorInfo, SentInfo
from .flyweight import Stack, ECType
from .flyweight import Topk, ClassifyResult
from .resource_manager import resource as rs
from ..utils.time_utils import timer
from .nnlm_newdetector import lm_request
# from .nnlm_detector import lm_request


def cmp(pma, pmb):
    return (pma > pmb) - (pma < pmb)


def mysort(pma, pmb):
    # rawbegin, rawend, cand, etype, dec_ratio, is_valid
    # 从小到大排序
    if pma[3] != pmb[3]:
        return cmp(pma[3], pmb[3])
    elif pma[4] < pmb[4]:
        return -1
    elif pma[4] > pmb[4]:
        return 1
    return 0


class LmProcessor():

    def __init__(self):
        # @param mode:0 单路径, 1 多路径 ?
        self.mode = rs.cfg.getboolean('config', 'server_mode')
        self.debug_mode = rs.cfg.getint('config', 'debug_mode') # 0服务模式 1debug 2诊断模式 4输出checkppl详细信息
        self.use_cache = rs.cfg.getboolean('config', 'use_cache') # use_cache=0

    def __judge_guwen(self, sent_str):
        chars = uniform_punc(sent_str)
        if len(chars) > 0 and chars[-1] == ',':
            chars = chars[:-1]
        if len(chars) > 0 and chars[0] == ',':
            chars = chars[1:]
        len_chars = len(chars)
        if len_chars > 1:
            line = " ".join(list(chars))
            guwen_label = rs.guwen_classifier.predict([line])[0][0][0]
            if guwen_label == '__label__1':
                return True, len_chars
        return False, len_chars

    def __is_words_oov(self, words_list):
        ret = []
        for word in words_list:
            word_gbk = word.encode('gb18030', errors='ignore')
            if word_gbk and word_gbk in rs.ngram_lm:
                ret.append((word, 0))
            else:
                ret.append((word, 1))
        return ret

    def __cal_ppl_threshold(self, length):
        # 根据句子长度设置用于序列标注粗召回的阈值
        len_thresholds = [15, 23, 30]
        ppl_thresholds = [0.8, 0.8, 0.9]
        p_val = 0.8
        for tlen, tppl in zip(len_thresholds, ppl_thresholds):
            if length <= tlen:
                return tppl
        return p_val

    def process(self, fly_weight):
        if fly_weight.is_debug:
            self.debug_mode = 2 # 2诊断模式
        tms = timer(sys._getframe().f_code.co_name, fly_weight.logid)
        self.fly_weight = fly_weight
        if self.debug_mode & 2 == 2:
            fly_weight.debug_info['ori_sent_seg'] = ' '.join(fly_weight.basic_list)
            fly_weight.debug_info['ner_info'] = ' '.join(fly_weight.ners_unicode)
            fly_weight.debug_info['items'] = []
            fly_weight.debug_info['is_gw'] = 0

        # title_thres = {0: 0, 1: 0.7, 5: 0.4, 6: 0.45, 51: 0.5, 52: 0.5, 2: 0.6, 3: 0.5, 4: 0.5, 7: 0, 57: 0, 9: 0, 10: 0.2, 11: 0.5, 12: 0.5, 17: 0.5, 23: 0.5, 20: 0.9} # 标题的阈值，相对正文的阈值要提高点

        # filter guwen
        is_guwen, uni_len = self.__judge_guwen(fly_weight.sentence_str)
        if uni_len <= 0:
            logging.info('logid:%s uni_skp:%s'
                         % (fly_weight.logid, fly_weight.sentence_str))
            return
        if is_guwen and fly_weight.data_type == 0:
            if self.debug_mode & 2 == 2:
                fly_weight.debug_info['is_gw'] = 1
            logging.info('logid:%s guwen_skip:\t%s'
                         % (fly_weight.logid, fly_weight.sentence_str))
            return
        # ([pers], [locations] [orgs] [vdos])
        doc_ners_set = fly_weight.doc_info.ners
        detect_res = fly_weight.doc_info.name_detect_res
        self.doc_ner_all = fly_weight.doc_info.ner_all
        self.ngram_lm = rs.ngram_lm # 使用一个kenlm的python包去训练一个语言模型

        classify_result = ClassifyResult() # 存储NN模型识别结果

        (ss_sents_unicode_list, ss_flags_list, ss_pos_idx_list, ss_pos_list) = self.__split_sent(fly_weight.basic_list_unicode, fly_weight.check_flags, fly_weight.word_pos_idx, fly_weight.basic_pos) # 将整句以逗号分隔，切分为多个子句，并处理《》“”()等内部字符串

        ss_idx = 0 # 短句在长句中的位置
        for ss_sent_unicode, ss_flags, ss_pos_idx, ss_pos_list in zip(ss_sents_unicode_list, ss_flags_list, ss_pos_idx_list, ss_pos_list): # 逐个处理每个子句
            if (len(ss_sent_unicode) == 0 or len(ss_sent_unicode) > 90):
                continue
            # 按逗号切分后的子句逐个处理
            ss_sent_norm, punc_list = self.norm_sent_for_ngramlm(ss_sent_unicode, ss_flags) # 子句归一化处理，（1）去除标点符号；（2）标记正常字、OOV字、非汉字词等；（ngramlm计算的需求）
            ppl_threshold_by_len = self.__cal_ppl_threshold(len(ss_sent_unicode)) # 根据句子长度设置用于序列标注粗召回的阈值，ppl=0.8或0.9
            ppl = rs.get_ppl_score(ss_sent_norm) # 代表一个句子的困惑度，低困惑度的概率分布模型或概率模型能更好地预测样本
            words_score = self.__get_words_score(ss_sent_norm) # # ngramlm给句子打分
            sent_info = SentInfo(ss_sent_unicode, words_score, ss_sent_norm, ss_flags, ss_pos_list, ss_pos_idx, punc_list, ppl, ss_idx, self.debug_mode) # 短句信息： 长句以逗号分隔的短句

            # ysj 刚切完词后sent_info，在策略前进行豁免
            self.before_exempt(sent_info)

            if self.debug_mode & 1 == 1: # 1debug
                # for debug
                ss_sent_str = ''.join(ss_sent_unicode)
                ss_sent_norm_str = ' '.join([i for i in ss_sent_norm])
                flags_str = ' '.join([str(i) for i in ss_flags])
                posidx_str = ' '.join([str(i) for i in ss_pos_idx])
                punc_str = ' '.join([str(i) for i in punc_list])
                score_str = ' '.join([str(i) for i in words_score])
                msg_str = (('shortsent_info logid:%s sidx:[%d] str:[%s]'
                            ' segstr:[%s] normstr:[%s] ppl:[%.4f]'
                            ' flags:[%s] pos:[%s] punc:[%s]'
                            ' score:[%s] ppl_thred:[%.3f] paraid:%d') % (
                                fly_weight.logid, ss_idx, ss_sent_str,
                                ss_sent_unicode, ss_sent_norm_str,
                                ppl, flags_str, posidx_str, punc_str,
                                score_str, ppl_threshold_by_len, self.fly_weight.paragraph_id))
                logging.info(msg_str)

            ss_idx += 1
            self.checked_dict = {}
            self.cache_ppl_dict = {}
            self.__detect_chengyu_error_pinyin(sent_info)

            self.__detect_all_ner_error(sent_info, doc_ners_set, detect_res) # doc_ners_set —— unique list: [[name], [location], [org], [vdo]]

            self.__detect_fixed_error_py(sent_info)
 
            self.__detect_single_char_nonword_error(sent_info, ppl_threshold_by_len)
            # 豁免字词
            if ((self.mode == 0 and sent_info.error_cnt[1] == 0) or self.mode == 1): # 0服务模式（判别模型） 1debug 
                if fly_weight.sent_no == 0: # 标题
                    self.__detect_single_char_word_error(sent_info, 0.1, ppl_threshold_by_len, classify_result) # 2-单字词
                else:
                    self.__detect_single_char_word_error(sent_info, 0.5, ppl_threshold_by_len, classify_result)

            if ((self.mode == 0 and sent_info.error_cnt[1] == 0) or self.mode == 1):
                if fly_weight.sent_no == 0: # 标题
                    self.__detect_word_error(sent_info, 0.36, 0.33, ppl_threshold_by_len, classify_result) # 识别词语错误 3：同音词 4：近音词
                else:
                    self.__detect_word_error(sent_info, 0.4, 0.5, ppl_threshold_by_len, classify_result)

            if ((self.mode == 0 and sent_info.error_cnt[1] == 0) or self.mode == 1):
                # 两单字同音组词
                self.__detect_multi_single_char_error(sent_info, ppl_threshold_by_len, classify_result) # 8-连续单字组成同音词——多对一

            # 杨守建 12.28
            if ((self.mode == 0 and sent_info.error_cnt[1] == 0) or self.mode == 1): 
                if fly_weight.sent_no == 0: # 标题
                    self.__detect_before_after_char_error(sent_info, 0.1, ppl_threshold_by_len, classify_result) # 11-定一换一
                else: # 正文
                    self.__detect_before_after_char_error(sent_info, 0.5, ppl_threshold_by_len, classify_result)

            # if ((self.mode == 0 and sent_info.error_cnt[1] == 0) or self.mode == 1):
            #     self.__detect_before_after_char_error(sent_info, ppl_threshold_by_len, classify_result)

            if (self.mode == 0 and sum(sent_info.error_cnt) == 0 and fly_weight.sent_no == 0 and fly_weight.doc_id != "0"):
                self.__detect_nnlm_errors(sent_info, ppl_threshold_by_len, classify_result)

            if ((self.mode == 0 and sent_info.error_cnt[1] == 0) or self.mode == 1):
                self.__detect_one_to_multi_error(sent_info, ppl_threshold_by_len, classify_result) # 19-一对多

            if self.mode == 0:
                self.__fill_single_path_info(sent_info, fly_weight) # 从候选矩阵中获取正确候选，并填充至白板
            else:
                self.__fill_info(sent_info, fly_weight)

            if self.debug_mode & 2 == 2:
                self.__fill_debug_info(sent_info)
            self.fly_weight.shortsent_infos.append(sent_info)

        self.__adjust_error_info(self.fly_weight.shortsent_infos, classify_result) # merge各个短句识别结果及长句序列标注结果,返回最终长句全部识别结果
        if self.debug_mode & 2 == 2: # 2诊断模式
            self.__fill_final_debug_info(self.fly_weight.shortsent_infos, classify_result) # shortsent_infos短句信息列表
        del tms

    def __fill_final_debug_info(self, sent_infos, classify_result):
        fly = self.fly_weight
        fly.debug_info['is_judged'] = classify_result.is_judged
        fly.debug_info['is_sent_wrong'] = classify_result.is_sent_wrong
        fly.debug_info['seq_prob'] = classify_result.sent_prob # 序列标注模型预测概率
        # fill 调整后的各分句修改信息
        final_cands = []
        for item in fly.error_infos:
            (_, _, original_sentence, wrong_ci, correct_ci, err_tp, ratio, _, term_imp, _) = item
            final_cands.append((original_sentence, wrong_ci, correct_ci, err_tp, ratio, term_imp))
            logging.info('logid:%s debug_final_cans:%s %s-%s type:%d ratio:%.4f'
                         % (fly.logid, original_sentence, wrong_ci, correct_ci, err_tp, ratio))
        fly.debug_info['final_cands'] = final_cands

        cresult = classify_result
        if cresult.is_sent_wrong is False:
            return
        dinfo = fly.debug_info
        for seq_idx, _, _, seq_word in zip(cresult.wrong_chars_pos, cresult.wrong_chars, cresult.wrong_chars_prob, cresult.wrong_chars_inwords):
            for sidx, sinfo in enumerate(sent_infos): # sidx：短句序号，sinfo：短句信息，遍历每个短句
                sbasic_list = sinfo.basic_list_unicode # 每个短句的raw basic list
                if (len(sinfo.word_pos_idx) > 0 and seq_idx <= sinfo.word_pos_idx[-1]):
                    for tmp_i, tmp_pos_idx in enumerate(sinfo.word_pos_idx): # sinfo.word_pos_idx：基本词粒度: 每个词在原长句中的字偏移。   遍历每个短句的每处错误。
                        if (tmp_pos_idx <= seq_idx and seq_idx < (tmp_pos_idx + len(sbasic_list[tmp_i]))): # 词在原长句中的字偏移 <= 错别字位置 <= （字偏移+词长度）
                            if (tmp_i not in dinfo['items'][sidx]['cands']):
                                tmp_item = [seq_word, "", "", "", ""]
                                dinfo['items'][sidx]['cands'][tmp_i] = tmp_item # items的第sidx个短句的候选词cands的第tmp_i个为tmp_item
                            dinfo['items'][sidx]['cands'][tmp_i][3] = "1"
                            break
                    break

    def __detect_one_to_multi_error(self, sent_info, ppl_threshold_by_len, classify_result): # 识别一个词拆分成多个字的错误

        # tm = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        words_uni = sent_info.basic_list_unicode
        ch_flags = sent_info.check_flags
        punc_list = sent_info.punc_list
        go_classify, max_dec_ratio = False, 0.0
        frag_right, bert_idx = '', -1
        for j in range(len(words_uni)):
            cur_word = words_uni[j]
            if (punc_list[j] in [0, 2, -1] or ch_flags[j] <= 0 or len(cur_word) <= 1): # ch_flags[j]为-4:<> -2:标点非汉字 -1:专名 0:已改 1:待检测
                continue # 只检查punc_list[j]=1:词典汉字词，ch_flags[j]=1:待检测，至少2个字的词
            cand_words = rs.get_one_to_multi_cand(cur_word) # one_to_multi_dict： 归整 贵/正 柜/正 归/正 鬼/正
            for cand in cand_words:
                cand_seg = cand.split('/') # ['贵','正']
                cand_str = ' '.join(cand_seg) # 贵 正
                is_valid_sent, ret_ratio = self.check_sent_valid_by_ppl(sent_info, j, j, cand_str, 0.1, ECType.ONE_TO_MULTI) # 19-
                if is_valid_sent is True:
                    if self.mode == 0: # mode:0 单路径
                        sent_info.set_check_flags(j, j)
                    break
                elif ret_ratio < 0.15:
                    go_classify = True # 判别模型
                    dec_ratio = 1 - ret_ratio
                    if dec_ratio > max_dec_ratio:
                        max_dec_ratio = dec_ratio
                        frag_right = ''.join(cand_seg) # 贵正
                        bert_idx = j

        if go_classify:
            self.__check_suspect_error(sent_info, classify_result, bert_idx, bert_idx, frag_right, ECType.ONE_TO_MULTI, max_dec_ratio) # 判别模型：19-

    def __adjust_error_info(self, sent_infos, classify_result):
        # merge各个短句识别结果及长句序列标注结果,返回最终长句全部识别结果
        # 序列标注结果与Ngramlm识别结果不一致,以序列标注结果为准
        not_adjust_list, new_error_infos = [], []
        adjust_pos_dict = {}
        long_sent_list = self.fly_weight.basic_list_unicode
        long_sent = ''.join(long_sent_list)
        sno = self.fly_weight.sent_no # 句子在原文中的编号，与旧版兼容、全文从0开始
        char_check_flags = self.fly_weight.char_check_flags
        de_list = ['的', '得', '地']
        reg_zis = set()
        ner_cnt = self.fly_weight.doc_info.ner_cnt
        data_type = self.fly_weight.data_type
        for sent_info in sent_infos: # sent_infos短句信息列表，遍历每条短句
            short_sent_str = ''.join(sent_info.basic_list_unicode) # short_sent_str是中间没有标点的短句
            for einfo in sent_info.error_infos: # 短句内错误信息记录
                wrong_frag = einfo.wrong_frag
                correct_frag = einfo.correct_frag
                (error_type, char_pos_begin) = (einfo.error_type, einfo.sent_char_pos_begin) # sent_char_pos_begin：unicde char pos
                term_imp = einfo.term_imp # 错字词重要性:[0:不重要,1:一般,2:重要]
                # 高准高召类型
                type_recall_pre = 0
                if (error_type in [7, 57] and ner_cnt[0] is not None and ner_cnt[0].get(wrong_frag, 0) >= 2): # 7-人名识别， NER_PERSON_RECALL = 57， ner_cnt = [None, None, None, None] # 统计文章整体信息：如人名出现频次等
                    # 出现两次以上则非高准版
                    type_recall_pre = 1
                elif (error_type == 9 and ner_cnt[3] is not None and ner_cnt[3].get(wrong_frag, 0) >= 2): # 9-其他专名识别
                    type_recall_pre = 1

                for char_idx, char_w in enumerate(wrong_frag):
                    if char_w == correct_frag[char_idx]:
                        continue
                    if error_type == 10 and char_w in ['的', '地']: # 10-固定搭配-词表
                        continue
                    char_pos = char_pos_begin + char_idx
                    new_error_infos.append((sno, char_pos, short_sent_str, char_w, correct_frag[char_idx], error_type, einfo.prob, long_sent, term_imp, type_recall_pre))
                    eprob = float(einfo.prob)
                    if char_pos not in adjust_pos_dict:
                        adjust_pos_dict[char_pos] = (sno, char_pos, short_sent_str, char_w, correct_frag[char_idx], error_type, einfo.prob, long_sent, term_imp, type_recall_pre)
                    if error_type in [0, 7, 57, 9, 10, 15, 17, 57]:
                        # 成语人名等策略召回准确率高, 直接保留
                        not_adjust_list.append((sno, char_pos, short_sent_str, char_w, correct_frag[char_idx], error_type, einfo.prob, long_sent, term_imp, type_recall_pre))
                        if char_w not in reg_zis:
                            reg_zis.add(char_w)
                    # 标题处理
                    eprob = float(einfo.prob)
                    if (((char_w in de_list and correct_frag[char_idx] in de_list)
                         or (error_type == 1 and eprob > 0.5)
                         or (error_type == 51 and eprob > 0.8)
                         or (error_type in [2, 3, 4] and eprob > 0.8))
                            and sno == 0):

                        not_adjust_list.append((sno, char_pos, short_sent_str, char_w, correct_frag[char_idx], error_type, einfo.prob, long_sent, term_imp, type_recall_pre))
                        if char_w not in reg_zis:
                            reg_zis.add(char_w)

        if (((sno < 3 and len(new_error_infos) == 0) or sno == 0) and classify_result.is_judged is False and self.fly_weight.doc_id != '0'):
            # 若未触发序列标注模型,主动触发
            prob_error, _ = self.__detect_sent_error(self.fly_weight.basic_list_unicode, classify_result)
            if prob_error > 0.5: # 序列标注模型预测概率
                classify_result.is_sent_wrong = True
            else:
                classify_result.is_sent_wrong = False
        if classify_result.is_sent_wrong is True:
            # 序列标注识别句子有错误，以序列标注结果为准
            new_error_infos_ajust = []
            new_error_infos_ajust.extend(not_adjust_list)
            cres = classify_result
            for seq_idx, seq_char, seq_prob, seq_word in zip(
                    cres.wrong_chars_pos, cres.wrong_chars,
                    cres.wrong_chars_prob, cres.wrong_chars_inwords):
                if (char_check_flags[seq_idx] > 0 and is_chinese_string(seq_char) and seq_char not in reg_zis):
                    if seq_char in rs.pass_chars or seq_char in rs.corr_chars: # 豁免的字
                        continue
                    if seq_word in rs.pass_words or seq_word in rs.corr_words: # 豁免的词
                        continue
                    if len(seq_word) > 1:
                        if (seq_word.endswith('哥') or seq_word.endswith('姐') or seq_word.endswith('氏') or seq_word.endswith('妃') or seq_word.startswith('小')):
                            continue
                    if seq_idx in adjust_pos_dict.keys():
                        if seq_char == adjust_pos_dict[seq_idx][3]:
                            new_error_infos_ajust.append(adjust_pos_dict[seq_idx])
                    else:
                        s_find = ''
                        for sinfo in sent_infos:
                            if (len(sinfo.word_pos_idx) > 0 and seq_idx <= sinfo.word_pos_idx[-1]):
                                s_find = ''.join(sinfo.basic_norm_unicode) # s_find是中间没有标点的短句
                                break
                        find_npos = s_find.find(seq_char)
                        if len(s_find) == 0 or find_npos == -1:
                            s_find = long_sent
                        correct_ci = self.__get_corrected(seq_idx, seq_char, seq_word, long_sent_list) # 纠错
                        term_imp = self.__get_term_imp(seq_char, correct_ci, 20) # 获取错字词重要性:[0:不重要,1:一般,2:重要]

                        type_recall_pre = 1
                        new_error_infos_ajust.append((sno, seq_idx, s_find, seq_char, correct_ci, 20, seq_prob, long_sent, term_imp, type_recall_pre))
            if len(new_error_infos_ajust) == 0 and len(new_error_infos) > 0:
                new_error_infos_ajust.append(new_error_infos[0])
                # tlast = new_error_infos_ajust[-1]
            new_error_infos = new_error_infos_ajust
        # elif sno == 0:
            # 标题识别时序列标注未识别结果，只采纳部分语言模型识别结果
            # new_error_infos = not_adjust_list

        final_new_error_infos = []
        if data_type == 1:
            # 短视频:调整高召高准标记\延迟分发
            sent_prob = classify_result.classify_prob
            prob_thres = {2: 0.7, 3: 0.5, 4: 0.5, 51: 0.6}
            for tinfo in new_error_infos:
                (sno, sidx, ssent, w_ci, c_ci, etype,
                 prob, longsent, timp, rp_type) = tinfo
                if ((etype == 0 and prob > 0.001)
                        or (etype == 1 and prob > 0.75)
                        or (etype == 2 and prob > 0.85)
                        or (sent_prob > 0.5 and prob > prob_thres.get(etype, 1.0))
                        or (prob > 0.5 and etype in [10, 12])
                        or (sent_prob > 0.6 and etype == 6 and prob > 0.5
                            and is_words_same_py(w_ci, c_ci))):
                    # 视频标题高准版
                    rp_type = 2
                elif (classify_result.classify_prob > 0.7
                      or etype == 51 or etype == 0
                      or (etype == 1 and (prob > 0.7 or sent_prob > 0.5))
                      or (etype == 2 and (prob > 0.68 or sent_prob > 0.5))
                      or (etype in [3, 4] and prob > 0.7)):
                    # 视频标题延迟分发版
                    rp_type = 3
                final_new_error_infos.append((sno, sidx, ssent, w_ci, c_ci, etype, prob, longsent, timp, rp_type))
        else:
            final_new_error_infos = new_error_infos

        self.fly_weight.error_infos.extend(final_new_error_infos)

    def __get_corrected(self, idx, wrong_zi, wrong_ci, sent_of_words):
        correct_ci = 'X'
        # find the begin of wrong_ci's idx
        begin_idx = -1
        pidx = 0
        for pidx in range(len(wrong_ci)):
            if wrong_ci[pidx] == wrong_zi: # 找到错字在错词中的位置pidx
                begin_idx = idx - pidx
                break
        wrong_zi_idx = pidx
        # find the idx of wrong_ci in seg sent
        pos_idx_word, pos_i = [], 0
        for t_word in sent_of_words:
            pos_idx_word.append(pos_i)
            pos_i += len(t_word)
        words_norm, idx_norm = self.__normalize_words(sent_of_words, pos_idx_word)
        err_idx = -1
        for tmp_idx in range(len(idx_norm)):
            if (idx_norm[tmp_idx] == begin_idx and words_norm[tmp_idx] == wrong_ci):
                err_idx = tmp_idx
                break
        cand_words = rs.get_tongyin_ci_zi(wrong_ci) # 多个同音词
        if wrong_ci in cand_words:
            cand_words.remove(wrong_ci)
        if err_idx >= 0:
            ppl = rs.get_ppl_score(words_norm)
            min_ppl = float('inf')
            for cand in cand_words:
                before = words_norm[: err_idx]
                after = words_norm[err_idx + 1:]
                center = [cand]
                new_ppl = rs.get_ppl_score(before + center + after)
                if new_ppl < 0.7 * ppl:
                    correct_ci = cand[wrong_zi_idx] # 错字在词中的位置
                    break
                if new_ppl < min_ppl:
                    min_ppl = new_ppl
                    correct_ci = cand[wrong_zi_idx]
        if correct_ci == 'X' or correct_ci == wrong_zi:
            cand_zis = rs.get_tongyin_ci_zi(wrong_zi) # 多个同音字
            if wrong_zi in cand_zis:
                cand_zis.remove(wrong_zi)
            if len(cand_zis) > 0: # 如果同音字是多个
                correct_ci = list(cand_zis)[0] # 选第一个同音字
        return correct_ci

    def __normalize_words(self, sent_of_words, pos_idx_word):
        # 去除标点，ngramlm的需要
        words, pos_ids = [], []
        for word, pos in zip(sent_of_words, pos_idx_word):
            if word not in PUNCTUATION_LIST:
                words.append(word)
                pos_ids.append(pos)
        return words, pos_ids

    def __detect_nnlm_errors(self, sent_info, ppl_threshold_by_len, classify_result):
        # 调用NNLM模型识别句子错误
        # tm = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        basic_norm_unicode = sent_info.basic_norm_unicode
        check_flags = sent_info.check_flags
        norm_to_raw = sent_info.norm_to_raw
        go_classify, max_ratio, find = False, 0.0, False
        prob_wrong, prob_correct, prob_pos = '', '', -1
        if (len(basic_norm_unicode) > 0 and is_chinese_string(basic_norm_unicode[-1])):
            basic_norm_unicode.append('，')
            check_flags.append(-1)
        find_cand_num = 0
        sent_of_char = []
        idx_of_word = []  # 单字对应词的位置
        idx_in_word = []  # 单字在对应词中位置
        for i, word in enumerate(basic_norm_unicode):
            for j, char in enumerate(word):
                sent_of_char.append(char)
                idx_of_word.append(i)
                idx_in_word.append(j)
        sent_str = ' '.join(sent_of_char)
        _, det_res, ret_code = lm_request(self.fly_weight.logid, sent_str, self.fly_weight.bid) # 调用nnlm子服务
        if ret_code == 0:
            for idx, ori_zi, _, cor_zi, \
                    _, _, ppl_ratio, reg_typ in det_res:
                if idx >= len(idx_of_word):
                    continue
                word_idx, charidx_in_word = idx_of_word[idx], idx_in_word[idx]
                if word_idx not in norm_to_raw:
                    continue
                raw_idx = norm_to_raw[word_idx]
                if reg_typ == 1:
                    cur_word = basic_norm_unicode[word_idx]
                    frag_list = []
                    frag_list.append(cur_word[0: charidx_in_word])
                    frag_list.append(cor_zi)
                    frag_list.append(cur_word[charidx_in_word + len(cor_zi):])
                    frag_right = ''.join(frag_list)
                    sent_info.add_cand(word_idx, word_idx, frag_right, ECType.NNLM_1, ppl_ratio, True)
                    find = True
                    break
                else:
                    if (is_chinese_string(ori_zi) and check_flags[raw_idx] > 0 and ppl_ratio > max_ratio):
                        max_ratio = ppl_ratio
                        prob_wrong = ori_zi
                        prob_correct = cor_zi
                        prob_pos = idx  # 字偏移
            if prob_pos != -1 and len(prob_wrong) > 0:
                go_classify = True
            if not find and go_classify:
                (word_idx, charidx_in_word) = (idx_of_word[prob_pos],
                                               idx_in_word[prob_pos])
                cur_word = basic_norm_unicode[word_idx]
                frag_list = []
                frag_list.append(cur_word[0: charidx_in_word])
                frag_list.append(prob_correct)
                frag_list.append(cur_word[charidx_in_word + len(prob_wrong):])
                frag_right = ''.join(frag_list)
                raw_idx = norm_to_raw[word_idx]
                ret_num = self.__check_suspect_error(sent_info, classify_result, raw_idx, raw_idx, frag_right, ECType.NNLM_2, max_ratio) # 22-NNLM扩大候选
                find_cand_num += ret_num
        return find_cand_num

    def __fill_debug_info(self, sent_info):
        st_data = {}
        st_data['sno'] = sent_info.short_idx
        st_data['ori_sent_seg'] = ' '.join(sent_info.basic_list_unicode)
        ori_sent_seg_list = [i for i in sent_info.basic_list_unicode]
        st_data['ppl'] = sent_info.raw_ppl
        st_data['flags'] = ' '.join([str(i) for i in sent_info.check_flags])
        st_data['punc'] = ' '.join([str(i) for i in sent_info.punc_list])
        # 候选词信息
        st_data['cands'], _ = self.__get_cands_debug_info(sent_info.debug_info['cands'], ori_sent_seg_list)
        st_data['frag_cands'], _ = self.__get_cands_debug_info(sent_info.debug_info['frag_cands'], ori_sent_seg_list, 1)
        # st_data['all_cands'] = sent_info.debug_info['cands']
        # st_data['all_cands'] = sorted_candlist
        # 最终入选的词信息
        add_cands = []
        for einfo in sent_info.error_infos:
            add_cands.append((einfo.sent_ci_pos_begin, einfo.sent_ci_pos_end, einfo.correct_frag, einfo.error_type, einfo.prob))
        st_data['add_cands'] = add_cands
        self.fly_weight.debug_info['items'].append(st_data)

    def __get_cands_debug_info(self, cands_dict, ori_sent_list, cand_type=0):
        # 0 同音近音LM, 判别模型候选,判别模型结果,其他
        # 类型及展示列位置关系如下
        basic_dict = {2: 0, 52: 0, 3: 0, 4: 0, 5: 1, 6: 1, 20: 2}
        phrase_dict = {0: 0, 1: 0, 51: 0, 7: 3, 57: 3, 8: 0, 9: 3, 10: 3, 11: 3, 12: 1, 19: 3, 17: 3, 22: 3} # 为了前端展示
        if cand_type == 0:
            index_dict = basic_dict
        else:
            index_dict = phrase_dict
        newcands = {}
        sorted_candlist = {}
        # 遍历每个词遍历候选
        for offset_t, cand_list in cands_dict.items():
            if isinstance(offset_t, tuple):
                offset = int(offset_t[0]) * 1000 + int(offset_t[1])
            else:
                offset = offset_t
            newcands[offset] = ["", "", "", "", ""]
            # 对当前位置所有候选按类型、类型ppl下降阈值
            tmplist = sorted(cand_list, key=functools.cmp_to_key(mysort))
            sorted_candlist[offset_t] = tmplist
            pre_etype, pre_idx, idx, cnt = -1, -1, -1, 0
            newlist = [[], [], [], []]  # 同音近音、序列标注、序列标注结果、其他
            bidx, eidx, etype = -1, -1, -1
            for bidx, eidx, cand, etype, dec_ratio, _ in tmplist:
                idx = index_dict.get(etype, 3)
                if etype != pre_etype:
                    if pre_etype != -1:
                        newcands[offset][0] = "".join(ori_sent_list[bidx: eidx + 1])
                        newcands[offset][pre_idx + 1] = "%s   %s[%s]" % (newcands[offset][pre_idx + 1], pre_etype, ' '.join(newlist[pre_idx]))
                        newlist = [[], [], [], []]  # 同音近音、序列标注、序列标注结果、其他
                    pre_etype, pre_idx = etype, idx
                    cnt = 0
                elif cnt >= 5:
                    # 每种类型最多只选5个候选
                    continue
                tmp_str = "%s:%.2f" % (cand, dec_ratio)
                newlist[idx].append(tmp_str) # 每种错误类型选5个候选
                cnt += 1
            if sum([len(item) for item in newlist]) > 0:
                # 第一个位置是词，其后offset+1
                newcands[offset][0] = "".join(ori_sent_list[bidx: eidx + 1])
                newcands[offset][idx + 1] = "%s   %s[%s]" % (newcands[offset][idx + 1], etype, ' '.join(newlist[idx]))
        return newcands, sorted_candlist

    def __detect_sent_error(self, sent_list, classify_result): # 触发序列标注模型
        (long_prob, ret_wrong_chars, ret_wrong_posis, ret_wrong_probs, ret_wrong_words, jd_type) = bert_long_pred(' '.join(sent_list), self.fly_weight.logid, self.fly_weight.sent_no, self.fly_weight.data_type, self.fly_weight.bid)
        classify_result.is_judged = True
        classify_result.sent_prob = long_prob
        classify_result.wrong_chars = ret_wrong_chars
        classify_result.wrong_chars_pos = ret_wrong_posis
        classify_result.wrong_chars_prob = ret_wrong_probs
        classify_result.wrong_chars_inwords = ret_wrong_words # 字节index到词的映射
        return long_prob, jd_type # 返回：序列标注模型预测概率

    def __check_suspect_error(
            self, sent_info, classify_result, err_bgidx,
            err_endidx, frag_right, error_type, max_dec_ratio):

        if classify_result.is_judged is False:
            (long_prob, ret_wrong_chars, ret_wrong_posis, ret_wrong_probs, ret_wrong_words, jd_type) = bert_long_pred(' '.join(self.fly_weight.basic_list_unicode), self.fly_weight.logid, self.fly_weight.sent_no, self.fly_weight.data_type, self.fly_weight.bid)
            classify_result.is_judged = True
            classify_result.sent_prob = long_prob # 序列标注模型预测概率
            classify_result.is_sent_wrong = True if jd_type > 0 else False
            classify_result.wrong_chars = ret_wrong_chars
            classify_result.wrong_chars_pos = ret_wrong_posis
            classify_result.wrong_chars_prob = ret_wrong_probs
            classify_result.wrong_chars_inwords = ret_wrong_words
        find_cand_num = 0
        if (classify_result.is_judged is True and classify_result.is_sent_wrong is True):
            frag_wrong = ''.join(sent_info.basic_list_unicode[err_bgidx: err_endidx + 1])
            is_exempt = self.check_is_exempt(frag_wrong, frag_right, error_type)
            sent_info.add_cand(err_bgidx, err_endidx, frag_right, error_type, max_dec_ratio, False, is_exempt) # frag_right=贵正
            sent_info.set_check_flags(err_bgidx, err_endidx)
            find_cand_num += 1
        return find_cand_num

    def __detect_before_after_char_error(self, sent_info, ppl_ratio, ppl_threshold_by_len, classify_result):

        find_cand_num = 0
        if len(sent_info.basic_norm_unicode) < 4:
            return find_cand_num

        sent_of_word = sent_info.basic_list_unicode
        len_sent_of_word = len(sent_of_word)
        check_flags = sent_info.check_flags
        punc_list = sent_info.punc_list
        start, num_of_words = -1, 0
        go_classify, max_dec_ratio = False, 0
        frag_right, bert_idx = '', -1
        find_cand = False
        for i in range(len(sent_of_word)):
            # char_word
            if (len(sent_of_word[i]) == 1 and is_chinese_string(sent_of_word[i])):
                if num_of_words == 0:
                    start = i
                num_of_words += 1
                if i < len_sent_of_word - 1:
                    continue
            # non_single_char word or punc occurs
            if num_of_words <= 1:
                num_of_words = 0
                start = -1
                continue

            # multi single char index from [start, start + num_of_words)
            for j in range(start, start + num_of_words):
                if check_flags[j] <= 0 or punc_list[j] <= 0:
                    continue
                if j > start:
                    fix_before_zi = sent_of_word[j - 1] + '+1' # 定前一个字，如：英+1
                    if sent_of_word[j] in rs.get_danzi_cand(fix_before_zi): # 如果前一个字属于单字的字典，则不检查。如：英+1:法    锦   拉   媒   短   台   达   属   良   剧   叔   战   红   王
                        continue
                if j < start + num_of_words - 1:
                    fix_after_zi = sent_of_word[j + 1] + '-1' # 定后一个字，如：缝-1
                    if sent_of_word[j] in rs.get_danzi_cand(fix_after_zi): # 如果后一个字属于单字的字典，则不检查。如：缝-1:条  美
                        continue
                py_j = ''.join(lazy_pinyin(sent_of_word[j])) # lazy_pinyin汉字转拼音，无间隔拼接
                if j > start and punc_list[j - 1] > 0 and punc_list[j] > 0:
                    fix_before_cands = rs.get_before_after_char_cand(fix_before_zi) # 如果不属于单字的字典，则获取前一个字的定一换一字典，如：英+1:雄    国   语   寸   文   超   镑   里   尺   俊   勇   伦   才   朗   烈   伟   气   美   军
                    for before_cand in fix_before_cands: # 遍历每个候选固定搭配字
                        py_c = ''.join(lazy_pinyin(before_cand)) # 汉字转拼音
                        if py_j[0] != py_c[0]: # 原字和候选固定搭配字的拼音不相同
                            continue
                        cand = sent_of_word[j - 1] + before_cand # 定前换后，和前一个字一起组成词
                        is_valid_sent, ret_ratio = \
                            self.check_sent_valid_by_ppl(sent_info, j - 1, j, cand, ppl_ratio, ECType.BEFORE_CHAR) # 重新计算ppl值，11-连续单字定一换一搭配

                        if is_valid_sent:
                            find_cand = True
                            if self.mode == 0:
                                sent_info.set_check_flags(j - 1, j)
                            break

                        if (is_valid_sent is False and ret_ratio < ppl_threshold_by_len):
                            go_classify = True
                            dec_ratio = 1 - ret_ratio
                            if dec_ratio > max_dec_ratio:
                                max_dec_ratio = dec_ratio
                                frag_right = cand
                                bert_idx = j - 1

                if find_cand is True:
                    find_cand_num += 1
                    break

                if (j < start + num_of_words - 1 and punc_list[j] > 0 and punc_list[j + 1] > 0):
                    fix_after_cands = rs.get_before_after_char_cand(fix_after_zi) # 如果不属于单字的字典，则获取后一个字的定一换一字典，如：缝-1:无   裂   裁   夹   门   接   牙   指   地   石   合   焊   中   眯   骑   隙   弥
                    for after_cand in fix_after_cands:
                        py_c = ''.join(lazy_pinyin(after_cand))
                        if py_j[0] != py_c[0]:
                            continue
                        cand = after_cand + sent_of_word[j + 1]
                        is_valid_sent, ret_ratio = \
                            self.check_sent_valid_by_ppl(sent_info, j, j + 1, cand, ppl_ratio, ECType.BEFORE_CHAR)
                        if is_valid_sent:
                            find_cand = True
                            if self.mode == 0:
                                sent_info.set_check_flags(j, j + 1)
                            break

                        if (is_valid_sent is False and ret_ratio < ppl_threshold_by_len):
                            go_classify = True
                            dec_ratio = 1 - ret_ratio
                            if dec_ratio > max_dec_ratio:
                                max_dec_ratio = dec_ratio
                                frag_right = cand
                                bert_idx = j
            # processed char word
            num_of_words = 0
            start = -1

        if not find_cand and go_classify is True:
            ret_num = self.__check_suspect_error(sent_info, classify_result, bert_idx, bert_idx + 1, frag_right, ECType.BEFORE_BERT, max_dec_ratio) # 13-定一换一判别模型
            find_cand_num += ret_num
        return find_cand_num

    def __detect_multi_single_char_error(
            self, sent_info, ppl_threshold_by_len, classify_result):

        # tm = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        if len(sent_info.basic_norm_unicode) < 4:
            return

        sent_of_word = sent_info.basic_list_unicode
        len_sent_of_word = len(sent_of_word)
        check_flags = sent_info.check_flags
        punc_list = sent_info.punc_list
        start, num_of_words = -1, 0
        go_classify, max_dec_ratio = False, 0
        frag_right, bert_idx = '', -1
        find_cand = False
        for i in range(len(sent_of_word)):
            # char_word
            if (len(sent_of_word[i]) == 1 and is_chinese_string(sent_of_word[i])):
                if num_of_words == 0:
                    start = i
                num_of_words += 1
                if i < len_sent_of_word - 1:
                    continue

            # non_single_char word or punc occurs
            if num_of_words <= 1:
                num_of_words = 0
                start = -1
                continue

            # multi single char index from [start, start + num_of_words)
            for j in range(start, start + num_of_words - 1):
                if check_flags[j] <= 0 or punc_list[j] in [0, 2]:
                    continue
                # get pinyin of continuous two single chars
                word_pinyin = ' '.join(lazy_pinyin(sent_of_word[j] + sent_of_word[j + 1])) # 相邻两个字组成词
                same_py_cands = rs.get_samepy_common_word(word_pinyin) # word_py_dict词典，如 yuan wei:原味 原为 原位 鸢尾 原委
                for cand in same_py_cands:
                    _, ret_ratio = self.check_sent_valid_by_ppl(sent_info, j, j + 1, cand, 0.01, ECType.MULTICHAR, False)
                    if ret_ratio < ppl_threshold_by_len:
                        go_classify = True
                        dec_ratio = 1 - ret_ratio
                        if dec_ratio > max_dec_ratio:
                            max_dec_ratio = dec_ratio
                            frag_right = cand
                            bert_idx = j
            # processed char word
            num_of_words = 0
            start = -1
        if not find_cand and go_classify is True:
            self.__check_suspect_error(sent_info, classify_result, bert_idx, bert_idx + 1, frag_right, ECType.MULTICHAR_BERT, max_dec_ratio) # 12-多对一判别模型

    def __detect_all_ner_error(self, sent_info, doc_ners, detect_res):
        # sent_info：短句信息：长句以逗号分隔的短句，doc_ners_set： unique list: [[name], [location], [org], [vdo]]
        # tm = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        per_res_uni = [per for per in doc_ners[0]] # 人名实体
        ch_flags = sent_info.check_flags # 词粒度: 每个词原长句中的flag，-4:<> -2:标点非汉字 -1:专名 0:已改 1:待检测
        words_uni = sent_info.basic_list_unicode # raw basic list
        punc_list = sent_info.punc_list # -1:OOV在字典以外的字 0:标点 1:词典汉字词 2:非汉字词数词等 3:oov
        find_err_num = 0 # 找到的错误数量
        for j in range(len(words_uni)):
            if (is_chinese_string(words_uni[j]) and ch_flags[j] < 0 and punc_list[j] not in [0, 2]):
                word = words_uni[j]
                if word in detect_res.keys() and word in per_res_uni:
                    type_str = detect_res[word][0]
                    wrong_zi = word
                    correct_zi = detect_res[word][1]

                    if type_str == "7_baidu" and wrong_zi == u'陈晴':
                        if (j + 1 < len(words_uni) and words_uni[j + 1] not in [u'男团', u'令']):
                            continue

                    if (wrong_zi != correct_zi and len(wrong_zi) == len(correct_zi)):
                        if (type_str == '7_baidu' or wrong_zi[-1] in (u'哥', u'姐')):
                            etype = ECType.NER_PERSON_RECALL # 57
                        else:
                            etype = ECType.NER_PERSON # 7-人名识别
                        sent_info.set_check_flags(j, j) # 设置j位置为0-已改
                        sent_info.add_cand(j, j, correct_zi, etype, 1.0) # 添加候选字/词
                        find_err_num += 1
                else:
                    pos_begin, pos_end = j, j
                    find_other_ner = False
                    for k in range(pos_begin + 1, len(words_uni)):
                        if ch_flags[k] > 0 or punc_list[k] == 0: # 待检测、标点
                            find_other_ner = True
                            pos_end = k - 1
                            break
                        elif k == len(words_uni) - 1:
                            find_other_ner = True
                            pos_end = k

                    if find_other_ner:
                        wrong_zi = ''.join(words_uni[pos_begin: pos_end + 1])
                        if (wrong_zi in detect_res.keys() and wrong_zi not in per_res_uni):
                            type_str = detect_res[wrong_zi][0]
                            correct_zi = detect_res[wrong_zi][1]

                            if (wrong_zi != correct_zi and len(wrong_zi) == len(correct_zi)):
                                sent_info.set_check_flags(pos_begin, pos_end)
                                sent_info.add_cand(pos_begin, pos_end, correct_zi, ECType.NER_OTHER, 1.0) # 9-其他专名识别
                                find_err_num += 1
        return find_err_num

    def __detect_fixed_error_py(self, sent_info):

        # tm = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        # 识别错误类型: 两个单字组词、一个字同词语同前缀或后缀
        raw_sent = ''.join(self.fly_weight.basic_list_unicode)
        ppl = sent_info.raw_ppl
        fixed_pairs = rs.fixed_pairs # # 共现数据?
        words_uni = sent_info.basic_list_unicode
        words_norm = sent_info.basic_norm_unicode
        ch_flags = sent_info.check_flags
        punc_list = sent_info.punc_list
        raw_to_norm = sent_info.raw_to_norm
        find_err_num = 0
        for j in range(len(words_uni)):
            if (not is_chinese_string(words_uni[j]) or ch_flags[j] <= 0 or punc_list[j] <= 0):
                continue
            cand_words_head = fixed_pairs['head'].get(words_uni[j], {}) # 前缀的固定搭配
            cand_words_tail = fixed_pairs['tail'].get(words_uni[j], {}) # 后缀的固定搭配
            find_head, find_tail = False, False
            ppl_head, ppl_tail = ppl, ppl
            head_begin, head_end = -1, -1
            break_flag = True if len(cand_words_head) <= 0 else False

            for win_len in range(1, 4):
                if (j + 1 + win_len) > len(words_uni) or break_flag: # 词越界
                    break
                flags_list = [ch_flags[ch] for ch in range(j + 1, j + 1 + win_len)]
                if sum(flags_list) < win_len: # 不是全为1-待检测
                    break
                has_punc = False
                for tmp_idx in range(j + 1, j + 1 + win_len):
                    if punc_list[tmp_idx] == 0: # 0:标点
                        has_punc = True
                        break
                if has_punc is True: # 标点不检查
                    break
                wrong_zi = "".join(words_uni[j + 1: j + win_len + 1]) # 前缀的原搭配
                wrong_zi_seg = "/".join(words_uni[j + 1: j + win_len + 1])
                wrong_py = " ".join(lazy_pinyin(wrong_zi)) # 原搭配的拼音
                wrong_cizu = words_uni[j] + wrong_zi # 原词组
                if (wrong_py in cand_words_head.keys() and raw_sent.find(wrong_cizu) >= 0): # 前缀的原搭配的拼音在固定搭配词典中，如：火影/忍着
                    cand_list = cand_words_head[wrong_py] # 固定搭配词典中的搭配列表
                    if wrong_zi_seg in cand_list:
                        continue
                    for cand_i in range(len(cand_list)): # 依次遍历每个候选搭配
                        before = words_norm[: raw_to_norm[j + 1]]
                        center = [cand_s for cand_s in cand_list[cand_i].split('/')]
                        after = words_norm[raw_to_norm[j + win_len] + 1:]
                        ppl_head = rs.get_ppl_score(before + center + after)
                        if ppl_head < ppl:
                            correct_head = "".join(cand_list[cand_i].split('/')) # 固定搭配词典中的正确搭配
                            wrong_head = wrong_zi
                            head_begin, head_end = j + 1, j + win_len
                            find_head = True
                            break
                    if find_head:
                        break

            tail_begin, tail_end = -1, -1
            break_flag = True if len(cand_words_tail) <= 0 else False
            for win_len in range(1, 4):
                if (j - win_len) < 0 or break_flag:
                    break
                flags_list = [ch_flags[ch] for ch in range(j - win_len, j)]
                if sum(flags_list) < win_len:
                    break
                has_punc = False
                for tmp_idx in range(j - win_len, j):
                    if punc_list[tmp_idx] == 0:
                        has_punc = True
                        break
                if has_punc is True:
                    break
                wrong_zi = "".join(words_uni[j - win_len: j])
                wrong_zi_seg = "/".join(words_uni[j - win_len: j])
                wrong_py = " ".join(lazy_pinyin(wrong_zi))
                wrong_cizu = wrong_zi + words_uni[j]
                if (wrong_py in cand_words_tail.keys()
                        and raw_sent.find(wrong_cizu) >= 0):
                    cand_list = cand_words_tail[wrong_py]
                    if wrong_zi_seg in cand_list:
                        continue
                    for cand_i in range(len(cand_list)):
                        cand = "".join(cand_list[cand_i].split('/'))
                        tmp_cand = ' '.join(cand_list[cand_i].split('/'))
                        _, ret_ratio = self.check_sent_valid_by_ppl(sent_info, j - win_len, j - 1, tmp_cand, 0.1, ECType.FIXED_ERROR, False)
                        if ret_ratio < 1:
                            correct_tail = cand
                            wrong_tail = wrong_zi
                            tail_begin, tail_end = j - win_len, j - 1
                            find_tail = True
                            ppl_tail = ppl * ret_ratio
                            break
                    if find_tail:
                        break

            if find_head or find_tail:
                if ppl_head <= ppl_tail: # ppl值更小
                    new_ppl = ppl_head
                    wrong_zi = wrong_head
                    correct_zi = correct_head
                    error_begin, error_end = head_begin, head_end
                else:
                    new_ppl = ppl_tail
                    wrong_zi = wrong_tail
                    correct_zi = correct_tail
                    error_begin, error_end = tail_begin, tail_end
                ratio = (ppl - new_ppl) / ppl
                if wrong_zi != correct_zi and len(wrong_zi) == len(correct_zi):
                    sent_info.set_check_flags(error_begin, error_end)
                    sent_info.add_cand(error_begin, error_end, correct_zi, ECType.FIXED_ERROR, ratio) # 10-词语固定搭配
                    find_err_num += 1
                break
        return find_err_num

    def __fill_single_path_info(self, sent_info, fly_weight):
        # 从候选矩阵中获取正确候选，并填充至白板
        basic_norm_unicode = sent_info.basic_norm_unicode
        cands_list = sent_info.cands_list
        word_pos_idx = sent_info.word_pos_idx
        norm_to_raw = sent_info.norm_to_raw
        for i in range(len(cands_list)):
            if len(cands_list[i]) > 1:
                item = cands_list[i][1] # cands_list[i][i : k] 为第i个词的所有正确候选
                basic_begin = item.basic_begin
                basic_end = item.basic_end
                wrong_frag = ''.join(basic_norm_unicode[basic_begin: basic_end + 1])
                right_frag = ''.join(item.cand_str.split())
                raw_begin = norm_to_raw[basic_begin]
                pos_begin = word_pos_idx[raw_begin]
                pos_end = pos_begin + len(wrong_frag)
                term_imp = self.__get_term_imp(wrong_frag, right_frag, item.cand_type) # 获取错字词重要性:[0:不重要,1:一般,2:重要]
                # 繁简关系直接调整为0
                errinfo = ErrorInfo(wrong_frag, right_frag, sent_info.short_idx, basic_begin, basic_end, item.cand_type, item.ppl_ratio, pos_begin, pos_end, term_imp)
                sent_info.error_infos.append(errinfo)

    def __get_term_imp(self, wrong_frag, right_frag, cand_type=-1): # 获取错字词重要性:[0:不重要,1:一般,2:重要]
        term_imp = 0
        try:
            is_same_py = False
            if ((wrong_frag != 'X' and right_frag != 'X') and isinstance(wrong_frag, str) and isinstance(right_frag, str)):
                is_same_py = is_words_same_py(wrong_frag, right_frag)
            if cand_type in [ECType.NER_PERSON.value, ECType.NER_OTHER.value, ECType.CHENGYU_SAMEPY.value, ECType.NNLM_1.value, ECType.FIXED_ERROR.value, ECType.NER_PERSON_RECALL.value]: # NNLM_1 = 17，NNLM召回固定搭配
                term_imp = 2
            elif self.fly_weight.paragraph_id <= 1 or is_same_py is False:
                term_imp = 2
            elif (len(wrong_frag) == 2 and len(right_frag) == 2 and wrong_frag[0] != right_frag[0] and wrong_frag[1] != right_frag[1]):
                term_imp = 2
            elif rs.term_imp.get(right_frag, -1) != -1:
                term_imp = 0
            else:
                term_imp = 1
        except Exception as err:
            term_imp = 1
            logging.error("logid:%s get_term_imp except:%s [%s-%s]"
                % (self.fly_weight.logid, str(err), wrong_frag, right_frag))
        return term_imp

    def __fill_item_info(self, item, basic_norm_unicode, word_idx, cand_idx, sent_info, checked_item):
        if checked_item.get((word_idx, cand_idx), -1) == 1:
            return
        basic_begin, basic_end = item.basic_begin, item.basic_end
        wrong_frag = ''.join(basic_norm_unicode[basic_begin: basic_end + 1])
        right_frag = ''.join(item.cand_str.split())
        errinfo = ErrorInfo(wrong_frag, right_frag, sent_info.short_idx, basic_begin, basic_end, item.cand_type)
        sent_info.error_infos.append(errinfo)
        checked_item[(word_idx, cand_idx)] = 1

    def norm_sent_for_ngramlm(self, sent_of_word, check_flags): # 子句归一化处理，去除标点符号等（ngramlm计算的需求）
        # strip punctuation, replace "<unk>" for unknow words
        oov_result = self.__is_words_oov(sent_of_word)
        norm_words, punc_list = [], []
        # for w, (word_gbk, is_oov) in zip(sent_of_word, oov_result):
        for idx, word in enumerate(sent_of_word):
            _, is_oov = oov_result[idx]
            if word in PUNCTUATION_LIST:
                # if PUNC_REGX.match(w) is not None:
                punc_list.append(0)    # 标点
                check_flags[idx] = -2
                continue
            norm_words.append(word)
            if is_oov == 1:
                punc_list.append(-1)    # OOV，在字典以外的字
            else:
                punc_list.append(1)    # 正常字词

            if is_chinese_string(word) is False: # 不是中文字
                check_flags[idx] = -2
                punc_list[-1] = 2      # 非汉字词

        return norm_words, punc_list

    def __get_words_score(self, words_list): # ngramlm给句子打分
        words_score = []
        for prob, _, _ in rs.ngram_lm.full_scores(' '.join(words_list).encode('gb18030'), bos=True, eos=False):
            words_score.append(prob)
        return words_score

    def diff_num(self, word1, word2): # 两个词的编辑距离
        same, diff = 0, 0
        same_lst = []
        for i in range(len(word1)):
            if word1[i] == word2[i]:
                same += 1
            else:
                diff += 1
                same_lst.append(same)
                same = 0
        same_lst.append(same)
        return diff, max(same_lst)

    def is_difftone(self, word1, word2): # 判断是否同音，True-不同音
        tone1 = lazy_pinyin(word1, style=TONE3)
        tone2 = lazy_pinyin(word2, style=TONE3)
        for i in range(len(tone1)):
            if tone1[i] != tone2[i]:
                return True
        return False

    def before_exempt(self, sent_info):
        sent_of_word = sent_info.basic_list_unicode # raw basic list
        # sent_str = ''.join(sent_of_word)
        check_flags = sent_info.check_flags # 词粒度: 每个词原长句中的flag，-4:<> -2:标点非汉字 -1:专名 0:已改 1:待检测
        # word_pos_idx = sent_info.word_pos_idx # 基本词粒度: 每个词在原长句中的字偏移，example: 北京 大学 word_pos_idx: 0 2 check_flags: 1 1
        punc_list = sent_info.punc_list # -1:OOV在字典以外的字 0:标点 1:词典汉字词 2:非汉字词数词等 3:oov
        for ci_idx in range(len(sent_of_word)): # 词索引
            # if punc_list[ci_idx] in [0, 2] or check_flags[ci_idx] <= 0: # 只检测1-待检测的切词
            #     zi_pos_idx += len(sent_of_word[ci_idx])
            #     continue

            # if (len(sent_of_word[ci_idx]) == 1 and is_chinese_string(sent_of_word[ci_idx])): # 1个字的切词
            #     if num_of_words == 0:
            #         start = ci_idx
            #     num_of_words += 1
            #     if ci_idx < len_sent_of_word - 1:
            #         continue
            # if (len(sent_of_word[ci_idx]) == 2 and is_chinese_string(sent_of_word[ci_idx])): # 2个字的切词
            #     if num_of_words == 0:
            #         start = ci_idx
            #     num_of_words += 2
            # # non_single_char word or punc occurs
            # if num_of_words <= 1:
            #     num_of_words = 0
            #     start = -1
            #     continue
            # # multi single char index from [start, start + num_of_words) 多个切词
            # for j in range(start, start + num_of_words): # num_of_words >= 2
            #     if check_flags[j] <= 0 or punc_list[j] <= 0:
            #         continue

            for win_len in range(4, 1, -1): # 窗口长度4 3 2
                if (ci_idx + win_len) > len(sent_of_word): # 词越界
                    continue
                flags_list = [check_flags[ch] for ch in range(ci_idx, ci_idx + win_len)]
                if sum(flags_list) < win_len: # 不是全为1-待检测
                    continue
                has_punc = False
                for tmp_idx in range(ci_idx, ci_idx + win_len):
                    if punc_list[tmp_idx] == 0: # 0:标点
                        has_punc = True
                        continue
                if has_punc is True: # 标点不检查
                    continue

                wrong_zi = "".join(sent_of_word[ci_idx: ci_idx + win_len]) # 左闭右开
                if len(wrong_zi)>4:
                    continue
                if wrong_zi in rs.pass_words:
                    sent_info.set_check_flags(ci_idx, ci_idx + win_len -1)

    def __detect_chengyu_error_pinyin(self, sent_info): # sent_info：短句信息：长句以逗号分隔的短句

        # tm = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        words_uni = sent_info.basic_list_unicode # raw basic list
        sent_str = ''.join(words_uni)
        check_flags = sent_info.check_flags # 词粒度: 每个词原长句中的flag，-4:<> -2:标点非汉字 -1:专名 0:已改 1:待检测
        word_pos_idx = sent_info.word_pos_idx # 基本词粒度: 每个词在原长句中的字偏移，example: 北京 大学 word_pos_idx: 0 2 check_flags: 1 1
        punc_list = sent_info.punc_list # -1:OOV在字典以外的字 0:标点 1:词典汉字词 2:非汉字词数词等 3:oov
        zi_pos_idx, find_err_num = 0, 0
        for ci_idx in range(len(words_uni)): # 词索引
            contain_single = False
            if punc_list[ci_idx] in [0, 2] or check_flags[ci_idx] <= 0:
                zi_pos_idx += len(words_uni[ci_idx])
                continue

            # not check
            zi_num = len(words_uni[ci_idx]) # 词长度
            if zi_num == 4: # 4字词
                zi_pos_idx += len(words_uni[ci_idx])
                continue
            if zi_num == 1: # 单字
                contain_single = True

            # find if there are 4 chars ahead
            cur_ci_idx = ci_idx
            begin_zi_idx = word_pos_idx[ci_idx] # 每个词在原长句中的开始字偏移
            while zi_num < 4 and cur_ci_idx + 1 < len(words_uni): # 字数小于4的词，没遍历到最后一个词
                cur_ci_idx = cur_ci_idx + 1 # 滑动到下一个词
                if punc_list[cur_ci_idx] in [0, 2]:
                    break
                zi_num += len(words_uni[cur_ci_idx])
                if len(words_uni[cur_ci_idx]) == 1:
                    contain_single = True
            if zi_num != 4:
                zi_pos_idx += len(words_uni[ci_idx])
                continue

            # 判断原始句子中字位置长度是否为4
            end_zi_idx = word_pos_idx[cur_ci_idx] + len(words_uni[cur_ci_idx]) # 每个词在原长句中的结束字偏移
            if end_zi_idx - begin_zi_idx != 4:
                zi_pos_idx += len(words_uni[ci_idx])
                continue
            # no single word, not check,  diff v1
            # badcase: 说得痛快林丽
            if contain_single is False:
                zi_pos_idx += len(words_uni[ci_idx])
                continue

            word_4 = sent_str[zi_pos_idx: zi_pos_idx + 4] # 4字的词
            # check 4-gram same pinyin
            word4_pinyin_list = lazy_pinyin(word_4)
            if len(word4_pinyin_list) < 4:
                zi_pos_idx += len(words_uni[ci_idx])
                continue

            find_error = False
            pinyin_str = ' '.join(word4_pinyin_list)
            same_py_candidates = rs.get_same_py_chengyu(pinyin_str) # 获取同音的4字的候选词
            if word_4 in same_py_candidates:
                zi_pos_idx += len(words_uni[ci_idx])
                continue
            if self.mode == 0 and contain_single is False:
                same_py_candidates = []
            for same_py_candidate in same_py_candidates:
                if word_4 == same_py_candidate:
                    continue
                wrong_ci = word_4 # 原来的4字的词为错词
                correct_ci = same_py_candidate # 同音不同字的4字候选词为正确词
                dff, continue_same = self.diff_num(wrong_ci, correct_ci)

                if dff > 2:
                    zi_pos_idx += len(words_uni[ci_idx])
                    continue

                _, ret_ratio = self.check_sent_valid_by_ppl(sent_info, ci_idx, cur_ci_idx, correct_ci, 0.01, ECType.CHENGYU_SAMEPY, False) # 计算新句子的新ppl值，判断是否有效

                if dff == 2:
                    if (ret_ratio > 1 and self.is_difftone(wrong_ci, correct_ci)): # 判断是否同音，True-不同音
                        zi_pos_idx += len(words_uni[ci_idx])
                        continue
                    # diff v1
                    if contain_single is False and ret_ratio > 0.5:
                        zi_pos_idx += len(words_uni[ci_idx])
                        continue

                elif (dff == 1 and continue_same != 3 and contain_single is False):
                    # diff v1
                    if ret_ratio > 0.9:
                        zi_pos_idx += len(words_uni[ci_idx])
                        continue
                # re-label check_flag
                sent_info.set_check_flags(ci_idx, cur_ci_idx) # # 设置当前词索引位置为0-已改
                sent_info.add_cand(ci_idx, cur_ci_idx, correct_ci, ECType.CHENGYU_SAMEPY, 1 - ret_ratio) # 添加候选词，0-成语识别
                find_error = True
                find_err_num += 1
                break

            if find_error is True:
                zi_pos_idx += len(words_uni[ci_idx])
                continue

            zi_pos_idx += len(words_uni[ci_idx])

        return find_err_num

    def __split_sent(self, sent_of_word_unicode, ch_flags,
                     word_pos_idx, basic_pos):
        """
        将sentence切分为短句，并处理《》“”等内部字符串
        """
        short_sents, ss_flags, ss_pos_idx, ss_part_of_speech = [], [], [], []
        prev = -1
        # 全半角、大小写转换等格式化处理
        sent_of_word_unicode = [uniform(w) for w in sent_of_word_unicode]
        len_sent = len(sent_of_word_unicode)
        for i in range(len_sent):
            if sent_of_word_unicode[i] in [u'\u002c', u'\uff0c', u'\u0020']:
                short_sents.append(sent_of_word_unicode[prev + 1: i])
                ss_flags.append(ch_flags[prev + 1: i])
                ss_pos_idx.append(word_pos_idx[prev + 1: i])
                ss_part_of_speech.append(basic_pos[prev + 1: i])
                prev = i
        if prev + 1 < len_sent:
            # 去除最后的标点
            if (prev + 1 < len_sent - 1 and sent_of_word_unicode[len_sent - 1] in PUNCTUATION_LIST):
                len_sent -= 1
            short_sents.append(sent_of_word_unicode[prev + 1: len_sent])
            ss_flags.append(ch_flags[prev + 1: len_sent])
            ss_pos_idx.append(word_pos_idx[prev + 1: len_sent])
            ss_part_of_speech.append(basic_pos[prev + 1: len_sent])

        for i in range(len(short_sents)):
            ldq_begin, lsm_begin, lsq_begin, lbq_begin = -1, -1, -1, -1
            leq_begin, lyc_begin, lye_begin = -1, -1, -1
            for j in range(len(short_sents[i])):
                cur_char = short_sents[i][j]
                if cur_char == '\"':
                    if leq_begin == -1:
                        leq_begin = j
                    else:
                        self.set_check_flag(ss_flags[i], leq_begin, j, -4, ss_pos_idx[i])
                        leq_begin = -1

                if cur_char == '《':
                    lsm_begin = j
                elif cur_char == '》':
                    if lsm_begin != -1 and lsm_begin < j:
                        self.set_check_flag(ss_flags[i], lsm_begin, j, -4, ss_pos_idx[i])
                        lsm_begin = -1

                if cur_char == '“':
                    ldq_begin = j
                elif cur_char == '”':
                    if ldq_begin != -1 and ldq_begin < j:
                        self.set_check_flag(ss_flags[i], ldq_begin, j, -4, ss_pos_idx[i])
                        ldq_begin = -1

                if cur_char == '‘':
                    lsq_begin = j
                elif cur_char == '’':
                    if lsq_begin != -1 and lsq_begin < j:
                        self.set_check_flag(ss_flags[i], lsq_begin, j, -4, ss_pos_idx[i])
                        lsq_begin = -1

                if cur_char == '【':
                    lbq_begin = j
                elif cur_char == '】':
                    if lbq_begin != -1 and lbq_begin < j:
                        self.set_check_flag(ss_flags[i], lbq_begin, j, -4, ss_pos_idx[i])
                        lbq_begin = -1

                if cur_char == '（':
                    lyc_begin = j
                elif cur_char == '）':
                    if lyc_begin != -1 and lyc_begin < j:
                        self.set_check_flag(ss_flags[i], lyc_begin, j, -4, ss_pos_idx[i])
                        lyc_begin = -1

                if cur_char == '(':
                    lye_begin = j
                elif cur_char == ')':
                    if lye_begin != -1 and lye_begin < j:
                        self.set_check_flag(ss_flags[i], lye_begin, j, -4, ss_pos_idx[i])
                        lye_begin = -1
        return short_sents, ss_flags, ss_pos_idx, ss_part_of_speech

    def set_check_flag(self, check_flags, begin, end, flag, ss_pos_idx):
        # 句子<<>>等符号内字、词修改豁免
        for i in range(begin, end + 1):
            check_flags[i] = flag
        bpos, epos = ss_pos_idx[begin], ss_pos_idx[end]
        char_check_flags = self.fly_weight.char_check_flags
        for i in range(bpos, epos + 1):
            char_check_flags[i] = -1 # 设置为专名

    def check_sent_valid_by_ppl(self, sent_info, in_begin, in_end, cand_unicode, ppl_ratio, cand_type, add_cand=True): # cand_unicode是用候选词，如“贵 正”

        # ppl_ratio:  decrease to the ratio of raw ppl
        # @param in_begin: 归一化前的原句分词偏移位置
        raw_to_norm = sent_info.raw_to_norm
        idx_begin = raw_to_norm[in_begin]
        idx_end = raw_to_norm[in_end]
        words_norm = sent_info.basic_norm_unicode
        center = cand_unicode.split() # center=['贵','正']

        is_valid, is_exempt = False, False
        wrong_para = ''.join(words_norm[idx_begin: idx_end + 1])
        correct_para = ''.join(center) # 贵正
        if (wrong_para == correct_para or len(wrong_para) != len(correct_para)):
            return is_valid, 1.0
        # 替换候选词后的新句子
        new_sent = []
        new_sent.extend(words_norm[: idx_begin])
        new_sent.extend(center)
        new_sent.extend(words_norm[idx_end + 1:])
        # new_sent = before + center + after
        # 计算新句子的新ppl值
        new_ppl = rs.get_ppl_score(new_sent)
        ppl = sent_info.raw_ppl
        ratio = new_ppl / ppl
        if new_ppl < ppl * ppl_ratio:
            is_exempt = self.check_is_exempt(wrong_para, correct_para, cand_type)
            is_valid = True
            if add_cand and is_exempt is False:
                sent_info.add_cand(in_begin, in_end, cand_unicode, cand_type, 1 - ratio, False) # 添加候选词

        if (self.debug_mode & 2 == 2 and is_valid is False and is_exempt is False):
            if in_begin == in_end:
                if in_begin not in sent_info.debug_info['cands']:
                    sent_info.debug_info['cands'][in_begin] = []
                sent_info.debug_info['cands'][in_begin].append((in_begin, in_end, cand_unicode, cand_type.value, ratio, is_valid))
            else:
                if in_begin not in sent_info.debug_info['frag_cands']:
                    sent_info.debug_info['frag_cands'][(in_begin, in_end)] = []
                sent_info.debug_info['frag_cands'][(in_begin, in_end)].append((in_begin, in_end, cand_unicode, cand_type.value, ratio, is_valid))

        return is_valid, ratio

    def check_is_exempt(self, wrong_ci, correct_ci, in_cand_type): # 判断是否豁免

        ret, cand_type = False, in_cand_type.value
        if (cand_type in [ECType.SINGLECHAR_NONWORD.value, ECType.SINGLECHAR_NONWORD_JY.value]): # 1-单字非词，51-
            wrong_ci_char = wrong_ci
            if len(wrong_ci) == 2 and len(correct_ci) == 2:
                if wrong_ci[0] == correct_ci[0]:
                    wrong_ci_char = wrong_ci[1]
                elif wrong_ci[1] == correct_ci[1]:
                    wrong_ci_char = wrong_ci[0]
            if (wrong_ci_char in rs.pass_chars or wrong_ci_char in rs.corr_chars):
                ret = True
        if (cand_type in [ECType.SINGLECHAR_WORD.value, ECType.SINGLECHAR_WORD_JY.value, ECType.SINGLECHAR_WORD_BERT.value]): # 2-单字词，6-单字判别模型
            if wrong_ci in rs.pass_chars or wrong_ci in rs.corr_chars: # 豁免：他 她 它 嘛 吗 呢 啊 么 呀 阿 哒 忒 似 式 之 亦 俺
                ret = True
            elif correct_ci in ['他', '她', '之', '阿', '呀']:
                ret = True
        elif (len(wrong_ci) > 1 and (cand_type in [ECType.WORD_SAMEPY.value, ECType.WORD_BERT.value, ECType.WORD_SIMILARPY.value])): # 3-同音词，4-近音词，5-词语判别模型
            if (wrong_ci in self.doc_ner_all or wrong_ci in rs.pass_words or wrong_ci in rs.corr_words):
                ret = True
            elif (wrong_ci.endswith(u'哥') or wrong_ci.endswith(u'姐') or wrong_ci.endswith(u'氏') or wrong_ci.endswith(u'妃') or wrong_ci.startswith(u'小')):
                ret = True
        return ret

    def __detect_single_char_nonword_error(
            self, sent_info, ppl_threshold_by_len):
        # 识别类型: 词语错误且切分为单字
        # eg 她很美里-> 她很美丽
        tms = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        if len(sent_info.basic_norm_unicode) < 3:
            return

        sent_of_word = sent_info.basic_list_unicode
        len_sent_of_word = len(sent_of_word)
        check_flags = sent_info.check_flags
        punc_list = sent_info.punc_list
        top_dict = rs.top_dict
        start, num_of_words = -1, 0
        for i in range(len(sent_of_word)):
            # char_word
            if (len(sent_of_word[i]) == 1 and is_chinese_string(sent_of_word[i])):
                if num_of_words == 0:
                    start = i
                num_of_words += 1
                if i < len_sent_of_word - 1:
                    continue
            # non_single_char word or punc occurs
            if num_of_words <= 1:
                num_of_words = 0
                start = -1
                continue

            # multi single char index from [start, start + num_of_words) 多个单字
            for j in range(start, start + num_of_words): # num_of_words >= 2
                if check_flags[j] <= 0 or punc_list[j] <= 0:
                    continue

                if sent_of_word[j] in rs.sig_char:
                    self.__detect_single_char_nonword_tongyin(sent_of_word, j, sent_info)
                    continue
                if (j + 1 < len(sent_of_word) and sent_of_word[j] + sent_of_word[j + 1] in top_dict): # 如果相邻两个单字拼在一起，在top_dict字典中，则设置为0-已改
                    sent_info.set_check_flags(j, j) # 设置j位置为0-已改
                    continue
                if j > 0 and sent_of_word[j - 1] + sent_of_word[j] in top_dict:
                    sent_info.set_check_flags(j, j) # 设置j位置为0-已改
                    continue
                # for tongyin candidate
                is_find_cand = self.__detect_single_char_nonword_tongyin(sent_of_word, j, sent_info)
                if is_find_cand:
                    continue
                # for jinyin candidate
                is_find_cand = self.__detect_single_char_nonword_jinyin(sent_of_word, j, sent_info)
                if is_find_cand:
                    continue

            # processed char word
            num_of_words = 0
            start = -1
        del tms
        return

    def __detect_single_char_nonword_tongyin(self, sent_of_word, j, sent_info): # 识别单字非词的同音错误，如“北京是手都” 中“首都”错写成“手都”
        # for tongyin candidate
        punc_list = sent_info.punc_list
        check_flags = sent_info.check_flags
        top_dict = rs.top_dict
        tongyin_cand_words = rs.get_tongyin_ci_zi(sent_of_word[j]) # 同音字
        tongyin_ppl_ratio, is_valid_cand = 0.7, False
        for cand in tongyin_cand_words:
            if (j + 1 < len(sent_of_word) and check_flags[j + 1] > 0 and punc_list[j + 1] > 0 and cand + sent_of_word[j + 1] in top_dict):
                cand_ci = cand + sent_of_word[j + 1] # 替换前字后的新词
                is_valid_cand, _ = self.check_sent_valid_by_ppl(sent_info, j, j + 1, cand_ci, tongyin_ppl_ratio, ECType.SINGLECHAR_NONWORD) # 对每个候选词，计算新ppl值，判断添加该候选词是否有效。
                if is_valid_cand:
                    if self.mode == 0:
                        sent_info.set_check_flags(j, j + 1) # 设置j和j+1位置为0-已改
                    break

            # from start to begin
            if (j > 0 and check_flags[j - 1] > 0 and punc_list[j - 1] > 0 and sent_of_word[j - 1] + cand in top_dict):
                cand_ci = sent_of_word[j - 1] + cand # 替换后字后的新词
                is_valid_cand, _ = self.check_sent_valid_by_ppl(sent_info, j - 1, j, cand_ci, tongyin_ppl_ratio, ECType.SINGLECHAR_NONWORD)
                if is_valid_cand:
                    if self.mode == 0:
                        sent_info.set_check_flags(j - 1, j)
                    break
        return is_valid_cand

    def __detect_single_char_nonword_jinyin(self, sent_of_word, j, sent_info): # 识别单字非词的近音错误
        # for jinyin candidate
        jinyin_cand_words = rs.get_jinyin_zi(sent_of_word[j]) # 近音字
        word_dict = rs.word_freq
        check_flags = sent_info.check_flags
        ppl_ratio, is_valid_cand = 0.5, False
        punc_list = sent_info.punc_list
        for cand in jinyin_cand_words:

            if (j + 1 < len(sent_of_word) and check_flags[j + 1] > 0 and punc_list[j + 1] > 0 and cand + sent_of_word[j + 1] in word_dict):
                cand_ci = cand + sent_of_word[j + 1] # 替换前字后的新词
                is_valid_cand, _ = self.check_sent_valid_by_ppl(sent_info, j, j + 1, cand_ci, ppl_ratio, ECType.SINGLECHAR_NONWORD_JY)
                if is_valid_cand:
                    if self.mode == 0:
                        sent_info.set_check_flags(j, j + 1)
                    break
            if (j > 0 and check_flags[j - 1] > 0 and punc_list[j - 1] > 0 and sent_of_word[j - 1] + cand in word_dict):
                cand_ci = sent_of_word[j - 1] + cand # 替换后字后的新词
                is_valid_cand, _ = self.check_sent_valid_by_ppl(sent_info, j - 1, j, cand_ci, ppl_ratio, ECType.SINGLECHAR_NONWORD_JY)
                if is_valid_cand:
                    if self.mode == 0:
                        sent_info.set_check_flags(j - 1, j)
                    break
        return is_valid_cand

    def __detect_single_char_word_error(self, sent_info, ppl_ratio, ppl_threshold_by_len, classify_result):
        # 识别类型: 单字错误 eg:他再吃饭->他在吃法
        tms = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        find_cand_num = 0
        if len(sent_info.basic_norm_unicode) < 4:
            return find_cand_num
        words_uni = sent_info.basic_list_unicode
        ch_flags = sent_info.check_flags
        punc_list = sent_info.punc_list
        find_cand, jy_ppl_ratio = False, 0.45
        go_classify, bert_idx, max_dec_ratio = False, -1, 0.0
        frag_right = ''
        for j in range(len(words_uni)):
            find_cand = False
            if (ch_flags[j] > 0 and punc_list[j] > 0 and len(words_uni[j]) == 1 and is_chinese_string(words_uni[j])): # 单字
                cand_words = rs.get_tongyin_ci_zi(words_uni[j]) # 同音字
                tmp_cacl_cnt = 0
                for cand in cand_words:
                    is_valid_sent, ret_ratio = self.check_sent_valid_by_ppl(sent_info, j, j, cand, ppl_ratio, ECType.SINGLECHAR_WORD) # 2-单字词——同音字
                    tmp_cacl_cnt += 1
                    if is_valid_sent is True:
                        find_cand = True # 找到候选的同音字
                        if self.mode == 0:
                            sent_info.set_check_flags(j, j) # 设置j位置为0-已改
                        break
                    if (is_valid_sent is False and ret_ratio < ppl_threshold_by_len): # ret_ratio即ppl降比
                        go_classify = True
                        dec_ratio = 1 - ret_ratio
                        if dec_ratio > max_dec_ratio: # 更新最大ppl降比
                            max_dec_ratio = dec_ratio
                            frag_right = cand
                            bert_idx = j
                if self.mode == 0 and find_cand is True:
                    find_cand_num += 1
                    break

                cand_words = rs.get_jinyin_zi(words_uni[j]) # 近音字
                tmp_cacl_cnt = 0
                for cand in cand_words:
                    is_valid_sent, ret_ratio = self.check_sent_valid_by_ppl(sent_info, j, j, cand, jy_ppl_ratio, ECType.SINGLECHAR_WORD_JY, False) # 为什么这里不判断is_valid_sent is True ?  52-单字词——近音字
                    tmp_cacl_cnt += 1
                    if ret_ratio < ppl_threshold_by_len:
                        go_classify = True
                        dec_ratio = 1 - ret_ratio
                        if dec_ratio > max_dec_ratio:
                            max_dec_ratio = dec_ratio
                            frag_right = cand
                            bert_idx = j
                        # break #add break

        if not find_cand and go_classify is True:
            ret_num = self.__check_suspect_error(sent_info, classify_result, bert_idx, bert_idx, frag_right, ECType.SINGLECHAR_WORD_BERT, max_dec_ratio) # 6-单字判别模型
            find_cand_num += ret_num
        del tms
        return find_cand_num

    def __detect_word_error(self, sent_info, tongyin_ppl_ratio, jinyin_ppl_ratio, ppl_threshold_by_len, classify_result):
        # 识别类型:词语错误 她/很/梅里/-> /她/很/美丽
        tms = timer(sys._getframe().f_code.co_name, self.fly_weight.logid)
        find_cand_num = 0
        if len(sent_info.basic_norm_unicode) < 5:
            return find_cand_num
        words_uni = sent_info.basic_list_unicode
        ch_flags = sent_info.check_flags
        punc_list = sent_info.punc_list
        go_classify, max_dec_ratio = False, 0
        frag_right, bert_idx = '', -1
        find_cand = False  # tongyin_ppl_ratio = 0.7
        for j in range(len(words_uni)):
            find_cand = False
            if punc_list[j] in [-1, 0, 2]:
                continue
            # if words_uni[j] in self.ent_tags:
            #    continue
            if (len(words_uni[j]) > 1 and ch_flags[j] > 0 and is_chinese_string(words_uni[j])): # 非单字
                cand_words = rs.get_tongyin_ci_zi(words_uni[j]) # 同音词
                for cand in cand_words:
                    is_valid_sent, ret_ratio = self.check_sent_valid_by_ppl(sent_info, j, j, cand, tongyin_ppl_ratio, ECType.WORD_SAMEPY, False) # 3-同音词
                    if is_valid_sent:
                        if rs.is_similar_word(cand, words_uni[j]): # word2vec的余弦相似度大于阈值才认为相似
                            continue
                        find_cand = True # 找到候选的同音词

                        is_exempt = self.check_is_exempt(words_uni[j], cand, ECType.WORD_SAMEPY) # 3-同音词
                        if is_exempt is False:
                            sent_info.add_cand(j, j, cand, ECType.WORD_SAMEPY, 1 - ret_ratio)
                        if self.mode == 0:
                            sent_info.set_check_flags(j, j)
                            break

                    if ret_ratio < ppl_threshold_by_len:
                        go_classify = True
                        dec_ratio = 1 - ret_ratio
                        if dec_ratio > max_dec_ratio:
                            max_dec_ratio = dec_ratio
                            frag_right = cand
                            bert_idx = j
                        # break #add break

                if find_cand:
                    find_cand_num += 1
                    break

                cand_words = rs.get_jinyin_ci_conf_ft(words_uni[j]) # 近音词
                for cand in cand_words:
                    is_valid_sent, ret_ratio = self.check_sent_valid_by_ppl(sent_info, j, j, cand, jinyin_ppl_ratio, ECType.WORD_SIMILARPY, False) # 4-近音词
                    if is_valid_sent:
                        if rs.is_similar_word(cand, words_uni[j]):
                            continue

                        is_exempt = self.check_is_exempt(words_uni[j], cand, ECType.WORD_SIMILARPY)
                        sent_info.add_cand(j, j, cand, ECType.WORD_SIMILARPY, 1 - ret_ratio, False, is_exempt)
                        find_cand = True
                        if self.mode == 0:
                            sent_info.set_check_flags(j, j)

                        break
                    if (is_valid_sent is False and ret_ratio < ppl_threshold_by_len): # 如果没有达到直接接受的阈值，但在一个范围内，则请求序列标注模型判断错误位置
                        go_classify = True
                        dec_ratio = 1 - ret_ratio
                        if dec_ratio > max_dec_ratio:
                            max_dec_ratio = dec_ratio
                            frag_right = cand
                            bert_idx = j
                        # break #add break

                if find_cand:
                    find_cand_num += 1
                    break

        if not find_cand and go_classify:
            ret_num = self.__check_suspect_error(sent_info, classify_result, bert_idx, bert_idx, frag_right, ECType.WORD_BERT, max_dec_ratio) # 5-词语判别模型
            find_cand_num += ret_num
        del tms
        return find_cand_num

    @classmethod
    def need_reload_dict(self):
        return True

    def is_frag_checked(self, in_begin, in_end, cand_type): # 错误类型

        checked_item = self.checked_dict.get((in_begin, in_end), [])
        if cand_type.value in checked_item:
            return True
        if len(checked_item) > 0:
            self.checked_dict[(in_begin, in_end)].append(cand_type.value) # 候选类型 -1原词, 0成语 1单字非词 2单字 3同义词 4近义词
        else:
            self.checked_dict[(in_begin, in_end)] = [cand_type.value]
        return False

    def check_adjacent_word(self, basic_list_unicode, idx, word_cand):
        top_dict = rs.top_dict # 我们 12920073 自己 12567910
        ret = []
        newword = basic_list_unicode[idx - 1] + \
            word_cand if idx - 1 >= 0 else ''
        if newword != '' and newword in top_dict:
            ret.append((idx - 1, idx, newword))

        if idx + 1 < len(basic_list_unicode):
            newword = word_cand + basic_list_unicode[idx + 1]
        else:
            newword = ''
        if newword != "" and newword in top_dict:
            ret.append((idx, idx + 1, newword))
        return ret

    def __fill_info(self, sent_info):
        basic_norm_unicode = sent_info.basic_norm_unicode
        len_basic_norm = len(basic_norm_unicode)
        cands_list = sent_info.cands_list

        topk_list = Topk(5)
        # topk_list.push((sent_info.raw_ppl, sent_info.basic_norm_gbk, []))
        stack = Stack()
        checked_item = {}
        # 记录当前遍历词的index, 记录已遍历词列表
        word_idx, word_list = 0, []
        while True:
            while word_idx < len_basic_norm: # 先一次性入栈
                stack.push((word_idx, 0))
                item = cands_list[word_idx][0] # cands_list[i][0] 为原句分词后第i个词信息
                word_list.append(item.cand_str) # cand_str：unicode编码
                word_idx = item.basic_end + 1

            if word_idx >= len_basic_norm:
                tmp_list = ' '.join(word_list).split() # ' '.join用空格间隔将列表转字符串，split()不带参数时以空字符（包括空格、换行(\n)、制表符(\t)等）分割
                score = rs.get_ppl_score(tmp_list) # 由分词组成的句子的ppl分数
                topk_list.push((score, tmp_list, copy.deepcopy(stack.items))) # 将分数压入栈中

            if stack.is_empty():
                break
            word_idx, cand_idx = stack.pop()
            word_list.pop()
            cur_cand_item = cands_list[word_idx][cand_idx] # 为原句分词后第word_idx个遍历词的第cand_idx个候选词
            if (cand_idx > 0 and cur_cand_item.cand_type in [ECType.CHENGYU_SAMEPY, ECType.NER_PERSON, ECType.NER_OTHER]):
                # 成语、人名等规则召回、不参与排序
                self.__fill_item_info(cur_cand_item, basic_norm_unicode, word_idx, cand_idx, sent_info, checked_item)

            # find the next valid element
            while (not stack.is_empty() and (cand_idx >= len(cands_list[word_idx]) - 1 or cur_cand_item.cand_type in [ECType.CHENGYU_SAMEPY, ECType.NER_PERSON, ECType.NER_OTHER])):
                word_idx, cand_idx = stack.pop()
                word_list.pop()

                cur_cand_item = cands_list[word_idx][cand_idx]
                if (cand_idx > 0 and cur_cand_item.cand_type in [ECType.CHENGYU_SAMEPY, ECType.NER_PERSON, ECType.NER_OTHER]):
                    self.__fill_item_info(cur_cand_item, basic_norm_unicode, word_idx, cand_idx, sent_info, checked_item)

            if cand_idx < len(cands_list[word_idx]) - 1: # 下一个候选词入栈
                cand_idx += 1 
                stack.push((word_idx, cand_idx))
                item = cands_list[word_idx][cand_idx]
                word_list.append(item.cand_str)
                word_idx = item.basic_end + 1

            if stack.is_empty():
                break

        top_cands = topk_list.get_top_k() # ppl值最小的k个候选词
        rawppl = sent_info.raw_ppl
        processed_str_list, has_raw_sent = [], False
        basic_norm_str = ''.join(basic_norm_unicode) # 转成字符串
        for score, word_list, _ in top_cands:
            if float(score) > rawppl:
                break
            cand_str = ''.join(word_list) # 候选词转成字符串
            if cand_str in processed_str_list:
                continue
            processed_str_list.append(cand_str) # 已遍历的候选词添加至列表
            is_raw_sent = 0
            if basic_norm_str == cand_str:
                is_raw_sent = 1
                has_raw_sent = True

            sent_info.topk_cand_list.append((cand_str, score, is_raw_sent)) # topk_cand_list：纠错topk候选句

        if has_raw_sent is False:
            sent_info.topk_cand_list.append((basic_norm_str, rawppl, 1))


if __name__ == '__main__':
    processor = LmProcessor()
    lst = ['北京', '大学', '怎么', '样']
    check_flags = [0, 0, 0, 0]
    print(' '.join(lst))
    ss_sent_gbk_norm, punc_list = processor.norm_sent_for_ngramlm(lst, check_flags) # 子句归一化处理，去除标点符号等（ngramlm计算的需求）
    print(type(ss_sent_gbk_norm[0]))
    print(' '.join(ss_sent_gbk_norm))
    ppl = rs.ngram_lm.perplexity(' '.join(ss_sent_gbk_norm))
    print(ppl)
