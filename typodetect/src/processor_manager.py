# -*- coding:utf-8 -*-

import os
import logging
import timeit
import traceback
from .lm_processor import LmProcessor
from ..utils.monitor_utils import update_monitor_info


class ProcessorManager():

    def __init__(self):
        # warning: do not change processor order
        self.processors = [LmProcessor()]

    def preprocess(self, fly_weight):
        pass

    def process(self, fly_weight):
        for processor in self.processors:
            processor.process(fly_weight)

    def afterprocess(self, fly_weight):
        pass

    def run_item(self, sent_info): # sent_info：flyweight
        err_infos = []
        ret = 0
        start_time = timeit.default_timer()
        try:
            self.process(sent_info) # 调用LmProcessor
            err_infos = sent_info.error_infos
        except Exception as err:
            err_detail = traceback.format_exc()
            logging.error("run_exceptions:%s docid:%s detail:%s"
                          % (str(err), sent_info.doc_id, err_detail))
            ret = -1
        elapse = (timeit.default_timer() - start_time) * 1000
        update_monitor_info(elapse, ret, log_id=sent_info.doc_id, module='gsentence', bid=sent_info.bid) # 更新监控信息
        logging.info('sent_cost:%d docid:%s[%d]' % (elapse, sent_info.doc_id, sent_info.sent_no))
        return err_infos

    def run_debug_item(self, sent_info):
        err_infos = []
        try:
            sent_info.is_debug = True
            self.process(sent_info)
            err_infos = sent_info.debug_info
        except Exception as err:
            err_detail = traceback.format_exc()
            logging.error("run_exceptions: %s docid:%s detail:%s"
                          % (str(err), sent_info.doc_id, err_detail))
        return err_infos

    def post_process(self, wrong_details, docid, data_type): # docid：文章ID  data_type：数据类型 0-图文 1-短视频 2-小视频
        try:
            wrong_zi_dict = {}
            # 统计错误字词频次
            for _, _, _, word, _, _, _, _, _, _ in wrong_details:
                if word in wrong_zi_dict:
                    wrong_zi_dict[word] += 1
                else:
                    wrong_zi_dict[word] = 1
            title_num, fst_num, all_num = 0, 0, 0
            final_result = []
            # 具体类型见flyweight定义
            highpr_thres = {
                0: 0, 1: 0.65, 51: 0.8, 52: 1, 2: 0.85,
                3: 0.9, 4: 0.9, 7: 0, 57: 0, 9: 0, 10: 0.5, 11: 0.7,
                12: 0.5, 17: 1, 23: 1} # 高准高召阈值
            popdelay_thres = {
                0: 0, 1: 0.7, 5: 0.4, 6: 0.45, 51: 0.5,
                52: 0.5, 2: 0.6, 3: 0.5, 4: 0.5, 7: 0, 57: 0, 9: 0,
                10: 0.2, 11: 0.5, 12: 0.5, 17: 0.5, 23: 0.5, 20: 0.9} # 是否延迟分发的阈值
            # title_thres = {0: 0, 1: 0.7, 5: 0.4, 6: 0.45, 51: 0.5, 52: 0.5, 2: 0.6, 3: 0.5, 4: 0.5, 7: 0, 57: 0, 9: 0, 10: 0.2, 11: 0.5, 12: 0.5, 17: 0.5, 23: 0.5, 20: 0.9} # 标题的阈值，相对正文的阈值要提高点
            doc_type, doc_type_score = 0, 0  # 文章粒度高召回或高准确
            is_doc_pop_delay, pop_delay_score = 0, 0  # 文章粒度是否延迟分发
            for (sno_, idx_, original_sentence, wrong_ci, correct_ci, errtype, ratio, ori_sent, term_imp, wlevel) in wrong_details:
                # 文章中检测多次重复字词错误，非指定类型错误则忽略
                if wrong_ci in wrong_zi_dict and wrong_zi_dict[wrong_ci] > 2:
                    if errtype not in [0, 7, 9, 10, 15, 57]:
                        continue
                    else:
                        wlevel = 1

                    logging.info('skip_final_err%d\t%s\t%d\t%s\t%s\t%d'
                                 '\t%s\t%s\t%d\t%f\t%d\n'
                                 % (os.getpid(), docid, sno_, original_sentence, ori_sent, idx_, wrong_ci, correct_ci, errtype, ratio, wlevel))
                if sno_ == 0:
                    title_num += 1
                else:
                    fst_num += 1
                level = 1  # 1-高召 2-重准
                # 是否延迟分发 0-不延迟分发  1-延迟分发
                is_pop_delay = 0 if wlevel != 3 else 1

                # 根据文章类型，错误类型及识别概率区分属于高召/高准版
                if data_type == 1:
                    # 短视频
                    if wlevel in [0, 1, 3]:
                        level = 1
                    elif wlevel == 2:
                        level = 2
                        is_pop_delay = 1
                elif data_type == 0:
                    # 图文
                    if (ratio < highpr_thres.get(errtype, 1.0) or wlevel == 1):
                        level = 1
                    else:
                        level = 2

                    if (ratio >= popdelay_thres.get(errtype, 0.2) and level == 1):
                        is_pop_delay = 1
                final_result.append((sno_, idx_, original_sentence, wrong_ci, correct_ci, errtype, ratio, ori_sent, term_imp, level, is_pop_delay))
                # 高准
                if level == 2:
                    if sno_ == 0:
                        doc_type_score += 3
                    else:
                        doc_type_score += 1
                elif level == 1 and sno_ == 0:
                    doc_type = 1
                # 延迟分发
                if is_pop_delay == 1:
                    if sno_ == 0:
                        pop_delay_score += 3
                    else:
                        pop_delay_score += 1

            # 区分文章粒度高准、高召、或延迟分发
            if doc_type_score >= 3:
                doc_type = 2
            elif (len(final_result) >= 3 or (data_type == 1 and len(final_result) > 0)):
                # 图文三个以上错别字、视频一个错别字
                doc_type = 1
            if pop_delay_score >= 3:
                is_doc_pop_delay = 1
            all_num = len(final_result)
            response_dict = dict()
            response_dict["details"] = final_result # 错别字信息
            response_dict["title_num"] = title_num
            response_dict["fst_num"] = fst_num
            response_dict["all_num"] = all_num # 错别字个数
            response_dict["typo_level"] = doc_type # 文章粒度高召回或高准确
            response_dict["typo_pop_result"] = is_doc_pop_delay # 文章粒度是否延迟分发

            for (sno, _, sent, wrong, corr, err_type, ratio, ori_sent, term_imp, level, is_pop) in final_result:
                logging.info('typos_detail %s\t%d\t%s\t%s\t%s\t%s'
                             '\t%d\t%f\t%d\t%d\t%d'
                             % (docid, sno, sent, ori_sent, wrong, corr, err_type, ratio, term_imp, level, is_pop))
        except Exception as err:
            traceback.print_exc()
            logging.error("postprocess_exceptions: %s docid:%s"
                          % (str(err), docid))

        return response_dict
