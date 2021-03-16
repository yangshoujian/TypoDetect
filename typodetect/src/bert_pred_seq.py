# -*- coding: utf-8 -*-
import sys
import logging
import time
import traceback
from io import BytesIO
import pycurl
import numpy as np
import l5sys
from ..utils.time_utils import timer
from ..utils.monitor_utils import update_monitor_info
from ..utils import tokenization

TOKENIZAER = tokenization.FullTokenizer(vocab_file='./data/bert_vocab.txt', do_lower_case=True)


def request_trt_client(host, port, model_name, payload,
                       seq_len, batch_size, class_len, logid):
    ret = None
    try:
        request_api = 'http://{}:{}/api/infer/{}'.format(host, port, model_name)
        header = ('batch_size: %d input { name: "input_ids" dims: [%d]} '
                  'input {name: "input_mask" dims: [%d]} '
                  'input {name: "segment_ids" dims: [%d]} '
                  'output {name: "output"}'
                  % (batch_size, seq_len, seq_len, seq_len))
        curl = pycurl.Curl() # #创建一个pycurl对象的方法
        curl.setopt(curl.URL, request_api)
        byte_str = BytesIO()
        curl.setopt(pycurl.POST, 1)
        curl.setopt(pycurl.TIMEOUT_MS, 200)
        curl.setopt(pycurl.HTTPHEADER, ['NV-InferRequest:' + header])
        curl.setopt(pycurl.WRITEFUNCTION, byte_str.write)
        curl.setopt(pycurl.POSTFIELDS, payload)
        curl.setopt(pycurl.VERBOSE, 0)
        curl.perform()
        result = np.frombuffer(byte_str.getvalue(), count=class_len * batch_size * seq_len, dtype='float32') # 实现动态数组
        ret = result.reshape([batch_size, seq_len, class_len])
    except Exception as ex:
        traceback.print_exc()
        logging.warning("logid:%s api:%s msg:%s"
                        % (logid, request_api, ex))
    return ret


def make_bert_feature(sent, seq_len, ptokenizer=TOKENIZAER):

    u_sent = sent.decode('utf-8')
    _tokens = []
    segment_ids = []
    real_n = 0
    chars = []

    _tokens.append("[CLS]")
    segment_ids.append(0)
    for token in u_sent:
        ori_token = token
        token = ptokenizer.tokenize(token)
        if len(token) < 1:
            continue
        for _, _w in enumerate(token):
            _tokens.append(_w)
            real_n += 1
            segment_ids.append(0)
            chars.append(ori_token)

    if len(_tokens) > seq_len - 1:
        _tokens = _tokens[0: seq_len - 1]
        segment_ids = segment_ids[0: seq_len - 1]
        chars = chars[0: seq_len - 2]
        # 不算[cls] [sep]
        real_n = seq_len - 2

    _tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = ptokenizer.convert_tokens_to_ids(_tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    # print (' '.join(_tokens))
    # print (' '.join([str(i) for i in input_ids]))
    # print (' '.join([str(i) for i in input_mask]))
    # print (' '.join([str(i) for i in segment_ids]))
    return (input_ids, input_mask, segment_ids, [tmp.encode("utf-8") for tmp in chars], real_n)


def get_trt_result(host_ip, host_port, sent_u8, sno, index_to_index_dict, logid):
    # timer(sys._getframe().f_code.co_name)
    ret, classify_pos_ratio, judge_type = -1, -1.0, 0
    need_tag_pos_char_list = []
    need_tag_pos_prob_list = []
    need_tag_pos_index_list = []
    need_tag_pos_word_list = []

    input_ids, input_mask, segment_ids, _, real_n = make_bert_feature(sent_u8, 128)
    model_name = 'seq_8_128'
    seq_len, batch_size, class_size = 128, 1, 4
    payload = np.array((input_ids * batch_size + input_mask * batch_size + segment_ids * batch_size), dtype='int32').tobytes()
    result = request_trt_client(host_ip, host_port, model_name, payload, seq_len, batch_size, class_size, logid)

    # O, X, E, B
    # print(real_n)
    # print(result[0][1 : 1 + real_n][:, 3])
    # print len(chars_list), ' '.join(chars_list)
    # print result
    if result is not None and len(result) > 0:
        sent_prob = result[0][1: 1 + real_n][:, 3]
        char_lst = [ch for ch in sent_u8.decode('utf8')]
        char_err_ret = np.where(np.array(result[0][1: 1 + real_n][:, 3] > 0.5))
        char_err_idxs = char_err_ret[0] if len(char_err_ret) > 0 else []

        # print (' '.join(char_lst))
        # print (char_err_ret)
        # print('len_idx:', char_err_idxs)
        for i in char_err_idxs:
            need_tag_pos_char_list.append(char_lst[i])
            need_tag_pos_prob_list.append(float(sent_prob[i]))
            need_tag_pos_index_list.append(i)
            need_tag_pos_word_list.append(index_to_index_dict[i])
        ret = 0
    else:
        ret = -1
    classify_pos_ratio = max(need_tag_pos_prob_list) if len(need_tag_pos_prob_list) > 0 else 0.0
    judge_type = 1 if classify_pos_ratio > 0.5 else 0
    # print('trt_result')
    # print(judge_type, classify_pos_ratio)
    # print(' '.join(need_tag_pos_char_list))
    # print(' '.join([str(i) for i in need_tag_pos_prob_list]))
    # print(' '.join([str(i) for i in need_tag_pos_index_list]))
    # print(' '.join(need_tag_pos_word_list))
    if judge_type == 1:
        logging.info('%s gettrt_result %d %s %f %d'
                     % (logid, sno, sent_u8.decode('utf8'),
                        classify_pos_ratio, judge_type))
    return (ret, judge_type, classify_pos_ratio,
            need_tag_pos_char_list, need_tag_pos_prob_list,
            need_tag_pos_index_list, need_tag_pos_word_list)


def bert_long_pred(sentence_fenci_unicode, logid='', sno=1, qtype=0, bid=''):
    tms = timer(sys._getframe().f_code.co_name, logid)
    # qtype:0 图文,  1短视频 2小视频
    fenci_list = sentence_fenci_unicode.split(" ")
    sentence_unicode = "".join(fenci_list)
    sentence = sentence_unicode.encode("utf-8", 'ignore')
    ret = -1
    retry_times = 0
    classify_pos_ratio = -1.0
    need_tag_pos_char_list = []
    need_tag_pos_prob_list = []
    need_tag_pos_index_list = []
    need_tag_pos_word_list = []
    judge_type = 0
    index_to_index_dict = {}  # 字节index到词的映射
    c_idx = 0
    for fenci in fenci_list:
        for _ in fenci:
            index_to_index_dict[c_idx] = fenci
            c_idx += 1
    sno = 0
    try:
        while ret < 0 and retry_times < 2:
            retry_times += 1
            ret, qos = l5sys.ApiGetRoute({'modId': 65319937, 'cmdId': 65536}, 0.2) # 获取路由
            if ret < 0:
                logging.error('logid:%s seqmodel_ApiGetRoute_failed,'
                              'retry_time:%d, ret:%d'
                              % (logid, retry_times, ret))
                ret = -1
                continue
            host_ip = qos['hostIp']
            host_port = qos['hostPort']
            logging.info("seqmodel_api, ip:%s port:%s logid:%s sno:%d"
                         % (host_ip, host_port, logid, sno))

            s_time = time.time()
            (ret, judge_type, classify_pos_ratio, need_tag_pos_char_list, need_tag_pos_prob_list, need_tag_pos_index_list, need_tag_pos_word_list) = get_trt_result(host_ip, host_port, sentence, sno, index_to_index_dict, logid)

            e_time = time.time()
            used_time = int((e_time - s_time) * 1000000)
            l5sys.ApiRouteResultUpdate(qos, ret, used_time) # 上报结果

            total_elapse = (e_time - s_time) * 1000
            logging.info('seqmodel_request_total ip:%s port:%s'
                         'cost:%s logid:%s sno:%d qtype:%d'
                         % (host_ip, host_port, total_elapse,
                            logid, sno, qtype))
            if total_elapse > 200:
                logging.warning('seqmodel_overtime ip:%s port:%s cost:%s'
                                'logid:%s sno:%d sent:%s'
                                % (host_ip, host_port, total_elapse,
                                   logid, sno, sentence_unicode))
            update_monitor_info(total_elapse, ret, log_id=logid,
                                module='gseqlabel',
                                bid=bid, request_ip=host_ip)
    except Exception as ex:
        ret = -1
        classify_pos_ratio = -1.0
        need_tag_pos_char_list = []
        need_tag_pos_prob_list = []
        need_tag_pos_index_list = []
        need_tag_pos_word_list = []
        judge_type = 0
        logging.error("seqmodel_exceptions:%s logid:%s" % (str(ex), logid))
    del tms
    return (classify_pos_ratio, need_tag_pos_char_list,
            need_tag_pos_index_list, need_tag_pos_prob_list,
            need_tag_pos_word_list, judge_type)


if __name__ == '__main__':
    SENTENCE = ('自赵丽颖与冯绍峰结婚后二人的一举一动便被谁关注,'
                '不过自从去年开始赵丽颖便一直深居简出,拍完《知否》'
                '后直接休长假,后来才人们才得知是为了养胎。')
    # print(SENTENCE)
    data = bert_long_pred(SENTENCE, logid=0, sno=int(sys.argv[1]), qtype=0)
