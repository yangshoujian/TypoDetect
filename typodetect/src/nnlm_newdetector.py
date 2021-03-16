# -*- coding: utf-8 -*-

import time
import logging
import json
import urllib.request as urllib2
import l5sys
from ..utils.monitor_utils import update_monitor_info

LM_API = "http://%s/"


def lm_request(logid, sentence, bid=""):
    res_json, ret_code = _perform(sentence, log_id=logid, bid=bid)
    ori_sent_ppl = 0.0
    lm_det_res = []
    if ret_code >= 0:
        lm_result = json.loads(res_json)
        r_info = lm_result["r_info"]
        ori_sent_ppl = lm_result["ori_ppl"]
        for info in r_info:
            lm_det_res.append(
                [info['pos'], info['ori_zi'], info['ori_ppl'],
                 info['best_word'], info['best_ppl'],
                 info['sent_new_ppl'],
                 info['sent_ppl_ratio'], info['type']])
    return ori_sent_ppl, lm_det_res, ret_code


def _perform(sentence, log_id, bid=""):
    ret = -1
    retry_times = 0
    res = None
    while ret < 0 and retry_times < 2:
        retry_times += 1
        ret, qos = l5sys.ApiGetRoute(
            {'modId': 64397633, 'cmdId': 6750208}, 0.2)
        if ret < 0:
            logging.warning('log_id:%s ApiGetRoute_failed,'
                            'ret:%d, retry_times:%d'
                            % (log_id, ret, retry_times))
            continue
        start_time = time.time()
        # call_api = LM_API % (qos['hostIp']+":"+str(qos['hostPort']))
        call_api = "http://%s:%s/" % (qos['hostIp'], qos['hostPort'])
        ret = -1
        try:
            post_data = {"sentence": sentence, "docid": log_id}
            req = urllib2.Request(url=call_api,
                                  headers={'Content-Type': 'application/json'},
                                  data=json.dumps(post_data).encode('utf8'))
            res = urllib2.urlopen(req, timeout=0.2).read().decode('utf8')
            # res = urllib2.urlopen(url=call_api,
            #                      data=urllib.parse.urlencode(
            #                          post_data).encode('utf8'),
            #                      timeout=0.2).read()
            ret = 0
        except Exception as ex:
            ret = -1
            logging.warning("request_nnlm_exception logid:%s api:%s msg:%s"
                            % (log_id, call_api, str(ex)))
        end_time = time.time()
        use_time = int((end_time - start_time) * 1000000)
        l5sys.ApiRouteResultUpdate(qos, ret, use_time)
        elapsed_time = use_time / 1000
        logging.info('nnlm_detect logid:%s ret:%d api:%s cost:%f'
                     % (log_id, ret, call_api, elapsed_time))
        if elapsed_time > 200:
            logging.warning('nnlm_detect logid:%s ret:%d api:%s cost:%f'
                            % (log_id, ret, call_api, elapsed_time))
        update_monitor_info(elapsed_time, ret, log_id=log_id, module='gnnlm',
                            bid=bid, request_ip=qos['hostIp'])
    return res, ret


if __name__ == '__main__':
    SENT = '让 你 的 小 孩 起 来 越 优 秀 育 儿 篇 ，'
    lm_request('cmsid123', SENT)
