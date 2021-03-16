# -*- coding: utf-8 -*-

import time
import logging
import json
import urllib
import urllib.request as urllib2
import l5sys
from ..utils.monitor_utils import update_monitor_info
# from ..utils.time_utils import timer

LM_API = "http://%s/lmserver"


def lm_request(docid, sentence, bid=""):
    res_json, ret_code = _perform(sentence, doc_id=docid, bid=bid)
    if ret_code < 0:
        ori_sent_ppl = 0.0
        lm_det_res = []
    else:
        lm_result = json.loads(res_json)
        lm_det_res = lm_result["detect_result"]
        ori_sent_ppl = lm_result["ori_sent_ppl"]
    return ori_sent_ppl, lm_det_res, ret_code


def _perform(sentence, doc_id, bid=""):
    ret = -1
    retry_times = 0
    res = None
    log_id = "%s_%s" % (bid, doc_id)
    while ret < 0 and retry_times < 2:
        retry_times += 1
        ret, qos = l5sys.ApiGetRoute({'modId': 64397633, 'cmdId': 5111808}, 0.2) # 获取路由
        if ret < 0:
            logging.warning('log_id:%s ApiGetRoute_failed,'
                            'ret:%d, retry_times:%d'
                            % (log_id, ret, retry_times))
            continue
        start_time = time.time()
        call_api = LM_API % (qos['hostIp'] + ":" + str(qos['hostPort']))
        ret = -1
        try:
            post_data = {"sent": sentence, "log_id": log_id}
            res = urllib2.urlopen(url=call_api, data=urllib.parse.urlencode(post_data).encode('utf8'), timeout=0.2).read()
            ret = 0
        except Exception as ex:
            ret = -1
            logging.warning("request_nnlm_exception logid:%s api:%s msg:%s"
                            % (log_id, call_api, str(ex)))
        end_time = time.time()
        use_time = int((end_time - start_time) * 1000000)
        l5sys.ApiRouteResultUpdate(qos, 0, use_time) # 上报结果
        elapsed_time = use_time / 1000
        logging.info('nnlm_detect logid:%s ret:%d api:%s cost:%f'
                     % (log_id, ret, call_api, elapsed_time))
        if elapsed_time > 100:
            logging.warning('nnlm_detect logid:%s ret:%d api:%s cost:%f'
                            % (log_id, ret, call_api, elapsed_time))
        update_monitor_info(elapsed_time, ret, log_id=log_id, module='gnnlm', bid=bid, request_ip=qos['hostIp'])
    return res, ret


if __name__ == '__main__':
    SENT = '让 你 的 小 孩 起 来 越 优 秀 育 儿 篇 ，'
    lm_request('cmsid123', SENT)
