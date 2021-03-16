import os
import logging
import time
from io import open
import traceback
import json
from datetime import datetime
import configparser
import asyncio
import pickle
import yaml
import uwsgidecorators
from pytz import timezone, utc
import grpc
from .common import BaseServicer, Request, Response, Drequest, Dresponse, \
    SentenceRequest, SentenceResponse, Stub, DebugResponse
from .src.processor_manager import ProcessorManager
from .src.prepost_processor import PrePostProcessor
from .utils.monitor_utils import update_monitor_info
from .include.ronda_paas_util import RondaPaasUtil, RondaPaasResultCode
from .include.ronda_paas_util import PaasRequestCommonInfo, PaaSRequestObj
from .include.ronda_paas_util import PaaSAuthorizeInfo, RondaPaasResult


def custom_time(*args):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("Asia/Shanghai")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


logging.Formatter.converter = custom_time

logger = logging.getLogger(__name__)

logging.info("image_version: %s", os.environ['IMAGE'])
config = yaml.load(open('etc/application.yml'))
logger.info(f"load config : {config}")

PAAS_CFG = configparser.ConfigParser()
PAAS_CFG.read('etc/paas.ini')


@uwsgidecorators.postfork
def init_ronda_pass():
    ret, err = RondaPaasUtil.init(model_id="ChineseTypoDetector")
    if RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value != ret:
        logging.error('error:RondaPaasUtil_init %s' % (err))
    return ret, err


BID_LIST = ['kandian', 'cdp_check', 'kuaibao']


class Service(BaseServicer):
    def __init__(self):
        self.prepost_processor = PrePostProcessor() # 文章预处理类、对文章分段、分句，分词识别ner，统计专名、人称代词等统计信息
        self.processor_manager = ProcessorManager() # 对文章预处理、各个ProcessManager之间的流水线调用

    def proc_seq(self, request: SentenceRequest, context=None) -> SentenceResponse:
        flyweight = pickle.loads(request.body)
        ret = self.processor_manager.run_item(flyweight)
        response = SentenceResponse()
        response.body = pickle.dumps(ret)
        return response

    def debug(self, empty, context=None) -> DebugResponse:
        res = DebugResponse()
        res.json = json.dumps(dict(os.environ))
        return res

    def typoserver(self, request: Request, context=None):
        typo_circuit = False  # 错别字熔断标记
        is_fail_safe = False  # 降级标记
        logid = ''

        response = Response()
        response.doc_id = request.doc_id
        response.ret_code = 0
        paas_result = RondaPaasResult()
        try:
            bid = request.business_id.decode('utf-8') # 业务ID，如快报：kuaibao
            doc_id = request.doc_id.decode('utf-8') # 文章ID
            title = request.title.decode('utf-8') # 标题(UTF8编码)  required
            content = request.body_content.decode('utf-8') # 正文(UTF8编码)  required
            channel_id = request.channel_id # 一级分类ID
            logid = "%s_%s" % (bid, doc_id)
            # channel_name = request.channel_name.decode('utf8')
            data_type = request.data_type # 数据类型 0-图文 1-短视频 2-小视频
            start_time = time.time()
            if 'FAIL_SAFE' in os.environ:
                fail_safe = int(os.environ['FAIL_SAFE'])
                if fail_safe == 119:
                    is_fail_safe = True
            if 'CIRCUIT_BREAKER' in os.environ:
                breaker_conf = os.environ['CIRCUIT_BREAKER']
                if breaker_conf == "119001":  # 错别字熔断：当某服务出现不可用或响应超时的情况时，为了防止整个系统出现雪崩，暂时停止对该服务的调用。
                    typo_circuit = True

            paas_info, paas_result = self.get_pass_authorize(bid, doc_id) # 返回：模型服务的调用信息，模型服务的调用结果

            if typo_circuit:
                sent_results = [] # 服务熔断，返回空
            else:
                sent_infos = self.prepost_processor.pre_process(title, content, doc_id, channel_id, int(data_type), bid)  # 文章预处理类 list of FlyWeight
                instance = os.environ['SUPERVISOR_PROCESS_NAME']
                channel = grpc.insecure_channel(f'unix:///dev/shm/{instance}-internal.sock')
                stub = Stub(channel)
                reqs = [SentenceRequest(body=pickle.dumps(i)) for i in sent_infos] # pickle.dumps序列化对象，并将结果数据流写入到文件对象中。
                futures = [stub.proc_seq.future(i, wait_for_ready=True) for i in reqs] # 运行规则
                sent_results = [pickle.loads(f.result().body) for f in futures] # pickle.loads反序列化对象，将文件中的数据解析为一个Python对象。
            result_merge = []
            for item in sent_results:
                result_merge.extend(item)
            final_result = self.processor_manager.post_process(result_merge, doc_id, data_type)
            self.fill_result_to_response(final_result, response)
            logging.info('typosinfo logid:%s title_cnt:%d all_cnt:%d '
                         'level:%d pop:%d'
                         % (logid, final_result['title_num'],
                            final_result['all_num'],
                            final_result['typo_level'],
                            final_result['typo_pop_result']))
            paas_result.ret_code =\
                RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value # 0-操作成功
        except Exception as err:
            err_detail = traceback.format_exc()
            # RondaPaasResultCode.RONDA_PAAS_RC_FAIL
            paas_result.ret_code =\
                RondaPaasResultCode.RONDA_PAAS_RC_INTERNAL_ERROR.value
            response.ret_code = -1
            logging.error('app_process_exception:%s logid:%s detail:%s'
                          % (str(err), logid, err_detail))

        used_time = 1000 * (time.time() - start_time)
        if used_time > 1200:
            logging.warning('doc_overtime logid:%s cost:%d'
                            % (logid, used_time))
        logging.info('doc_process_end logid:%s doc_cost:%d' %
                     (logid, used_time))

        try:
            update_monitor_info(used_time, response.ret_code, doc_id, 'gtotaldoc', bid) # 更新监控信息
            asyncio.run(RondaPaasUtil.report('grpc', paas_info, paas_result)) # 模型服务的调用
        except Exception as err:
            traceback.print_exc()
            logging.error('updateinfo_exception:%s logid:%s'
                          % (str(err), logid))

        if is_fail_safe and response.ret_code < 0: # 降级标记
            response.ret_code = 0
            logging.warning("logid:%s is_fail_safe"
                            % (logid))

        if typo_circuit: # 错别字熔断标记
            response.ret_code = 0
            logging.warning("logid:%s typo_circuit"
                            % (logid))

        return response

    def fill_result_to_response(self, res, response):
        typo_result = response.typo_result # 错别字识别结果
        typo_result.title_num = res['title_num']
        typo_result.fst_num = res['fst_num']
        typo_result.all_num = res['all_num'] # 错别字个数
        typo_result.typo_level = res['typo_level'] # 文章粒度高召回或高准确
        typo_result.typo_pop_result = res['typo_pop_result'] # 文章粒度是否延迟分发
        for item in res['details']: # 错别字信息
            (sno, idx, original_sentence, wrong_ci, correct_ci, typ, ratio, ori_sent, term_imp, level, pop_result) = (item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10])
            tmp_typo_info = typo_result.details.add()
            tmp_typo_info.sno = sno # 句子索引 
            tmp_typo_info.idx = idx # 错别字在句子中的位置索引
            tmp_typo_info.sent = original_sentence.encode('utf-8') # 句子
            tmp_typo_info.wrong = wrong_ci.encode('utf-8') # 错字
            tmp_typo_info.correct = correct_ci.encode('utf-8') # 改正的字
            tmp_typo_info.typo = typ # 错别字类型
            tmp_typo_info.prob = ratio # 概率
            tmp_typo_info.ori_sent = ori_sent.encode('utf-8') # 长句
            tmp_typo_info.term_imp = term_imp # 级别
            tmp_typo_info.level = level # 置信度 1-轻度 2-重度
            tmp_typo_info.pop_result = pop_result # 文章是否pop延迟分发 0-否 1-是

    def get_pass_authorize(self, bid, doc_id):
        start_time = time.time()
        # set router
        # RondaPaasUtil.set_router('ChineseTypoDetector')
        if bid not in BID_LIST: # 业务类型
            paas_bid = 'other'
        else:
            paas_bid = bid
        user_id = PAAS_CFG.get(paas_bid, 'user_id') # 调用方用户标记
        skey = PAAS_CFG.get(paas_bid, 'skey') # 鉴权的skey信息
        model = PAAS_CFG.get(paas_bid, 'model') # 被调服务ID
        version = PAAS_CFG.get(paas_bid, 'version') # 被调模型版本
        paas_info = PaasRequestCommonInfo(PaaSRequestObj(model, version), PaaSAuthorizeInfo(user_id, skey)) # 模型服务的调用信息
        paas_result = RondaPaasResult() # 模型服务的调用结果
        auth_result = RondaPaasUtil.authorize(paas_info)
        authorize_flag = True
        if (auth_result.ret_code != RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value):
            authorize_flag = False
            logging.error("pass_authorize_fail logid:%s_%s msg:%s"
                          % (bid, doc_id, auth_result.message))
            paas_result.ret_code =\
                RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value
        used_time = 1000 * (time.time() - start_time)
        logging.info('request_authorize cost:%.4f ms logid:%s_%s is_auth:%d'
                     % (used_time, bid, doc_id, authorize_flag))

        return paas_info, paas_result # 返回：模型服务的调用信息，模型服务的调用结果

    def typoserver_debug(self, request: Drequest, context=None):

        response = Dresponse()
        response.doc_id = request.doc_id # 文章ID
        response.ret_code = 0 # 错误码：0-成功 非0-失败
        try:
            bid = request.business_id
            doc_id = request.doc_id
            title = request.title
            content = request.body_content
            channel_id = request.channel_id # 一级分类ID
            # channel_name = request.channel_name.decode('utf8')
            data_type = request.data_type

            sent_infos = self.prepost_processor.pre_process(title, content, doc_id, channel_id, int(data_type), bid) # 文章预处理类、对文章分段、分句，分词识别ner，统计专名、人称代词等统计信息
            final_result = self.processor_manager.run_debug_item(sent_infos[0]) # 错误信息
            response.typo_result = json.dumps(final_result) # 错别字识别结果
            logging.info('debug_info: %s' % (response.typo_result))
        except Exception as err:
            traceback.print_exc()
            response.ret_code = -1
            logging.error('debug_process_exception:%s logid:%s_%s'
                          % (str(err), bid, doc_id))

        return response # 返回：文章ID，错别字识别结果，错误码


service = Service()
