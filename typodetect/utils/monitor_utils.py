# -*- coding=utf-8 -*-

import socket # 监控SD
import logging
import sys
import os
import time
from .. import tegmonitor  # 监控SDK包
#logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO,stream=sys.stdout)

def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

TEGMONITOR_CLIENT = tegmonitor.reportClient()
IS_CLIENT_INIT = False
IP = socket.gethostbyname(socket.gethostname())
if IP == '127.0.0.1':
    IP = get_host_ip()

def init_monitor_client():
    global TEGMONITOR_CLIENT, IS_CLIENT_INIT
    retcode = -1
    try:
        retcode, retmsg = TEGMONITOR_CLIENT.Init() # 监控客户端初始化
        if retcode != 0:
            logging.warning("monitor_init_error, code: %d, msg: %s", retcode, retmsg)
            IS_CLIENT_INIT = False
            return
        IS_CLIENT_INIT = True
        logging.warning("monitor_init, code: %d, msg: %s pid:%d is_init:%d", retcode, retmsg, os.getpid(), IS_CLIENT_INIT)
    except Exception as e:
        logging.warning("monitor_init_fail code:%d msg: %s pid:%d", retcode, str(e), os.getpid())

def update_monitor_info(time_used, ret_code = 0, log_id = '', module = '', bid = '', request_ip = ''):
    global TEGMONITOR_CLIENT, IS_CLIENT_INIT
    if IS_CLIENT_INIT is False:
        init_monitor_client()
        logging.info('monitor_update_init pid:%d is_init:%d' %(os.getpid(), IS_CLIENT_INIT))
    tag_set = {"bid": bid, 'module': module, 'requestip' : request_ip}
    total_req = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 1) # 总请求量
    fail_cnt = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 0) #失败数
    delay = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.AVG, time_used) # 延时
    timeout_cnt = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 0) # 超时数
    gt1200 = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 0) # 超过1200ms的数量
    if ret_code < 0:
        fail_cnt = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 1)
    if time_used > 5000:
        timeout_cnt = tegmonitor.MetricItem(
                tegmonitor.CurveCalMethod.ADD, 1)
    if time_used > 1200:
        gt1200 = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 1)

    nn_cnt = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 0) # 调用NN模型次数
    nn_timeout_cnt = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 0) # 调用NN模型次数
    if module in ['nnlm', 'seqlabel']:
        nn_cnt = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 1)
        if time_used > 50:
            nn_timeout_cnt = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, 1)

    metricMap = {
        "total_req": total_req,
        "typo_delay": delay,
        "typo_fail_cnt": fail_cnt,
        "typo_timeout_cnt": timeout_cnt,
        "typo_delay_gt": gt1200,
        "nn_cnt": nn_cnt,
        "nn_timeout_cnt": nn_timeout_cnt
    }
    monitor_data = tegmonitor.SDKCurveReport(
        instanceMark=IP, tagSet=tag_set,
        appMark="392_2180_typodetect", metricVal=metricMap
    )
    ret = TEGMONITOR_CLIENT.AddCurveData(monitor_data)
    if ret < 0:
        logging.warning("monitor_upload_err logid:%s ret:%d", log_id, ret) 

#for i in range(10000):
#    update_monitor_info(400, 0, 'cmsid', module = 'sentence', bid = '', request_ip = '127.0.0.1')
#    time.sleep(3)
