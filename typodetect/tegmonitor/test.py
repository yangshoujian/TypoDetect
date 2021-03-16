# -*- coding: utf-8 -*-

import tegmonitor
import time
import threading

def reportThread1(test_client):
    for i in range(1, 11):
        # 上报维度组合，类型dict
        tag_set = {'host': 'google.com', 'ip': '1.1.1.1'}

        # 指标item，支持的上报方式有：ADD, AVG, MAX, MIN, MIDDLE, NONE(覆盖)，下面的i为指标的值(支持浮点数)
        metric_item1 = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, i)
        metricMap = {}
        # 上报指标，下例指标为req_cnt
        metricMap.update(req_cnt = metric_item1)

        # 组上报数据
        test_data = tegmonitor.SDKCurveReport(instanceMark = '192.168.0.1', tagSet = tag_set, appMark = "test_app1", metricVal = metricMap)

        # 将数据放入队列
        ret = test_client.AddCurveData(test_data)
        print("name: test1, ret: %d" %(ret))

def reportThread2(test_client):
    for i in range(1, 11):
        tag_set = {'host': 'google.com', 'ip': '1.1.1.2'}

        metric_item1 = tegmonitor.MetricItem(tegmonitor.CurveCalMethod.ADD, i)
        metricMap = {}
        metricMap.update(req_cnt = metric_item1)

        test_data = tegmonitor.SDKCurveReport(instanceMark = '127.0.0.1', tagSet = tag_set, appMark = "test_app2", metricVal = metricMap)

        ret = test_client.AddCurveData(test_data)
        print("name: test2, ret: %d" %(ret))

        time.sleep(10)

if __name__ == "__main__":
    test_client = tegmonitor.reportClient()

    '''
    reportClient的Init函数参数
    :param confPath: 配置文件路径，默认：/usr/local/zhiyan/agent/etc/agent.yaml
    :param queueSize: 接收数据队列长度，默认：1000000
    :param sendGap: 发送间隔，默认：1000 ms
    :param isOpenLog: 是否开启日志，默认：True
    :param logPath: 日志文件路径，默认：/usr/local/zhiyan/agent/log/python_sdk.log
    :param logLevel: 日志等级：1为debug，2为info，3为warning，4为error，5为critical，默认：2
    :param logMaxBytes: 日志大小，默认：10000000 Bytes
    :param logBackupCount: 日志个数，默认：5
    '''
    retCode, retMsg = test_client.Init(isOpenLog = True, logLevel = 2)
    
    t1 = threading.Thread(target= reportThread1, name = 'test1', args = (test_client, ))
    t1.setDaemon(True)
    t1.start()

    t2 = threading.Thread(target= reportThread2, name = 'test2', args = (test_client, ))
    t2.setDaemon(True)
    t2.start()

    time.sleep(50)

    test_client.ReloadLog(1)

    time.sleep(60)
