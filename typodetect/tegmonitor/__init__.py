# -*- coding: utf-8 -*-
import yaml
import socket
import sys 
import os
from . import curve_data_pb2
import struct
import threading
import queue
import time
import random
import logging
from enum import Enum
from . import myLogger

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

version = '3.0.7'

class ConcurrentQueue:
    def __init__(self, capacity = -1):
        self.__capacity = capacity
        self.__mutex = threading.Lock()
        self.__cond = threading.Condition(self.__mutex)
        self.__queue = queue.Queue()

    def get(self):
        '''
        if self.__cond.acquire():
            while self.__queue.empty():
                self.__cond.wait()
            elem = self.__queue.get()
            self.__cond.notify()
            self.__cond.release()
        return elem
        '''
        #不等待，直接返回空
        if self.__cond.acquire():
            if self.__queue.empty():
                self.__cond.notify()
                self.__cond.release()
                return -1
            elem = self.__queue.get()
            self.__cond.notify()
            self.__cond.release()
            return elem

    def put(self, elem):
        '''
        if self.__cond.acquire():
            while self.__queue.qsize() >= self.__capacity:
                self.__cond.wait()
            self.__queue.put(elem)
            self.__cond.notify()
            self.__cond.release()
        '''
        #不等待，直接返回队列满
        if self.__cond.acquire():
            if self.__queue.qsize() >= self.__capacity:
                self.__cond.notify()
                self.__cond.release()
                return -1001
            self.__queue.put(elem)
            self.__cond.notify()
            self.__cond.release()
            return 0

    def clear(self):
        if self.__cond.acquire():
            self.__queue.clear()
            self.__cond.release()
            self.__cond.notifyAll()

    def empty(self):
        isEmpty = false
        if self.__mutex.acquire():
            isEmpty = self.__queue.empty()
            self.__mutex.release()
        return isEmpty

    def size(self):
        size = 0
        if self.__mutex.acquire():
            size = self.__queue.size()
            self.__mutex.release()
        return size

    def resize(self, capacity = -1):
        self.__capacity = capacity
            
class CurveCalMethod(Enum):
    NONE = 0
    ADD = 1
    AVG = 2
    MAX = 3
    MIN = 4
    MIDDLE = 5

class MetricItem(object):
    def __init__(self, method, value):
        self.method = method
        self.value = value

class SDKCurveReport(object):
    def __init__(self, appMark, tagSet, metricVal, instanceMark):
        self.appMark = appMark                   # string
        self.tagSet = tagSet                     # dict, key: string, value: string
        self.metricVal = metricVal               # dict, key: string, value: MetricItem
        self.instanceMark = instanceMark          # string

class TegMonitor(object):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(TegMonitor, "_instance"):
            with TegMonitor._instance_lock:
                if not hasattr(TegMonitor, "_instance"):
                    TegMonitor._instance = object.__new__(cls)
        return TegMonitor._instance
        
    def __init__(self, socketPath, queueSize, sendGap, isOpenLog, logger):
        self.socketPath = socketPath
        self.queueSize = queueSize
        self.sendGap = sendGap
        self.isOpenLog = isOpenLog
        self.logger = logger

        self.lastReportTime = int(round(time.time() * 1000))
        self.msgQueue = ConcurrentQueue(queueSize)
        self.sdkSeq = random.randint(0, 1000)
        
        # 用于打印统计信息
        self.lastPrintStatTime = int(round(time.time() * 1000))        
        self.success_cnt = 0
        self.failed_cnt = 0

        # 用于存用户上报的数据，key为appMark#tagSet#instanceMark，value也为一个字典，key为metric#method，value为该上报组合的值
        self.dataCache = {}
        self.dataCacheAVG = {}
        self.dataCacheMID = {}

    def getInstance():
        return TegMonitor._instance

    def enqueueData(self, data):
        return self.msgQueue.put(data) 

    def SendAndRecv(self, serverAddr, pkg):
        startTime = int(round(time.time() * 1000))
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        #设置超时时间，1000ms
        sock.settimeout(1)

        try:
            sock.connect(serverAddr)
        except socket.error:
            sock.close()
            if self.isOpenLog == True:
                self.logger.warning("send pkg failed, connect socket error, seq: %d" %(pkg.seq))
            return -1, "connect socket error"

        leftTime = 1000 - (int(round(time.time() * 1000)) - startTime)

        if leftTime <= 0:
            sock.close()
            if self.isOpenLog == True:
                self.logger.warning("send pkg failed, send pkg timeout, seq: %d" %(pkg.seq))
            return -1, "send pkg timeout"

        pkgString = pkg.SerializeToString()

        pkgBytes = bytes(pkgString)

        length = int(len(pkgBytes))
        lenBytes = struct.pack(">i", length)

        reportData = lenBytes + pkgBytes

        # 发送数据
        sock.sendall(reportData)

        leftTime = 1000 - (int(round(time.time() * 1000)) - startTime)
        if leftTime <= 0:
            sock.close()
            self.logger.warning("send pkg failed, send pkg timeout, seq: %d" %(pkg.seq))
            return -1, "send pkg timeout"

        # 接收响应
        rspLenBytes = sock.recv(4)
        rspLen = struct.unpack(">i", rspLenBytes)[0]

        rspSeqBytes = sock.recv(8)
        rspSeq = struct.unpack(">Q", rspSeqBytes)[0]

        rspRetcodeBytes = sock.recv(4)
        rspRetcode = struct.unpack(">i", rspRetcodeBytes)[0]

        rspRetmsgBytes = sock.recv(rspLen - 12)
        rspRetmsg = bytes.decode(rspRetmsgBytes)

        sock.close()
        return rspRetcode, rspRetmsg

    def sendPkgToAgent(self, pkg):
        isSendSucc = False

        for i in range(1, 4):
            retCode, retMsg = self.SendAndRecv(self.socketPath, pkg)
            if retCode == 0:
                isSendSucc = True
                break
        if isSendSucc == True:
            return True
        else:
            return False
    
    def getSeq(self):
        self.sdkSeq = self.sdkSeq + 1
        if self.sdkSeq >= 1000000000:
            self.sdkSeq = 0
        return self.sdkSeq

    def sendAllToAgent(self):
        for key in self.dataCacheAVG.keys():
            for metricKey in self.dataCacheAVG[key].keys():
                valSum = 0.0
                for val in self.dataCacheAVG[key][metricKey]:     
                    valSum += val
                valTmp = float(valSum) / len(self.dataCacheAVG[key][metricKey])
               
                if key in self.dataCache:
                    if metricKey in self.dataCache[key]:
                        # 如果在dataCache中存在key+metricKey的组合，那么忽略（这种情况不可能出现）
                        pass
                    else:
                        self.dataCache[key][metricKey] = valTmp
                else:
                    tmp = {}
                    tmp[metricKey] = valTmp
                    self.dataCache[key] = tmp
        
        for key in self.dataCacheMID.keys():
            for metricKey in self.dataCacheMID[key].keys():
                sortData = sorted(self.dataCacheMID[key][metricKey])
                size = len(sortData)
                if size % 2 == 0: 
                    median = (sortData[size/2] + sortData[size/2-1]) / 2
                else: 
                    median = sortData[(size-1)/2]
                
                if key in self.dataCache:
                    if metricKey in self.dataCache[key]:
                        # 如果在dataCache中存在key+metricKey的组合，那么忽略（这种情况不可能出现）
                        pass
                    else:
                        self.dataCache[key][metricKey] = median
                else:
                    tmp = {}
                    tmp[metricKey] = median
                    self.dataCache[key] = tmp
        
        # 组包
        curvePkg = curve_data_pb2.SDKCurvePkg()
        dataCnt = 0
 

        for key in self.dataCache.keys():
            keyList = key.split('#')
            appMark = keyList[0]
            instanceMark = keyList[1]
            tagSet = eval(keyList[2])

            curveDataItem = curvePkg.curve_datas.add()

            curveDataItem.app_mark = keyList[0]
            curveDataItem.instance_mark = keyList[1]

            for k in tagSet.keys():
                curveDataItem.tag_set[k] = tagSet[k]

            for metricKey in self.dataCache[key].keys():
                metricKeyList = metricKey.split('#')
                metricName = metricKeyList[0]
                metricMethod = metricKeyList[1]
                if metricMethod == "0" or metricMethod == "CurveCalMethod.NONE":
                    curveDataItem.metric_val[metricName].method = curve_data_pb2.NONE
                elif metricMethod == "1" or metricMethod == "CurveCalMethod.ADD":
                    curveDataItem.metric_val[metricName].method = curve_data_pb2.ADD
                elif metricMethod == "2" or metricMethod == "CurveCalMethod.AVG":
                    curveDataItem.metric_val[metricName].method = curve_data_pb2.AVG
                elif metricMethod == "3" or metricMethod == "CurveCalMethod.MAX":
                    curveDataItem.metric_val[metricName].method = curve_data_pb2.MAX
                elif metricMethod == "4" or metricMethod == "CurveCalMethod.MIN":
                    curveDataItem.metric_val[metricName].method = curve_data_pb2.MIN
                elif metricMethod == "5" or metricMethod == "CurveCalMethod.MIDDLE":
                    curveDataItem.metric_val[metricName].method = curve_data_pb2.MIDDLE

                curveDataItem.metric_val[metricName].value = self.dataCache[key][metricKey]
                if self.isOpenLog == True:
                    self.logger.debug("Curvedata: appMark: %s, instanceMark: %s, tagSet: %s, metricName: %s, metricMethod: %s, value: %f" 
                                      %(appMark, instanceMark, keyList[2], metricName, metricMethod, self.dataCache[key][metricKey]))

            dataCnt = dataCnt + 1
            if dataCnt >= 1000:
                curvePkg.seq = self.getSeq()
                curvePkg.version = version
                if self.sendPkgToAgent(curvePkg) == True:
                    self.success_cnt += 1
                    if self.isOpenLog == True:
                        self.logger.info("send pkg to agent success, seq: %d" %(curvePkg.seq))
                else:
                    self.failed_cnt += 1
                    if self.isOpenLog == True:
                        self.logger.warning("send pkg to agent failed, seq: %d" %(curvePkg.seq))
                curvePkg.Clear()
                dataCnt = 0

        if dataCnt > 0:
            curvePkg.seq = self.getSeq()
            curvePkg.version = version
            if self.sendPkgToAgent(curvePkg) == True:
                self.success_cnt += 1
                if self.isOpenLog == True:
                    self.logger.info("send pkg to agent success, seq: %d" %(curvePkg.seq))
            else:
                self.failed_cnt += 1
                if self.isOpenLog == True:
                    self.logger.warning("send pkg to agent failed, seq: %d" %(curvePkg.seq))
        self.dataCache.clear()
        self.dataCacheAVG.clear()
        self.dataCacheMID.clear()

    def mergeCurveData(self, data):
        for metricName in data.metricVal.keys():
            key = ""
            key += data.appMark + "#"
            key += data.instanceMark + "#"
            key += str(data.tagSet)
           
            metricKey = metricName + "#" + str(data.metricVal[metricName].method)

            #增加数据校验，过滤掉不能转换成float的数据
            try:
                float(data.metricVal[metricName].value)
            except ValueError:
                if isOpenLog == True:
                    self.logger.warning("this report data's value can not change to float, we ignore it, appMark: %s, instanceMark: %s, tagSet: %s, metric: %s", data.appMark, data.instanceMark, str(data.tagSet), metricName)
                continue

            # 累加计算方式
            if data.metricVal[metricName].method == CurveCalMethod.ADD:
                if key in self.dataCache:
                    if metricKey in self.dataCache[key]:
                        self.dataCache[key][metricKey] += float(data.metricVal[metricName].value)
                    else:
                        self.dataCache[key][metricKey] = float(data.metricVal[metricName].value)
                else:
                    tmp = {}
                    tmp[metricKey] = float(data.metricVal[metricName].value)
                    self.dataCache[key] = tmp
            # 求平均计算方式
            elif data.metricVal[metricName].method == CurveCalMethod.AVG:
                if key in self.dataCacheAVG:
                    if metricKey in self.dataCacheAVG[key]:
                        self.dataCacheAVG[key][metricKey].append(float(data.metricVal[metricName].value))
                    else:
                        self.dataCacheAVG[key][metricKey] = []
                        self.dataCacheAVG[key][metricKey].append(float(data.metricVal[metricName].value))
                else:
                    tmp = {}
                    tmp[metricKey] = []
                    tmp[metricKey].append(float(data.metricVal[metricName].value))
                    self.dataCacheAVG[key] = tmp
            # 求最大计算方式
            elif data.metricVal[metricName].method == CurveCalMethod.MAX:
                if key in self.dataCache:
                    if metricKey in self.dataCache[key]:
                        if float(data.metricVal[metricName].value) > float(self.dataCache[key][metricKey]):
                            self.dataCache[key][metricKey] = float(data.metricVal[metricName].value)              
                        else:
                            pass
                    else:
                        self.dataCache[key][metricKey] = float(data.metricVal[metricName].value)
                else:
                    tmp = {}
                    tmp[metricKey] = float(data.metricVal[metricName].value)
                    self.dataCache[key] = tmp
            # 求最小计算方式
            elif data.metricVal[metricName].method == CurveCalMethod.MIN:
                if key in self.dataCache:
                    if metricKey in self.dataCache[key]:
                        if float(data.metricVal[metricName].value) < float(self.dataCache[key][metricKey]):
                            self.dataCache[key][metricKey] = float(data.metricVal[metricName].value)
                        else:
                            pass
                    else:
                        self.dataCache[key][metricKey] = float(data.metricVal[metricName].value)
                else:
                    tmp = {}
                    tmp[metricKey] = float(data.metricVal[metricName].value)
                    self.dataCache[key] = tmp
            # 求中位数计算方式
            elif data.metricVal[metricName].method == CurveCalMethod.MIDDLE:
                if key in self.dataCacheMID:
                    if metricKey in self.dataCacheMID[key]:
                        self.dataCacheMID[key][metricKey].append(float(data.metricVal[metricName].value))
                    else:
                        self.dataCacheMID[key][metricKey] = []
                        self.dataCacheMID[key][metricKey].append(float(data.metricVal[metricName].value))
                else:
                    tmp = {}
                    tmp[metricKey] = []
                    tmp[metricKey].append(float(data.metricVal[metricName].value))
                    self.dataCacheMID[key] = tmp
            # 其他计算方式
            else:
                if key in self.dataCache:
                    self.dataCache[key][metricKey] = float(data.metricVal[metricName].value)
                else:
                    tmp = {}
                    tmp[metricKey] = float(data.metricVal[metricName].value)
                    self.dataCache[key] = tmp

    def sendData(self):
        while True:
            nowTime = int(round(time.time() * 1000))
            # 是时候发送数据了
            if nowTime - self.lastReportTime >= self.sendGap:
                self.lastReportTime = nowTime
                self.sendAllToAgent()

            # 一分钟打印一次统计日志
            if nowTime - self.lastPrintStatTime >= 60000:
                self.lastPrintStatTime = nowTime
                if self.isOpenLog == True:
                    self.logger.info("send success: %d, send failed: %d" %(self.success_cnt, self.failed_cnt))
                success_cnt = 0
                failed_cnt = 0

            # 没到发送时间，从队列中获取数据进行汇聚
            data = self.msgQueue.get()
            if data == -1:
                time.sleep(0.005)
            else:
                self.mergeCurveData(data)     

    def start(self):
        # 启动发送数据线程
        t = threading.Thread(target = self.sendData, name = "send")
        t.setDaemon(True)
        t.start()

class reportClient(object):
    def Init(self, confPath = '/usr/local/zhiyan/agent/etc/agent.yaml', queueSize = 1000000, sendGap = 1000, 
             isOpenLog = True, logPath = '/usr/local/zhiyan/agent/log/python_sdk.log', logLevel = 2, 
             logMaxBytes = 10000000, logBackupCount = 5):
        try:
            f = open(confPath, 'r')
            confStr = f.read()
            cfg = yaml.load(confStr)
            sockPath = cfg['custom']['socket_path']
            
            self.logger = myLogger.getLogger(logPath = logPath, logLevel = logLevel, logMaxBytes = logMaxBytes, logBackupCount = logBackupCount)
            self.tegMonitor = TegMonitor(socketPath = sockPath, queueSize = queueSize, sendGap = sendGap, isOpenLog = isOpenLog, logger = self.logger)
            self.tegMonitor.start()
        except Exception as e:
            return -1, "init socket path error %s" %(str(e))

        return 0, "init ok"

    def ReloadLog(self, logLevel = 2):
        if logLevel == 1:
            level = logging.DEBUG
        elif logLevel == 2:
            level = logging.INFO
        elif logLevel == 3:
            level = logging.WARNING
        elif logLevel == 4:
            level = logging.ERROR
        elif logLevel == 5:
            level = logging.CRITICAL
        else:
            level = logging.INFO

        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        
    def AddCurveData(self, data):
        return self.tegMonitor.enqueueData(data)        
       
