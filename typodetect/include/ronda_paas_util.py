# coding=utf-8
from enum import Enum, auto
from attaapi import Atta
from trpc_report_api_python import report
from trpc_report_api_python import server_info
from trpc_report_api_python import pcgmonitor
from trpc_report_api_python.pb import nmnt_pb2
import asyncio
from datetime import datetime
import re
from . import __version__
SUFFIX_OF_PAAS_MONITOR = "paas_stat"
SUFFIX_OF_PAAS_VERSION_MONITOR = "paas_sdk_ver"
LEN_OF_FEATURE_LIST = 9
VTYPE = dict(
    video=0,
    audio=1,
    text=2,
    image=3,
    vector=4,
    other=5,
)
# 模型服务请求对象信息
class PaaSRequestObj(object):
    def __init__(self, model_id='', version=''):
        super(PaaSRequestObj, self).__init__()
        self.model_id = model_id  # 被调服务ID
        self.version = version  # 被调模型版本
# 模型服务的鉴权信息
class PaaSAuthorizeInfo(object):
    def __init__(self, user_id='', skey=''):
        super(PaaSAuthorizeInfo, self).__init__()
        self.user_id = user_id  # 调用方用户标记
        self.skey = skey  # 鉴权的skey信息
# 模型服务的调用信息
class PaasRequestCommonInfo(object):
    def __init__(self, obj=None, auth=None):
        super(PaasRequestCommonInfo, self).__init__()
        self.obj = obj
        self.auth = auth
class UidType(Enum):
    MODULE = 0  # 模块
    PRODUCT = auto()  # 运营产品
class ReportMode(Enum):
    ASYNC = 0  # 异步/非阻塞
    SYNC = auto()  # 同步/阻塞
class FrameWork(Enum):
    TRPC = auto()  # trpc 框架
    NONE = auto()  # 无框架
class RondaPaasResultCode(Enum):
    RONDA_PAAS_RC_SUCC = 0  # 操作成功
    RONDA_PAAS_RC_AUTH_DENIED = auto()  # 没有操作权限
    RONDA_PAAS_RC_USER_ERROR = auto()  # 用户错误
    RONDA_PAAS_RC_INTERNAL_ERROR = auto()  # 服务内部错误
    RONDA_PAAS_RC_UNKNOWN = auto()  # 未知错误 (为了向下兼容，不建议使用)
    RONDA_PAAS_RC_FAIL = auto()  # 操作失败 (为了向下兼容，不建议使用)
    RONDA_PAAS_RC_ERROR = auto()  # 操作异常，无法判定操作结果（为了向下兼容，不建议使用
class RondaPaasResult(object):
    def __init__(self, ret_code=RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value,
                 message=RondaPaasResultCode.RONDA_PAAS_RC_SUCC.name):
        super(RondaPaasResult, self).__init__()
        self.ret_code = ret_code
        self.message = message
class RondaPaasUtil(object):
    __router_name = 'unset'
    __monitor_name = SUFFIX_OF_PAAS_MONITOR
    __report_mode = ReportMode.ASYNC.value
    __frame_work = FrameWork.NONE.value
    __server_info = server_info.ServerInfo()
    __atta = Atta()
    @classmethod
    def init(self, model_id: str, framework: int = FrameWork.NONE.value, app: str = None, server: str = None) \
            -> (int, str):
        """
        初始化，每次初始化会上报一次 sdk 版本信息
        :param model_id: 生产者router信息
        :param framework: 框架类型信息
        :param app: trpc app name
        :param server: trpc server name
        :return: ret_code, err
        """
        try:
            loop = asyncio.get_event_loop()
            self.__router_name = model_id
            if FrameWork.NONE.value == framework:
                self.__server_info.frame_code = "comm"
                self.__server_info.comm_name = SUFFIX_OF_PAAS_VERSION_MONITOR
                # init 007 for sdk
                pcgmonitor.init_api(self.__server_info)
                loop.run_until_complete(
                    self.__report_007([self.__router_name, __version__], [[float(1), 1, nmnt_pb2.SUM]]))
                # init 007 for service
                self.__server_info.comm_name = f"{self.__router_name}_{self.__monitor_name}"
                pcgmonitor.init_api(self.__server_info)
            elif FrameWork.TRPC.value == framework:
                if app is None or server is None:
                    return -1, f"app or server is None, pls check"
                self.__frame_work = framework
                self.__server_info.frame_code = "trpc"
                self.__server_info.app = app
                self.__server_info.server = server
                pcgmonitor.init_api(self.__server_info)
                loop.run_until_complete(self.__report_007(
                    [self.__router_name, __version__],
                    [[float(1), 1, nmnt_pb2.SUM]],
                    SUFFIX_OF_PAAS_VERSION_MONITOR
                ))
            return RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value, None
        except Exception as e:
            return -1, e
    @classmethod
    async def report(
            self, interface_name: str,
            paas_info: PaasRequestCommonInfo,
            result: RondaPaasResult,
            duration=0.0) \
            -> (int, str):
        """
        生产者调用上报接口
        :param interface_name: 接口名
        :param paas_info: PaaS 信息
        :param result: 消费者调用结果
        :param duration: 调用耗时
        :return: ret_code, err
        """
        if not isinstance(interface_name, str):
            return RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value, \
                   f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name} \
                   : the type of interface must be str"
        if not isinstance(paas_info, PaasRequestCommonInfo):
            return RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value, \
                   f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name} \
                   : the type of paas_info must be PaasRequstCommomInfo"
        if not isinstance(result, RondaPaasResult):
            return RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value, \
                   f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name} \
                   : the type of result must be RondaPaasResult"
        if not isinstance(duration, float):
            try:
                duration = float(duration)
            except ValueError as e:
                return RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value, \
                       f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name} \
                       : {e}, the type of duration must be float"
        auth_result = self.authorize(paas_info, UidType.MODULE.value)
        if auth_result.ret_code != RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value:
            return auth_result.ret_code, auth_result.message
        try:
            if FrameWork.NONE.value == self.__frame_work:
                await self.__report_007(
                    [
                        self.__router_name,
                        "0.0.0.1",
                        str(paas_info.auth.user_id),
                        str(paas_info.obj.model_id),
                        str(paas_info.obj.version),
                        str(interface_name),
                        result.message,
                        "",
                        "",
                        ""
                    ],
                    [
                        [float(1), 1, nmnt_pb2.SUM],
                        [duration, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.SUM],
                        [0, 1, nmnt_pb2.SUM],
                        [0, 1, nmnt_pb2.SUM],
                        [0, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.MAX],
                        [0, 1, nmnt_pb2.MAX],
                        [0, 1, nmnt_pb2.MAX],
                        [0, 1, nmnt_pb2.MIN],
                        [0, 1, nmnt_pb2.MIN],
                        [0, 1, nmnt_pb2.MIN],
                        [0, 1, nmnt_pb2.SET],
                        [0, 1, nmnt_pb2.SET],
                        [0, 1, nmnt_pb2.SET]
                    ]
                )
            elif FrameWork.TRPC.value == self.__frame_work:
                await self.__report_007(
                    [
                        str(paas_info.auth.user_id),
                        str(paas_info.obj.model_id),
                        str(paas_info.obj.version),
                        str(interface_name),
                        result.message,
                        "",
                        "",
                        ""
                    ],
                    [
                        [float(1), 1, nmnt_pb2.SUM],
                        [duration, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.SUM],
                        [0, 1, nmnt_pb2.SUM],
                        [0, 1, nmnt_pb2.SUM],
                        [0, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.AVG],
                        [0, 1, nmnt_pb2.MAX],
                        [0, 1, nmnt_pb2.MAX],
                        [0, 1, nmnt_pb2.MAX],
                        [0, 1, nmnt_pb2.MIN],
                        [0, 1, nmnt_pb2.MIN],
                        [0, 1, nmnt_pb2.MIN],
                        [0, 1, nmnt_pb2.SET],
                        [0, 1, nmnt_pb2.SET],
                        [0, 1, nmnt_pb2.SET]
                    ]
                )
            return RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value, None
        except Exception as e:
            return -1, e
    @classmethod
    async def report_feature(self, feature_req: list) -> (int, str):
        """
        特征上报接口
        :param feature_req: 特征数据list
        :return: ret_code, err
        """
        try:
            # check the type of feature_req
            if not isinstance(feature_req, list):
                return -1, "error feature request type, expected list"
            # check the length of feature list
            if len(feature_req) < LEN_OF_FEATURE_LIST:
                return -1, "error length of feature req list"
            # vid and vsrc should not be empty at the same time
            if feature_req[3] == "" and feature_req[4] == "":
                return -1, f"vid = {feature_req[3]}, vsrc={feature_req[4]} in feature list, should not be empty at the same time."
            # check vtype 
            if VTYPE.get(feature_req[5], None) is None:
                return -1, f"vtype = {feature_req[5]} in feature list, should use {list(VTYPE.keys())}"
            # substitute the time by SDK generated
            feature_req = feature_req[:9] + [datetime.now().strftime("%F %T")]
            result = self.authorize(
                PaasRequestCommonInfo(PaaSRequestObj(), PaaSAuthorizeInfo(user_id=feature_req[0])),
                UidType.MODULE.value
            )
            if result.ret_code != RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value:
                return result.ret_code, result.message
            feature_req = [str(i) for i in feature_req]
            await self.__atta.init_protocol("udp", "127.0.0.1")
            await self.__atta.send_fields("08400014925", "8775859689", feature_req, True)
            return RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value, None
        except Exception as e:
            return -1, e
    @classmethod
    async def report_sla(self, sla_req: list) -> (int, str):
        """
        SLA 上报接口
        :param sla_req: SLA 数据 list
        :return: ret_code, err
        """
        if not isinstance(sla_req, list):
            return -1, "error SLA request type, expected list"
        if len(sla_req) < 4 or len(sla_req) > 9:
            return -1, f"not enough length for sla report, expected in [4, 9] given length={len(sla_req)}"
        # 补齐长度
        sla_req = sla_req + [0] * (9 - len(sla_req))
        if re.search(r"(\d{4}-\d{1,2}-\d{1,2}$)", sla_req[0]) is None and \
                re.search(r"(\d{4}-\d{1,2}$)", sla_req[0]) is None:
            return -1, f"error dtime format, expected %Y-%m or %Y-%m-%d, given {sla_req[0]}"
        try:
            result = self.authorize(
                PaasRequestCommonInfo(PaaSRequestObj(), PaaSAuthorizeInfo(user_id=sla_req[2])),
                UidType.PRODUCT.value
            )
            if result.ret_code != RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value:
                return result.ret_code, result.message
            # 每个字段转换成str 类型
            sla_req = [str(i) for i in sla_req]
            await self.__atta.init_protocol("udp", "127.0.0.1")
            await self.__atta.send_fields("0f100014939", "7595963572", sla_req, True)
            return RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value, None
        except Exception as e:
            return -1, e
    @classmethod
    def authorize(self, paas_authorize_info: PaasRequestCommonInfo, uid_type: int = UidType.MODULE.value) \
            -> RondaPaasResult:
        """
        uid 范围鉴权
        :param paas_authorize_info: 用户信息
        :param uid_type: uid类型标识
        :return: result: RondaPaasResult()
        """
        result = RondaPaasResult()
        try:
            if str(paas_authorize_info.auth.user_id).isdigit():
                if uid_type == UidType.MODULE.value:
                    if not int(paas_authorize_info.auth.user_id) > 10000:
                        result.ret_code = RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value
                        result.message = f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name}:" \
                                         f"{UidType.MODULE.name} expected uid(your {paas_authorize_info.auth.user_id}) \
                            bigger than 10000"
                        return result
                elif uid_type == UidType.PRODUCT.value:
                    if not 0 < int(paas_authorize_info.auth.user_id) < 10000:
                        result.ret_code = RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value
                        result.message = f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name}:" \
                                         f"{UidType.PRODUCT.name} expected uid(your {paas_authorize_info.auth.user_id}) \
                            less than 10000"
                        return result
            else:
                result.ret_code = RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value
                result.message = f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name}:" \
                                 f"uid={paas_authorize_info.auth.user_id} must be digit"
                return result
        except Exception as e:
            result.ret_code = RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.value
            result.message = f"{RondaPaasResultCode.RONDA_PAAS_RC_AUTH_DENIED.name}:{e}"
        return result
    @classmethod
    async def __report_007(self, dimensions: list, values: list, custom_key: str = SUFFIX_OF_PAAS_MONITOR):
        """
        上报 PCG 007 监控平台
        :param dimensions: 维度
        :param values: 监控数据
        :param custom_key: trpc框架自定义上报
        :return:
        """
        try:
            loop = asyncio.get_event_loop()
            if FrameWork.NONE.value == self.__frame_work:
                report.comm_report(dimensions, values)
            elif FrameWork.TRPC.value == self.__frame_work:
                report.report_custom(custom_key, dimensions, values)
            return RondaPaasResultCode.RONDA_PAAS_RC_SUCC.value, None
        except Exception as e:
            return -1, e
    @classmethod
    def set_report_mode(self, report_mode=ReportMode.ASYNC.value):
        self.__report_mode = report_mode
