# -*- coding: utf-8 -*-
import logging.config 

# 定义三种日志输出格式 开始

standard_format = '%(asctime)s %(levelname)s [PID:%(process)d][%(threadName)s:%(thread)d][%(filename)s:%(lineno)d]' \
                  ':  %(message)s'
simple_format = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s'
id_simple_format = '[%(levelname)s][%(asctime)s] %(message)s'

# 定义日志输出格式 结束
logfile_path_staff = '/usr/local/zhiyan/agent/log/python_sdk.log'

# log配置字典
# LOGGING_DIC第一层的所有的键不能改变
LOGGING_DIC = {
    'version': 1,  # 版本号
    'disable_existing_loggers': False,  #　固定写法
    'formatters': {
        'standard': {
            'format': standard_format
        },
        'simple': {
            'format': simple_format
        },
    },
    'filters': {},
    'handlers': {
        #打印到终端的日志
        'sh': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',  # 打印到屏幕
            'formatter': 'simple'
        },
        #打印到文件的日志,收集info及以上的日志
        'fh': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件
            'formatter': 'standard',
            'filename': logfile_path_staff,  # 日志文件
            'maxBytes': 10000000,  # 日志大小
            'backupCount': 5,  # 轮转文件的个数
            'encoding': 'utf-8',  # 日志文件的编码
        },
    },
    'loggers': {
        'monitor': {
            'handlers': ['fh'],  # 这里把上面定义的两个handler都加上，即log数据既写入文件又打印到屏幕
            'level': 'INFO',
            'propagate': False,  # 向上（更高level的logger）传递
        },
    },
}

def getLogger(logPath = '/usr/local/zhiyan/agent/log/python_sdk.log', logLevel = 1, logMaxBytes = 10000000, logBackupCount = 5):
    LOGGING_DIC['handlers']['fh']['filename'] = logPath
    LOGGING_DIC['handlers']['fh']['maxBytes'] = logMaxBytes
    LOGGING_DIC['handlers']['fh']['backupCount'] = logBackupCount
    if logLevel == 1:
       LOGGING_DIC['handlers']['fh']['level'] = 'DEBUG'
       LOGGING_DIC['loggers']['monitor']['level'] = 'DEBUG'
    elif logLevel == 2:
       LOGGING_DIC['handlers']['fh']['level'] = 'INFO'
       LOGGING_DIC['loggers']['monitor']['level'] = 'INFO'
    elif logLevel == 3:
       LOGGING_DIC['handlers']['fh']['level'] = 'WARNING'
       LOGGING_DIC['loggers']['monitor']['level'] = 'WARNING'
    elif logLevel == 4:
       LOGGING_DIC['handlers']['fh']['level'] = 'ERROR'
       LOGGING_DIC['loggers']['monitor']['level'] = 'ERROR'
    elif logLevel == 5:
       LOGGING_DIC['handlers']['fh']['level'] = 'CRITICAL'
       LOGGING_DIC['loggers']['monitor']['level'] = 'CRITICAL'

    logging.config.dictConfig(LOGGING_DIC)  # 导入上面定义的logging配置 通过字典方式去配置这个日志
    logger = logging.getLogger('monitor')  # 生成一个log实例  这里可以有参数 传给task_id
    return logger

