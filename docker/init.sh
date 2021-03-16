#!/bin/bash

source /lib.sh

wait_instance uwsgi
supervisorctl start nginx

# 当定义了环境变量RELOAD_CYCLE时每隔RELAOD_CYCLE秒reload一次
if [[ -v RELOAD_CYCLE ]]; then
    while sleep $RELOAD_CYCLE; do
        reload.sh
    done
fi
