#!/bin/bash

export HTTP_LISTEN=${HTTP_HOST}:${HTTP_PORT}
export GRPC_LISTEN=${GRPC_HOST}:${GRPC_PORT}
export LOG_TO_STDERR=${LOG_TO_STDERR:-false}

(
    # nginx setup
    filename=/usr/local/openresty/nginx/conf/nginx.conf
    if [[ HTTP_PORT -gt  0 ]]; then
cat >>$filename <<EOF
    server  {
        listen ${HTTP_LISTEN};
        rewrite ^/$ /typoserver break;
        location / {
            uwsgi_pass uwsgicluster;
            include uwsgi_params;
        }
    }
EOF
    fi

    if [[ GRPC_PORT -gt 0 ]]; then
cat >>$filename <<EOF
    server  {
        listen ${GRPC_LISTEN} http2;
        location / {
            grpc_pass grpccluster;
        }
    }
EOF
    fi
    echo "}" >> $filename
)

mkdir -p /dev/shm/uwsgi
mkdir -p /dev/shm/uwsgi2

(
cat<<EOF
    ini = /dev/shm/uwsgi.sock.ini
    module = ${PACKAGE_NAME}.server:serve()
    threads = ${THREAD_NUM}
    processes = $((PROCESS_NUM + INTERNAL_PROCESS_NUM))
EOF
for i in `seq 1 $PROCESS_NUM`; do
    echo "map-socket = $((i-1)):$i"
done
) >> /etc/uwsgi.ini # mapping of processes

source /lib.sh

setup_instance uwsgi

mkdir -p log # ensure log directory exists
mkdir -p log/nginx
mkdir -p log/uwsgi
mkdir -p log/debug
mkdir -p log/warn
mkdir -p log/rsyslog

nohup /init.sh &
exec supervisord
