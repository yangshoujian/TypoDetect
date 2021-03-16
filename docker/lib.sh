
function setup_instance {
    instance="$1"
    rm -rf /dev/shm/$instance
    mkdir -p /dev/shm/$instance
    (
        echo '[uwsgi]'
        for i in `seq 1 $PROCESS_NUM`; do
            echo "socket = /dev/shm/$instance/${i}.sock"
        done
    ) > /dev/shm/uwsgi.sock.ini

    (
    if [[ $HTTP_PORT -gt 0 ]]; then
        echo "upstream uwsgicluster {"
        for i in `seq 1 $PROCESS_NUM`; do
            echo "server unix:///dev/shm/$instance/${i}.sock;"
        done
        echo "}"
    fi

    if [[ $GRPC_PORT -gt 0 ]]; then
        echo "upstream grpccluster {"
        for i in `seq 1 $PROCESS_NUM`; do
            echo "server unix:/dev/shm/$instance/${i}-grpc.sock;"
        done
        echo "}"
    fi

    if [[ $INTERNAL_PROCESS_NUM -gt 0 ]]; then
        echo "upstream grpcinternal {"
        for i in `seq 1 $INTERNAL_PROCESS_NUM`; do
            echo "server unix:/dev/shm/$instance/$((PROCESS_NUM + i))-grpc.sock;"
        done
        echo "}"

    cat <<EOF
    server  {
        listen unix:/dev/shm/${instance}-internal.sock http2;
        location / {
            grpc_pass grpcinternal;
        }
    }
EOF
    fi
    )> /dev/shm/uwsgi_servers

    echo $instance > /dev/shm/instance
}

function wait_instance {
    (
        instance=$1
        timeout=${STARTUP_TIMEOUT} # 启动的最长时间, 超过则重新尝试
        count=0
        while sleep 1; do
            if ((count > timeout)); then
                exit 1
            fi
            grpc_num=`(ls /dev/shm/$instance/*.sock | grep grpc | wc -l) 2>/dev/null`
            http_num=`(ls /dev/shm/$instance/*.sock | grep -v grpc | wc -l) 2>/dev/null`
            if ((grpc_num == PROCESS_NUM+INTERNAL_PROCESS_NUM)) && ((http_num == PROCESS_NUM)); then
                echo
                exit 0
            else
                echo -n .
                ((count++))
            fi
        done
    )
}
