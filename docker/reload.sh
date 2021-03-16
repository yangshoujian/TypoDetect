#!/bin/bash

source /lib.sh

(
    flock -e 200 # prevent multiple reload simutaneously
    old=`supervisorctl status | awk '{if ($2 == "RUNNING") print $1}' | grep uwsgi`
    if [[ "$old" == "uwsgi" ]]; then
        new="uwsgi2"
    else
        new="uwsgi"
    fi

    echo "setup instance $new"
    setup_instance $new
    echo "start $new"
    supervisorctl start "$new"
    while ! wait_instance "$new"; do
        supervisorctl stop "$new"
        rm -f /dev/shm/$new/*.sock
        supervisorctl start "$new"
    done
    sleep 5
    echo "reload nginx"
    openresty -s reload
    sleep 5
    echo "stop $old"
    supervisorctl stop "$old"
    echo "cleanup $old files"
    rm -f /dev/shm/$old/*.sock
) 200< /etc/supervisord.conf
