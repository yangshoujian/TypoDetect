FROM docker.oa.com/nlpc/typodetect_data:0.2.0
COPY docker/reload.sh /usr/local/bin/
COPY docker/uwsgi.ini docker/supervisord.conf /etc/
COPY docker/nginx.conf /usr/local/openresty/nginx/conf/nginx.conf
COPY docker/entrypoint.sh /
COPY docker/lib.sh /
RUN ln -s /dev/shm/uwsgi_servers /usr/local/openresty/nginx/conf/uwsgi_servers
RUN rmdir /usr/local/openresty/nginx/logs/ && ln -s /app/log /usr/local/openresty/nginx/logs

#RUN mkdir -p /usr/local/services/
#COPY software/AttaAgent-1.7.330-install.tar.gz /usr/local/services/
#RUN tar zxvf /usr/local/services/AttaAgent-1.7.330-install.tar.gz  -C /usr/local/services/
#RUN sh /usr/local/services/AttaAgent-1.7.330-install/install.sh
#RUN /usr/local/services/AttaAgent-1.7/admin/restart.sh all


COPY setup.py /tmp
ENV PACKAGE_NAME=typodetect
COPY build/*.whl /build/
#RUN /usr/local/bin/pip3 install -e /tmp
RUN bash -c '/usr/local/bin/pip3 install -U /build/*.whl'
#RUN /usr/local/services/AttaAgent-1.7/admin/restart.sh all
COPY docker/init.sh /
RUN mkdir -p /app /app/log
RUN rm -f /etc/ld.so.preload
COPY etc /app/etc
COPY packages /app/packages
WORKDIR /app
ENV PROCESS_NUM=1 \
    INTERNAL_PROCESS_NUM=1 \
    THREAD_NUM=1 \
    GRPC_THREAD_NUM=1 \
    HTTP_PORT=0 \
    GRPC_PORT=0 \
    HTTP_HOST=0.0.0.0 \
    GRPC_HOST=0.0.0.0 \
    STARTUP_TIMEOUT=1200
CMD ["/entrypoint.sh"]
