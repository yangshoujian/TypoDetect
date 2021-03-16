# vim: set fileencoding=utf8
import sys
import grpc
from concurrent import futures
from .common import add_to_server
import time
import os
import uwsgi
import uwsgidecorators
import flask
from inspect import signature
from google.protobuf.json_format import Parse, MessageToJson
import yaml
from io import open
import logging
import logging.config

logger = logging.getLogger(__name__)


def setup_logging(master):
    config = yaml.load(open("etc/logging.yml"), Loader=yaml.FullLoader)
    if not master:
        worker_id = uwsgi.worker_id()
        if 'handlers' in config:
            for val in config['handlers'].values():
                if 'filename' in val:
                    val['filename'] = val['filename'] + f'.worker-{worker_id}'
    else:
        for val in config['handlers'].values():
            if 'filename' in val:
                val['filename'] += '.master'
    logging.config.dictConfig(config)
    if master:
        logger.info("logging setup for master")
    else:
        logger.info(f"logging setup for worker:{uwsgi.worker_id()}")


setup_logging(True)


@uwsgidecorators.postfork
def initlog():
    setup_logging(False)


def init_service():
    from .application import service
    try:
        from .application import init as init_postfork
    except ImportError:
        def init_postfork():
            pass  # void callback
    return service, init_postfork


service, init_postfork = init_service()


def serve():
    app = flask.Flask(__name__)

    @app.route('/<stub_name>', methods=['POST'])
    def route(stub_name=None):
        stub = getattr(service, stub_name)
        req_class = next(iter(signature(stub).parameters.values())).annotation
        is_json = flask.request.content_type == 'application/json'
        data = flask.request.get_data()
        if is_json:
            req_obj = Parse(data, req_class(), ignore_unknown_fields=True)
        else:
            req_obj = req_class.FromString(data)
        rsp = stub(req_obj)
        if is_json:
            out = MessageToJson(rsp, indent=0,
                                including_default_value_fields=True,
                                preserving_proto_field_name=True)
            return flask.Response(out, mimetype="application/json")
        else:
            return flask.Response(rsp.SerializeToString(),
                                  mimetype="application/protobuf")

    @uwsgidecorators.postfork
    @uwsgidecorators.thread
    def grpc_server():
        worker_num = 1
        if 'GRPC_THREAD_NUM' in os.environ:
            worker_num = int(os.environ['GRPC_THREAD_NUM'])
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=worker_num))
        add_to_server(service, server)
        instance = os.environ['SUPERVISOR_PROCESS_NAME']
        worker_id = uwsgi.worker_id()
        sockname = f"unix:///dev/shm/{instance}/{worker_id}-grpc.sock"
        print("grpc listen on {}".format(sockname))
        server.add_insecure_port(sockname)
        server.start()
        try:
            _ONE_DAY_IN_SECONDS = 60 * 60 * 24
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        finally:
            server.stop(0)
    return app
