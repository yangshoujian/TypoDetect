# vim: set fileencoding=utf8
import grpc
import fire
from .common import Stub, Request


def main(addr, name, cls, *args, **kwargs):
    channel = grpc.insecure_channel(addr)
    stub = Stub(channel)
    cls = globals()[cls]
    method = getattr(stub, name)
    in_arg = cls(*args, **kwargs)
    return method(in_arg)


if __name__ == "__main__":
    fire.Fire(main)
