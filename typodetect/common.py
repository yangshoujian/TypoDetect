from .proto.typodetect_pb2_grpc import\
    add_TypodetectServicer_to_server as add_to_server
from .proto.typodetect_pb2_grpc import\
    TypodetectServicer as BaseServicer
from .proto.typodetect_pb2_grpc import\
    TypodetectStub as Stub
from .proto.typodetect_pb2 import Request, Response, Drequest, Dresponse,\
    SentenceRequest, SentenceResponse, DebugResponse
