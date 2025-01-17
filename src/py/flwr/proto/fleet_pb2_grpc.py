# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from flwr.proto import fleet_pb2 as flwr_dot_proto_dot_fleet__pb2


class FleetStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetTasks = channel.unary_unary(
                '/flwr.server.fleet.proto.Fleet/GetTasks',
                request_serializer=flwr_dot_proto_dot_fleet__pb2.GetTasksRequest.SerializeToString,
                response_deserializer=flwr_dot_proto_dot_fleet__pb2.GetTasksResponse.FromString,
                )
        self.CreateResults = channel.unary_unary(
                '/flwr.server.fleet.proto.Fleet/CreateResults',
                request_serializer=flwr_dot_proto_dot_fleet__pb2.CreateResultsRequest.SerializeToString,
                response_deserializer=flwr_dot_proto_dot_fleet__pb2.CreateResultsResponse.FromString,
                )


class FleetServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetTasks(self, request, context):
        """Get tasks
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateResults(self, request, context):
        """Get results
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FleetServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetTasks': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTasks,
                    request_deserializer=flwr_dot_proto_dot_fleet__pb2.GetTasksRequest.FromString,
                    response_serializer=flwr_dot_proto_dot_fleet__pb2.GetTasksResponse.SerializeToString,
            ),
            'CreateResults': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateResults,
                    request_deserializer=flwr_dot_proto_dot_fleet__pb2.CreateResultsRequest.FromString,
                    response_serializer=flwr_dot_proto_dot_fleet__pb2.CreateResultsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'flwr.server.fleet.proto.Fleet', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Fleet(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetTasks(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flwr.server.fleet.proto.Fleet/GetTasks',
            flwr_dot_proto_dot_fleet__pb2.GetTasksRequest.SerializeToString,
            flwr_dot_proto_dot_fleet__pb2.GetTasksResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateResults(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flwr.server.fleet.proto.Fleet/CreateResults',
            flwr_dot_proto_dot_fleet__pb2.CreateResultsRequest.SerializeToString,
            flwr_dot_proto_dot_fleet__pb2.CreateResultsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
