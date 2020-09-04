import grpc

import primefactor_pb2
import primefactor_pb2_grpc
from concurrent import futures
import random
import time


class PrimeFactorService(primefactor_pb2_grpc.FactorsServicer):

    def PrimeFactors(self, request_iterator, context):
        for req in request_iterator:
            factors = prime_factors(req.num)
            for fac in factors:
                yield primefactor_pb2.Response(result=fac)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    primefactor_pb2_grpc.add_FactorsServicer_to_server(PrimeFactorService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


if __name__ == '__main__':
    try:
        print('Server running on port: 50052')
        serve()
    except KeyboardInterrupt:
        pass
