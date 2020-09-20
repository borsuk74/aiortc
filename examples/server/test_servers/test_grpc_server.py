import grpc

import primefactor_pb2
import primefactor_pb2_grpc
from concurrent import futures
import random
import time


class PrimeFactorService(primefactor_pb2_grpc.FactorsServicer):

    def PrimeFactors(self, request_iterator, context):
        start = time.time()
        print("Request in PrimeFactor")
        for req in request_iterator:
            factors = prime_factors(req.num)
            print(factors)
            for fac in factors:
                print(f"Before sending { fac } time is { time.time()}")
                yield primefactor_pb2.Response(result=fac)
        print("Server run for: ")
        print(time.time() - start)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    primefactor_pb2_grpc.add_FactorsServicer_to_server(PrimeFactorService(), server)
    server.add_insecure_port('[::]:50089')
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
        print('Server running on port: 50089')
        serve()
    except KeyboardInterrupt:
        pass
