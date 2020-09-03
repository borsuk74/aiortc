import asyncio
import time
from aiohttp_requests import requests
#import requests

'''This one is not working'''

def background(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        loop = asyncio.get_event_loop()
        if callable(f):
            return loop.run_in_executor(None, f, *args, **kwargs)
        else:
            raise TypeError('Task must be a callable')
    return wrapped


@background
async def send_and_forget(host_name, data, callback):
    response = await requests.post(SERVER_URL, data=predict_request)
    #prediction = await response.text()  # for testing
    time.sleep(1)
    #callback(str(time.time())+prediction)
    #callback(str(time.time()))
    print("foo() completed")


globalvar = ""
SERVER_URL = 'http://localhost:5000/api'
def callback(text):
    global globalvar
    globalvar = text
    print(globalvar)


predict_request = '{"instances": ['  ']}'
print("Hello")
predict_request = '{"instances": [' + str(1) + ']}'
send_and_forget(SERVER_URL, predict_request, callback)
predict_request = '{"instances": [' + str(2) + ']}'
send_and_forget(SERVER_URL, predict_request, callback)
predict_request = '{"instances": [' + str(3) + ']}'
send_and_forget(SERVER_URL, predict_request, callback)
print("I didn't wait for foo()")
input("Press Enter to continue...")