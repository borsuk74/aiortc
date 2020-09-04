import asyncio
import time
#from aiohttp_requests import requests
import requests

'''This one is working, it demonstrates a concept of sending request asynchronously
and fire some callback when got response back. Works because mainThread contains a loop'''

def background(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        # works
        loop = asyncio.get_event_loop()
        # This version works as well
        #loop = asyncio.new_event_loop()
        #asyncio.set_event_loop(loop)
        if callable(f):
            return loop.run_in_executor(None, f, *args, **kwargs)
        else:
            raise TypeError('Task must be a callable')
    return wrapped


@background
def send_and_forget(host_name, data, callback):
    response = requests.post(host_name, data=data)
    response.raise_for_status()
    prediction = response.text  # for testing
    callback(str(time.time())+prediction)
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