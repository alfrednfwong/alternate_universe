import numpy as np
import time

def timer():
    def timer(fnc):
        def inner(arg):
            # inner function
            start = time.time()
            fnc(arg)
            end = time.time()
            elapsed = end - start
            return f"Time elapsed: {str(elapsed)} ms"

        return inner

    return timer