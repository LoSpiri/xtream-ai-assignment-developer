from logging import Logger
import time


def time_it(func, logger: Logger):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Time elapsed {func.__name__}: {elapsed_time} seconds")
        return result
    return wrapper
