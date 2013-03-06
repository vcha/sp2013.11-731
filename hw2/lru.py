import collections
import functools

def lru_cache(maxsize=100):
    def decorating_function(user_function):
        cache = collections.OrderedDict()    # order: least recent to most recent

        @functools.wraps(user_function)
        def wrapper(*args):
            key = tuple(map(tuple, args)) # fix for lists
            try:
                result = cache.pop(key)
                wrapper.hits += 1
            except KeyError:
                result = user_function(*args)
                wrapper.misses += 1
                if len(cache) >= maxsize:
                    cache.popitem(0)    # purge least recently used cache entry
            cache[key] = result         # record recent use of this key
            return result
        wrapper.hits = wrapper.misses = 0
        return wrapper
    return decorating_function
