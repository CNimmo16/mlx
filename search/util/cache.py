import redis

client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class VectorCache:
    def __init__(self):
        self.client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    def __float_list_to_string(self, l):
        return ' '.join([str(x) for x in l])

    def __string_to_float_list(self, s: str):
        return list(map(float, s.split()))

    def get(self, ref: str):
        return self.__string_to_float_list(self.client.get(ref))
        
    def set(self, ref: str, vector: list[float]):
        self.client.set(ref, self.__float_list_to_string(vector))

vectors = VectorCache()
