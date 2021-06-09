def gen():
    counter = 0
    while counter < 6:
        yield counter
        counter += 1

def outer():
    yield -1
    yield from gen()
    yield 10

for result in outer():
    print(result)
