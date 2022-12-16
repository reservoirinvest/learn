def gen():
    counter = 0
    while counter < 6:
        yield counter
        counter += 1


g = gen()
print(type(g))

print(g.gi_running)
print(g.gi_frame)
print(g.gi_frame.f_locals)
print(next(g))
print(g.gi_frame.f_locals)

# suspended execution till we run to the end
print(next(g))
print(next(g))
print(next(g))
print(next(g))
print(next(g))
print(g.gi_frame.f_locals)

print(next(g))

print(g.gi_running) # will run only on python interpreter!
