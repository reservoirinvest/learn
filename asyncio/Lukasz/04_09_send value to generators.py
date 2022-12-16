def gen():
    counter = 0
    while counter < 10:
        new_value = (yield counter)
        if new_value is not None:
            counter = new_value
        else:
            counter += 1

g = gen()
print(g.gi_frame.f_locals)
next(g)
next(g)
print(g.gi_frame.f_locals)

# send is a special form of next
g.send(7)
print(g.gi_frame.f_locals)

next(g)
print(g.gi_frame.f_locals)

g.send(10)

g.gi_running # works in interpreter