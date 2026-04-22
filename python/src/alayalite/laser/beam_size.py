from math import log10


def beam_size_gen(k):
    assert k >= 1
    e = max(0, int(log10(k)) - 1)
    index = 0
    bases = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
    while True:
        yield bases[index] * int(10**e)
        index += 1
        if index == len(bases):
            e += 1
            index = 0
