import numpy as np
from numpy import rot90

org = np.array([0, 0], int)
PIT_LEN = 8
pit = np.array([PIT_LEN, 0], int)
pit_list = []

def set_pit_len(l):
    pit[0] = l

def clear():
    global pit_list
    pit_list = []

def rot0(ar):
    return ar
def rot9(ar):
    return (0, ar[0])
def rot18(ar):
    return (-ar[0], 0)
def rot27(ar):
    return (0, -ar[0])
def plot(n: int, to_direc: str):
    ''' to_direc in ('>', '^', '<', 'v')'''
    assert to_direc in ('>', '^', '<', 'v')
    op_tbl = {'>': rot0, '^': rot9, '<': rot18, 'v': rot27}
    op = op_tbl[to_direc]
    frm = np.array([0, 0])
    to = frm + pit
    diff = pit[0]
    def inc():
        frm[0] += diff
        to[0] += diff #to[0] + diff
    for i in range(n):
        if i in (3, 6, 9):
            inc()
        yield [org + op(frm), org + op(to)]
        inc()
        inc()

def set_org(x, y):
    org[0] = x
    org[1] = y

def fwd(y):
    org[1] += y