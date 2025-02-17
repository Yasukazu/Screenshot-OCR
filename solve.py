class WSolve:
    def __init__(self, wt: int, n: int, pdr: float=0.3, er=0.8):
        self.wt = wt
        self.n = n
        self.w = int(wt / ((1 + pdr) * n + pdr))
        self.ew = int(self.w * er)
        self.eh = self.ew * 2
        self.pd = int(self.w * pdr)
        self.oo = (self.pd, self.pd)
        self.ht = self.eh + self.pd * 2
    
    @property
    def scale(self):
        return self.ew

    @property
    def box_size(self):
        return self.wt, self.ht

    @property
    def offsets(self):
        return (((i * (self.w + self.pd)) + self.pd, self.pd) for i in range(self.n))

    '''@property
    def pdr(self):
        return self.pdp / 100
    @property
    def pd(self):
        return self.w() * self.pdr
    @property
    def wb(self):
        return self.wt / self.n

    def w(self):
        return (self.wt + (self.n - 1) * self.pd) / self.n '''

if __name__ == '__main__':
    from pprint import pp
    ws = WSolve(40, 2, pdr=0.2)
    pp(ws)
