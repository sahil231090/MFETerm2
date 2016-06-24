import numpy as np

rseed = lambda : np.random.randint(0, np.iinfo(np.int32).max)
np.random.seed(rseed())

"""
l = linear_congruential_generator(9, 5,  16, 0)
for i in l.sample(16):
   print i

print l.nxt
"""
class linear_congruential_generator:
    def __init__(self, a, c, N, seed=rseed()):
        self.a = a
        self.c = c
        self.N = N
        self.seed = seed % N

    def set_seed(self, val):
        self.seed = val
    
    def sample(self, size=1):
        a = self.a
        c = self.c
        N = self.N
        I = self.seed
        for i in xrange(size):
            yield I
            I = ( a*I + c ) % N 
            self.seed = I
    
    @property
    def nxt(self):
        return next(self.sample())

    
class box_muller_generator:
"""
b = box_muller_generator()
for i in b.sample(20):
   print i

print b.nxt
"""
    def __init__(self):
        self.cache = None

    def set_seed(self, val):
        np.random.seed(val)

    def _n_uni_dist(self, size):
        return np.random.uniform(size=size)
        
    def sample(self, size=1):
        for i in xrange(size):
            if (i %2 == 1) and (self.cache is not None) and (i==0):
                yield self.cache
            else:
                if (i %2 == 0):
                    uni = self._n_uni_dist(2*(size/2) + 2)
                    u1, u2 = uni[i], uni[i+1]
                    r, theta = np.sqrt(-2*np.log(u1)), 2*np.pi*u2
                    z1, z2 = r*np.cos(theta), r*np.sin(theta)
                    yield z1
                else:
                    yield z2

    @property
    def nxt(self):
        return next(self.sample())
                

class polar_rejection_method:
"""
p = polar_rejection_method()
for i in p.sample(20):
   print i

print p.nxt
"""
    def __init__(self):
        self.cache = None

    def set_seed(self, val):
        np.random.seed(val)
    
    def sample(self, size=1):
        for i in xrange(size):
            if (i %2 == 1) and (self.cache is not None) and (i==0):
                yield self.cache
            else:
                if (i %2 == 0):
                    while True:
                        uni = np.random.uniform(-1, 1, size=2)
                        u1, u2 = uni[0], uni[1]
                        r = np.sqrt(u1**2 + u2**2)
                        if r < 1:
                            break
                    cos = np.true_divide(u1, r)
                    sin = np.true_divide(u2, r)
                    r_ = np.sqrt(-4*np.log(r))
                    z1, z2 = r_*cos, r_*sin
                    yield z1
                else:
                    yield z2

    @property
    def nxt(self):
        return next(self.sample())
 
