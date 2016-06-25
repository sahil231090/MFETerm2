import numpy as np
import scipy.stats as ss
import logging
from statsmodels.tsa.stattools import acf
from statsmodels.graphics import tsaplots

import matplotlib.pyplot as plt
%matplotlib inline 

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

rseed = np.random.randint(0, np.iinfo(np.int32).max)
logging.info("The random seed is {0}".format(rseed))
np.random.seed(rseed)

# Q1



# the uniform sequences
u6 = np.random.uniform(size=10e6)

u5 = u6[:10e5]
# or
# np.random.seed(rseed)
# u5 = np.random.uniform(size=10e5)

u4 = u5[:10e4]
# or
# np.random.seed(rseed)
# u4 = np.random.uniform(size=10e4)

u3 = u4[:10e3]
# or
# np.random.seed(rseed)
# u3 = np.random.uniform(size=10e3)


err_Eus = []
err_std_us = []
err_var_us = []
err_skw_us = []
err_kut_us = []
err_ex_kut_us = []
Ns = []
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 10))


## part b
for i in range(3,7):
#for i in range(5,23):
    np.random.seed(rseed)
    N = np.power(10,i)
    #N = np.power(2, i)
    
    u = np.random.uniform(size=N)
    Ns.append(N)
    
    print "{0} & {1} & {2} & {3} & {4} & {5} & {6} \\\\ \\hline".format(N, np.mean(u), np.std(u), np.var(u), ss.skew(u), ss.kurtosis(u, fisher=False), ss.kurtosis(u))
    
    # Mean - emperical 
    Eu = np.abs(np.mean(u)-0.5)
    std_u = np.abs(np.std(u) - np.sqrt(1.0/12.0))
    var_u = np.abs(np.var(u) - 1.0/12.0)
    skw_u = np.abs(ss.skew(u))
    kut_u = np.abs(ss.kurtosis(u, fisher=False) - 9.0/5.0)
    ex_kut_u = np.abs(ss.kurtosis(u) + 6.0/5.0)
   
    err_Eus.append(Eu)
    err_std_us.append(std_u)
    err_var_us.append(var_u)
    err_skw_us.append(skw_u)
    err_kut_us.append(kut_u)
    err_ex_kut_us.append(ex_kut_u)

axs[0][0].set_title('Error in Mean of sample')
axs[0][0].loglog(Ns, err_Eus)

axs[1][0].set_title('Error in Std of sample')
axs[1][0].loglog(Ns, err_std_us)

axs[0][1].set_title('Error in Var of sample')
axs[0][1].loglog(Ns, err_var_us)

axs[1][1].set_title('Error in Skew of sample')
axs[1][1].loglog(Ns, err_skw_us)

axs[0][2].set_title('Error in Kurt of sample')
axs[0][2].loglog(Ns, err_kut_us)

axs[1][2].set_title('Error in Excess Kurt of sample')
axs[1][2].loglog(Ns, err_ex_kut_us)



Ns = []
fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(15, 15))

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1]

## part c
for idx, i in enumerate(range(3,7)):
#for i in range(5,23):
    np.random.seed(rseed)
    N = np.power(10,i)
    #N = np.power(2, i)
    
    u = np.random.uniform(size=N)
    Ns.append(N)
    
    axs[idx][0].set_title('Histogram')
    b = axs[idx][0].hist(u, bins)

    ys = [b_/sum(b[0]) for b_ in b[0]]
    xs = [(b[1][jdx]+b[1][jdx+1])*0.5 for jdx in xrange(len(b[1])-1) ]
    
    yys = np.cumsum(ys)
    xxs = b[1][1:]
    ks = max([np.abs(y-xx) for y, xx in zip(yys, xxs)])
    
    axs[idx][2].set_title('Kolmogorov-Smirnov Stat:{0:.4f}'.format(ks))
    axs[idx][2].plot(xxs, yys)
    axs[idx][2].plot(xxs, xxs)
    
    axs[idx][1].set_title('Error from the mean')
    axs[idx][1].plot(xs, [10*y for y in ys])
    axs[idx][1].plot(xs, [1]*len(xs))
    
## part d

fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(15, 15))
Ns = []
for idx, i in enumerate(range(3,7)):
#for i in range(5,23):
    np.random.seed(rseed)
    N = np.power(10,i)
    #N = np.power(2, i)
    
    u = np.random.uniform(size=N)
    
    u2 = np.power(u, 2)
    Ns.append(N)
    acfs, confint, qstat, pvalues = acf(u, nlags=20, qstat=True, alpha=0.05)
    
    tsaplots.plot_acf(u, axs[idx][0], lags=20)
    tsaplots.plot_acf(u2, axs[idx][1], lags=20)
    axs[idx][0].set_title('Autocorrelation Series. Length 10e{0}'.format(i))
    axs[idx][1].set_title('Autocorrelation Square Series. Length 10e{0}'.format(i))


Ns = []
fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(15, 15))

## part e
for idx, i in enumerate(range(3,7)):
#for i in range(5,23):
    np.random.seed(rseed)
    N = np.power(10,i)
    #N = np.power(2, i)
    u = np.random.uniform(size=N)
    for k in [1,2,3]:
        axs[idx][k-1].set_title('Length 10e{0} Lag{1}'.format(i, k))
        axs[idx][k-1].scatter(u[:-k], u[k:])
    

