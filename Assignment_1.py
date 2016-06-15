from FixedIncomeLib import *
from __future__ import division
from IPython.display import Latex
import os 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Q1
PV = 20000
T = 8

r_1 = ConstCurveFromConst(0.06)
r_2 = ConstCurveFromConst(0.06)

FV_1 = FVFromSpotCurve_k(r_1, 1)
FV_2 = FVFromSpotCurve_k(r_2, 2)

print """\tFuture Value at annual compounding:\t\t{0:.2f} 
        Future Value at simi-annual compounding:\t{1:.2f}
        """.format(FV_1(T, PV), FV_2(T, PV))


# Q2
PV = 80000
PMT = 3000
T = 20
k = 12

r = ConstCurveFromConst(0.055)

FV = FVFromSpotCurve_k(r, k)

FV_val = fold(lambda val, t: val+ FV(t,PMT), np.linspace(0,T-1/k,T*k), FV(T,PV))

print """Total Value at the end: {0:.2f} """.format(FV_val)

# Q7

base_path = os.path.dirname( os.getcwd() ) + os.sep
file_name = 'dat.xls'
base_path = base_path + file_name

df = pd.read_excel(base_path, sep="\t")
discount_list = []
for idx, row in df.iterrows():
    discount_list.append((row['Maturity'], row['Price']/100))

Z = UnitLocalLinearCurveFromArray(discount_list)
r = SpotCurveFromDiscountCurve_k(Z,2)
f = ForwardCurveFromSpotCurve_k(r,2)
f_3m = lambda t: f(t,t+3/12)
p = SemiCouponParYeildCurveFromDiscountCurve(Z)
p_5 = ForwardSemiCouponParYeildCurveFromDiscountCurve_N(Z, 5)

n=103
spot = []
forward_3m = []
par_yeild = []
par_5y_for = []
time = []
for months in np.linspace(1,3*(n-1)+1, n):
    t = months / 12
    time.append(t)
    spot.append(r(t))
    forward_3m.append(f_3m(t))
    par_yeild.append(p(t))
    par_5y_for.append(p_5(t))

    
fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(18, 4))
axs[0].plot(time, spot, '-b')
axs[0].set_title('Spot Rate')

axs[1].plot(time, forward_3m, '-r')
axs[1].set_title('3Month Forward Rate')

axs[2].plot(time, par_yeild, '-g')
axs[2].set_title('Par Yeild')

axs[3].plot(time, par_5y_for, '-k')
axs[3].set_title('5 Year Forward Par Yeild')


# Q8
Ts = np.linspace(1/12,307/12, 103)
lZ = np.matrix([np.log(Z(t)) for t in Ts]).T
poly_T = np.matrix([[t, t**2, t**3, t**4, t**5] for t in Ts])

theta = (poly_T.T * poly_T).I * poly_T.T * lZ
lZ_hat = poly_T * theta

a = theta[0,0]
b = theta[1,0]
c = theta[2,0]
d = theta[3,0]
e = theta[4,0]

Z_hat = lambda t: np.exp(a*t + b*(t**2) + c*(t**3)+ d*(t**4) + e*(t**5) )
r_hat = SpotCurveFromDiscountCurve_k(Z_hat, 2)
p_hat = SemiCouponParYeildCurveFromDiscountCurve(Z_hat)
f_hat = ForwardCurveFromSpotCurve_k(r_hat, 2)
f_3m_hat = lambda t: f_hat(t,t+6/12)

n=60
discount = []
spot = []
forward_3m = []
par_yeild = []
par_5y_for = []
time = []
for months in np.linspace(1,n, n):
    t = months
    time.append(t)
    discount.append(Z_hat(t))
    spot.append(r_hat(t))
    forward_3m.append(f_3m_hat(t))
    par_yeild.append(p_hat(t))
    
fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(18, 4))
axs[0].plot(time[0:30], spot[0:30], '-b')
axs[0].set_title('Spot Rate')

axs[1].plot(time, forward_3m, '-r')
axs[1].set_title('3Month Forward Rate')

axs[2].plot(time[0:30], par_yeild[0:30], '-g')
axs[2].set_title('Par Yeild')

axs[3].plot(time[0:30], discount[0:30], '-k')
axs[3].set_title('Discount Curve')


# Q9
from scipy.optimize import minimize

n=103
def opt_(theta):
    r_hat = NelsonSiegelSpotCurve(beta_0 = theta[0],
                                  beta_1 = theta[1],
                                  beta_2 = theta[2],
                                  tau_1 = theta[3])
    Z_hat = DiscountCurveFromSpotCurve(r_hat)
    diff = [ (r(m/12),r_hat(m/12) ) for m in np.linspace(1,3*(n-1)+1, n)]
    return fold(lambda err, ps: err+10000*np.power(ps[0]-ps[1],2),diff,0)

theta_0 = [ 0.05108172, -0.01513864, -0.02367088,  2.0012328 ]

theta = minimize(opt_, theta_0, method='Powell')
print theta
theta = theta['x']
r_hat = NelsonSiegelSpotCurve(beta_0 = theta[0],
                                  beta_1 = theta[1],
                                  beta_2 = theta[2],
                                  tau_1 = theta[3])
Z_hat = DiscountCurveFromSpotCurve(r_hat)
r_hat = SpotCurveFromDiscountCurve_k(Z_hat, 2)
par_hat = SemiCouponParYeildCurveFromDiscountCurve(Z_hat)
f_hat = ForwardCurveFromSpotCurve_k(r_2, 2)



# Q10
n = 103
def opt_(theta):
    r_hat = SvenssonSpotCurve(beta_0 = theta[0],
                              beta_1 = theta[1],
                              beta_2 = theta[2],
                              beta_3 = theta[3],
                              tau_1 = theta[4],
                              tau_2 = theta[5])
    Z_hat = DiscountCurveFromSpotCurve(r_hat)
    diff = [ (r(m/12),r_hat(m/12) ) for m in np.linspace(1,3*(n-1)+1, n)]
    return fold(lambda err, ps: err+10000*np.power(ps[0]-ps[1],2),diff,0)

theta_0 = [1.31087807e-05,   2.78009393e-02,   1.09704823e-01,
         2.79885656e-02,   1.68638929e+01,   1.98315133e-01]
theta = minimize(opt_, theta_0, method='Powell')
print theta
theta = theta['x']

r_hat = SvenssonSpotCurve(beta_0 = theta[0],
                              beta_1 = theta[1],
                              beta_2 = theta[2],
                              beta_3 = theta[3],
                              tau_1 = theta[4],
                              tau_2 = theta[5])
Z_hat = DiscountCurveFromSpotCurve(r_hat)
r_hat = SpotCurveFromDiscountCurve_k(Z_hat, 2)
par_hat = SemiCouponParYeildCurveFromDiscountCurve(Z_hat)
f_hat = ForwardCurveFromSpotCurve_k(r_2, 2)

