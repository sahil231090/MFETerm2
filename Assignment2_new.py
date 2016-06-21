
from FixedIncomeLib import *
from __future__ import division
from IPython.display import Latex
import os 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 

base_path = os.path.dirname( os.getcwd() ) + os.sep
file_name = 'dat.xls'
base_path = base_path + file_name

df = pd.read_excel(base_path, sep="\t")
discount_list = []
for idx, row in df.iterrows():
    discount_list.append((row['Maturity'], row['Price']/100))

Z = UnitLocalLinearCurveFromArray(discount_list)

Ts = np.linspace(1/12,307/12, 103)
lZ = np.matrix([np.log(Z(t)) for t in Ts]).T
poly_T = np.matrix([[t, t**2, t**3, t**4, t**5] for t in Ts])

theta = np.dot(np.dot(np.linalg.inv(np.dot(poly_T.T, poly_T)), poly_T.T), lZ)
lZ_hat = poly_T * theta

a = theta[0,0]
b = theta[1,0]
c = theta[2,0]
d = theta[3,0]
e = theta[4,0]

a = -0.0326115568489289
b = -0.00107789195334756
c = -0.0000198829877521821
d = 0.0000028532567382765
e = -4.78160450945504E-08

print a,b,c,d,e
Z = lambda t: np.exp(a*t + b*(t**2) + c*(t**3)+ d*(t**4) + e*(t**5) )
"""
xs = np.linspace(1,30,30)
ys = [Z(x) for x in xs]
print zip(xs,ys)
plt.plot(xs, ys)
"""
print a*(30**1), b*(30**2), c*(30**3), d*(30**4), e*(30**5) 
print a*(30**1) + b*(30**2) + c*(30**3) + d*(30**4) + e*(30**5)
print 0.0339874358437*Z(0.5)/2 + (1+0.0339874358437/2)*Z(1), Z(1)

# Q1
r = SpotCurveFromDiscountCurve_k(Z, 2)
p_old = SemiCouponParYeildCurveFromDiscountCurve(Z)
p = SemiCouponParYeildCurveFromDiscountCurve(Z)
n = 30

for t in range(1,n+1):
    par = 100
    coupon = p(t)
    coupon_old = p_old(t)
    freq = 2
    mat = t
    
    cfs = BondParametersToCashFlows(par, mat, coupon, freq, True)
    Dmac = MacaulayDurationFromDiscountCurve(cfs, Z)
    Dmod = ModifiedDurationFromSpotCurve_k(cfs, r, 2)
    NPV = NPVFromDiscountCurve(cfs, Z)
    DV = DV01FromSpotCurve_k(cfs, r, 2)
    Convex = ConvexityFromSpotCurve_k(cfs, r, 2)
    print t, NPV, coupon, coupon_old, Dmac, Dmod, DV, Convex



# Q2
n = 30
for t in range(1,n+1):
    par = 100
    coupon_2 = 0.02
    coupon_12 = 0.12
    freq = 2
    mat = t
    
    cfs_2 = BondParametersToCashFlows(par, mat, coupon_2, freq, True)
    cfs_12 = BondParametersToCashFlows(par, mat, coupon_12, freq, True)
    
    Dmac_2 = MacaulayDurationFromDiscountCurve(cfs_2, Z)
    Dmac_12 = MacaulayDurationFromDiscountCurve(cfs_12, Z)
    
    print t, Dmac_2, Dmac_12

