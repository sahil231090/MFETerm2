from FixedIncomeLib import *
from __future__ import division
from IPython.display import Latex
import os 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 


# Construst the Discount and Spot Yield Cruve

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

theta = (poly_T.T * poly_T).I * poly_T.T * lZ
lZ_hat = poly_T * theta

a = theta[0,0]
b = theta[1,0]
c = theta[2,0]
d = theta[3,0]
e = theta[4,0]


Z = lambda t: np.exp(a*t + b*(t**2) + c*(t**3)+ d*(t**4) + e*(t**5) )


# Q1
r = SpotCurveFromDiscountCurve_k(Z, 2)
p = SemiCouponParYeildCurveFromDiscountCurve(Z)

n = 30

for t in range(1,n+1):
    par = 100
    coupon = p(t)
    freq = 2
    mat = t
    
    cfs = BondParametersToCashFlows(par, mat, coupon, freq, True)
    Dmac = MacaulayDurationFromDiscountCurve(cfs, Z)
    Dmod = ModifiedDurationFromSpotCurve_k(cfs, r, 2)
    DV = DV01FromSpotCurve_k(cfs, r, 2)
    Convex = ConvexityFromSpotCurve_k(cfs, r, 2)
    print Dmac, Dmod, DV, Convex


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


# Q3
t_1 = 10
t_2 = 5
t_3 = 15

par = 100
coupon = p(t_1)
freq = 2
mat = t_1
cf1 = BondParametersToCashFlows(par, mat, coupon, freq, True)
P1 = NPVFromDiscountCurve(cf1, Z)

par = 100
coupon = p(t_2)
freq = 2
mat = t_2
cf2 = BondParametersToCashFlows(par, mat, coupon, freq, True)
P2 = NPVFromDiscountCurve(cf2, Z)

par = 100
coupon = p(t_3)
freq = 2
mat = t_3
cf3 = BondParametersToCashFlows(par, mat, coupon, freq, True)
P3 = NPVFromDiscountCurve(cf3, Z)

D1 = DV01FromSpotCurve_k(cf1, r, 2)
D2 = DV01FromSpotCurve_k(cf2, r, 2)
D3 = DV01FromSpotCurve_k(cf3, r, 2)

print D1, D2, D3, P1, P2, P3
print D1*P2/(D2*P1)
print D1*P3/(D3*P1)

# Q4
x = ( D2 - D3 ) / ( D1 - D3 )
print x
print D2, x*D1 + D3*(1-x)

# Q5
C1 = ConvexityFromSpotCurve_k(cf1, r, 2)
C2 = ConvexityFromSpotCurve_k(cf2, r, 2)
C3 = ConvexityFromSpotCurve_k(cf3, r, 2)

det = D1*C3 - D3*C1
x = (C3*D2 - D3*C2)/det
y = (D1*C2 - C1*D2)/det

print D1*x + D3*y, D2
print C1*x + C3*y, C2

# Q6
print 5000000*Z(3)

# Q7

t_1 = 3
t_2 = 1
t_3 = 7

par = 100
coupon = 0
freq = 1
mat = t_1
cf1 = BondParametersToCashFlows(par, mat, coupon, freq, True)
P1 = NPVFromDiscountCurve(cf1, Z)

par = 100
coupon = 0
freq = 1
mat = t_2
cf2 = BondParametersToCashFlows(par, mat, coupon, freq, True)
P2 = NPVFromDiscountCurve(cf2, Z)

par = 100
coupon = 0
freq = 1
mat = t_3
cf3 = BondParametersToCashFlows(par, mat, coupon, freq, True)
P3 = NPVFromDiscountCurve(cf3, Z)

D1 = DV01FromSpotCurve_k(cf1, r, 2)
D2 = DV01FromSpotCurve_k(cf2, r, 2)
D3 = DV01FromSpotCurve_k(cf3, r, 2)

C1 = ConvexityFromSpotCurve_k(cf1, r, 2)
C2 = ConvexityFromSpotCurve_k(cf2, r, 2)
C3 = ConvexityFromSpotCurve_k(cf3, r, 2)

det = D1*C3 - D3*C1
x = (C3*D2 - D3*C2)/det
y = (D1*C2 - C1*D2)/det

print D1*x + D3*y, D2
print C1*x + C3*y, C2


# Q8

r_p10 = lambda t: r(t) + 0.001
r_n10 = lambda t: r(t) - 0.001
r_p300 = lambda t: r(t) + 0.03
r_n300 = lambda t: r(t) - 0.03


Z_p10 = DiscountCurveFromSpotCurve_k(r_p10, 2)
Z_n10 = DiscountCurveFromSpotCurve_k(r_n10, 2)
Z_p300 = DiscountCurveFromSpotCurve_k(r_p300, 2)
Z_n300 = DiscountCurveFromSpotCurve_k(r_n300, 2)

n = 30

for t in range(1,n+1):
    par = 100
    coupon = p(t)
    freq = 2
    mat = t
    
    cfs = BondParametersToCashFlows(par, mat, coupon, freq, True)
    Dmod = ModifiedDurationFromSpotCurve_k(cfs, r, 2)
    Conv = ConvexityFromSpotCurve_k(cfs, r, 2)
    Price = NPVFromDiscountCurve(cfs, Z)
    
    Price_p10 = NPVFromDiscountCurve(cfs, Z_p10)
    Price_n10 = NPVFromDiscountCurve(cfs, Z_n10)
    Price_p300 = NPVFromDiscountCurve(cfs, Z_p300)
    Price_n300 = NPVFromDiscountCurve(cfs, Z_n300)
    
    Do_p10 = Price - Price * Dmod * 0.001
    Do_n10 = Price + Price * Dmod * 0.001
    Do_p300 = Price - Price * Dmod * 0.03
    Do_n300 = Price + Price * Dmod * 0.03
    
    DC_p10 = Price - Price * Dmod * 0.001 + 0.5 * Conv * 0.001 * 0.001
    DC_n10 = Price + Price * Dmod * 0.001 + 0.5 * Conv * 0.001 * 0.001
    DC_p300 = Price - Price * Dmod * 0.03 + 0.5 * Conv * 0.03 * 0.03
    DC_n300 = Price + Price * Dmod * 0.03 + 0.5 * Conv * 0.03 * 0.03
    
    print Price_p10, Do_p10, DC_p10

