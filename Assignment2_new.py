from FixedIncomeLib import *
from __future__ import division
from IPython.display import Latex
import os 
import pandas as pd
import matplotlib.pyplot as plt

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

theta = np.dot(np.dot(np.linalg.inv(np.dot(poly_T.T, poly_T)), poly_T.T), lZ)
lZ_hat = poly_T * theta

a = theta[0,0]
b = theta[1,0]
c = theta[2,0]
d = theta[3,0]
e = theta[4,0]

print a,b,c,d,e
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
    Dmac = MacaulayDurationFromSpotCurve_k(cfs, r, 2)
    Dmod = ModifiedDurationFromSpotCurve_k(cfs, r, 2)
    NPV = NPVFromDiscountCurve(cfs, Z)
    DV = DV01FromSpotCurve_k(cfs, r, 2)
    Convex = ConvexityFromSpotCurve_k(cfs, r, 2)
    
    print """\\multicolumn{{1}}{{|l|}}{{{0}}}               & {1}                & {2}                & {3}    & {4}        \\\\ \\hline""".format(t, Dmac, Dmod, DV, Convex)


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
    
    npv = NPVFromDiscountCurve(cfs_2, Z)
    ytm = IRRFromPrice(cfs_2, npv)
    ytm2 = 2*(np.power(1+ytm,0.5)-1)
    
    npv = NPVFromDiscountCurve(cfs_12, Z)
    ytm = IRRFromPrice(cfs_12, npv)
    ytm12 = 2*(np.power(1+ytm,0.5)-1)
    
    Dmac_2 = MacaulayDurationFromSpotCurve_k(cfs_2, r, 2)
    Dmac_12 = MacaulayDurationFromSpotCurve_k(cfs_12, r, 2)
    
    print """{0}                 & {1}                      & {2}                       \\\\ \\hline""".format(t, Dmac_2, Dmac_12, ytm2, ytm12)
    


# Q3
t_1 = 5
t_2 = 10
t_3 = 15
t_4 = 20

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

par = 100
coupon = p(t_4)
freq = 2
mat = t_4
cf4 = BondParametersToCashFlows(par, mat, coupon, freq, True)
P4 = NPVFromDiscountCurve(cf4, Z)


D1 = DV01FromSpotCurve_k(cf1, r, 2)
D2 = DV01FromSpotCurve_k(cf2, r, 2)
D3 = DV01FromSpotCurve_k(cf3, r, 2)
D4 = DV01FromSpotCurve_k(cf4, r, 2)

print ("Durations:\n 5Y Bond: {0:.4f} \n 10Y Bond: {1:.4f} \n 15Y Bond: {2:.4f} \n ".format(D1, D2, D3))
print ("We need to short {0:.4f} 5-year bonds in order to hedge a 10 year bond".format(D2*P2/(D1*P1)))
print ("We need to short {0:.4f} 15-year bonds in order to hedge a 10 year bond".format(D2*P3/(D3*P1)))


# Q4
x = ( D2 - D3 ) / ( D1 - D3 )
print x, 1-x
print D2, x*D1 + D3*(1-x)

# Q5
C1 = ConvexityFromSpotCurve_k(cf1, r, 2)
C2 = ConvexityFromSpotCurve_k(cf2, r, 2)
C3 = ConvexityFromSpotCurve_k(cf3, r, 2)
C4 = ConvexityFromSpotCurve_k(cf4, r, 2)

Matrix_eq =  np.matrix([ P1, P3, P4, D1, D3, D4, C1, C3, C4 ]).reshape((3,3))
vector_eq = np.matrix([ P2, D2, C2 ]).reshape((3,1))

print Matrix_eq.I * vector_eq

# Q6
print Z(3)
print 500000*Z(3)

# Q7

t_1 = 1
t_2 = 3
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

DM1 = ModifiedDurationFromSpotCurve_k(cf1, r, 2)
DM2 = ModifiedDurationFromSpotCurve_k(cf2, r, 2)
DM3 = ModifiedDurationFromSpotCurve_k(cf3, r, 2)

Dmac1 = MacaulayDurationFromSpotCurve_k(cf1, r, 2)
Dmac2 = MacaulayDurationFromSpotCurve_k(cf2, r, 2)
Dmac3 = MacaulayDurationFromSpotCurve_k(cf3, r, 2)


C1 = ConvexityFromSpotCurve_k(cf1, r, 2)
C2 = ConvexityFromSpotCurve_k(cf2, r, 2)
C3 = ConvexityFromSpotCurve_k(cf3, r, 2)

Matrix_eq =  np.matrix([ P1, P3, D1, D3 ]).reshape((2,2))
vector_eq = np.matrix([ P2, D2 ]).reshape((2,1))

print("Number of units")
print(50000*(Matrix_eq.I * vector_eq))

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
    
    DC_p10 = Price - Price * Dmod * 0.001 + 0.5 * Price * Conv * 0.001 * 0.001
    DC_n10 = Price + Price * Dmod * 0.001 + 0.5 * Price * Conv * 0.001 * 0.001
    DC_p300 = Price - Price * Dmod * 0.03 + 0.5 * Price * Conv * 0.03 * 0.03
    DC_n300 = Price + Price * Dmod * 0.03 + 0.5 * Price * Conv * 0.03 * 0.03
    
    # Duration
    #print """{0}                 & {1}       & {2}         & {3}        & {4}          \\\\ \\hline""".format(t, Do_p10, Do_n10, Do_p300, Do_n300)
    # Duration + Convex
    #print """{0}                 & {1}       & {2}         & {3}        & {4}          \\\\ \\hline""".format(t, DC_p10, DC_n10, DC_p300, DC_n300)
    # Spot Rate jump
    #print """{0}                 & {1}       & {2}         & {3}        & {4}          \\\\ \\hline""".format(t, Price_p10, Price_n10, Price_p300, Price_n300)
    
    
