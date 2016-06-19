from __future__ import division
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.optimize import newton, minimize
import numpy as np

"""
Util Functions
"""
integrate = lambda *args: quad(*args)[0]

"""
f: the function to apply
l: the list to fold
a: the accumulator, who is also the 'zero' on the first call
"""
def fold(f, l, a):
    return a if(len(l) == 0) else fold(f, l[1:], f(a, l[0]))

"""
A curve with a simple constant value
"""
ConstCurveFromConst = lambda val: lambda t: val

"""
Step functions: Use these functions to convert
discreet set of data points to a curve
"""


"""
The following function takes an array of tuples
each tuple contains a time, rate

The output is a uni step function of Spot Curve
and is only valid till the last maturity

For Example:
r = UnitStepCurveFromArray([(1,0.1), (2,0.12), (3,0.14)]))

r(0) == 0.1
r(0.5) == 0.1
r(0.99) == 0.1
r(1) == 0.1
r(1.01) == 0.12
r(1.5) == 0.12
r(2) == 0.12
r(2.01) == 0.14
"""
UnitStepCurveFromArray = lambda arr: lambda t: [a[1] for a in sorted(arr, key=lambda x:x[0]) if a[0]>=t ][0]


"""
The following function takes an array of tuples
each tuple contains a time, rate

The output is a uni local step function of Spot Curve
the value returned is closest time value is found

For Example:
r = UnitLocalStepCurveFromArray([(1,0.1), (2,0.12), (3,0.14)]))

r(0) == 0.1
r(0.5) == 0.1
r(1) == 0.1
r(1.49) == 0.1
r(1.5) == 0.1
r(1.51) == 0.12
r(2) == 0.12
r(2.49) == 0.12
r(2.50) == 0.12
r(2.51) == 0.14
r(3) == 0.14
r(3.5) == 0.14
r(999) == 0.14
"""
def UnitLocalStepCurveFromArray(arr):
    ts = [a[0] for a in arr]
    def _c(t):
        temp = [np.abs(t-t_) for t_ in ts]
        idx = temp.index(min(temp))
        return arr[idx][1]
    return _c

"""
Use Local Linear Interpolation to get curve objects
in between those that are specified in the array

array input should be of type: [(T,val), ..]
"""
def UnitLocalLinearCurveFromArray(arr):
    arr = sorted(arr, key=lambda x:x[0])
    def _c(t):
        l = [(idx, a[0], a[1]) for idx,a in enumerate(arr) if a[0]>t]
        if len(l) == 0:
            return arr[-1][1]
        else:
            j, tj, vj = l[0]
            ti, vi = arr[j-1] 
            return vi*(tj-t)/(tj-ti) + vj*(t-ti)/(tj-ti)
    return _c


"""
Curve Conversion Functions
"""

"""
Get the Discount Curve from the Spot curve.
The latex formula is 

Z = e^{-r_T T}

"""
DiscountCurveFromSpotCurve = lambda r: lambda T: np.exp(-r(T)*T)

"""
Get the Discount Curve from the Spot curve.

Modified for k compounding frequency

The latex formula is 

Z = (1 + \frac{r}{k})^{-kT}

"""
DiscountCurveFromSpotCurve_k = lambda r, k: lambda T: np.power(1+(r(T)/k),-k*T)



"""
Get the Spot Curve from the Discount curve.
The latex formula is 

r = -\frac{ln(Z_T)}{T}

"""
SpotCurveFromDiscountCurve = lambda Z: lambda T: -np.log(Z(T))/T

"""
Get the Spot Curve from the Discount curve.
Modified for k compounding frequency

The latex formula is 

r = k((1/Z_T)^{1/(kT)}-1)

"""
SpotCurveFromDiscountCurve_k = lambda Z, k: lambda T: k*(np.power((1/Z(T)), (1/(k*T))) - 1)



"""
This functions give the present value of a amount in the future.
The default amount is 1 which is the same as the Discount Curve

For Example:
r = UnitStepCurveFromArray([(1,0.10), (2,0.12), (3,0.14)])
PV = PVFromSpotCurve( r ) # r is a Spot Curve

print PV(0, 100)
print PV(0.5, 100)
print PV(0.99, 100)
print PV(1, 100)
print PV(1.01, 100)
print PV(1.5, 100)
print PV(2, 100)
print PV(2.01, 100)

Output:

100.0
95.1229424501
90.5742708024
90.4837418036
88.585677052
83.5270211411
78.6627861067
75.472638454
"""
PVFromSpotCurve = lambda r: lambda t, FV=1: FV * DiscountCurveFromSpotCurve(r)(t)

PVFromSpotCurve_k = lambda r, k: lambda t, FV=1: FV * DiscountCurveFromSpotCurve_k(r, k)(t)


"""
Extension of PVFromSpotCurve - Uses Discount Curve directly
"""
PVFromDiscountCurve = lambda Z: lambda t, FV=1: FV * Z(t)

"""
This functions give the future value of an amount in the future.
The default amount is 1 which is the inverse of the Discount Curve

For Example:
r = UnitStepCurveFromArray([(1,0.10), (2,0.12), (3,0.14)])
FV = FVFromSpotCurve( r ) # r is a Spot Curve

print FV(0, 100)
print FV(0.5, 100)
print FV(0.99, 100)
print FV(1, 100)
print FV(1.01, 100)
print FV(1.5, 100)
print FV(2, 100)
print FV(2.01, 100)

Output:

100.0
105.127109638
110.406629956
110.517091808
112.885065992
119.721736312
127.124915032
132.498349135
"""
FVFromSpotCurve = lambda r: lambda t, PV=1: PV / DiscountCurveFromSpotCurve(r)(t)
FVFromSpotCurve_k = lambda r, k: lambda t, PV=1: PV / DiscountCurveFromSpotCurve_k(r, k)(t)


"""
Get the Forward Curve From the Spot Curve.
The Latex Formula is 

F(t,T) = \frac{r_T T - r_t t}{T-t}
"""
ForwardCurveFromSpotCurve = lambda r: lambda t, T: (r(T)*T - r(t)*t)/(T-t)
ForwardCurveFromSpotCurve_k = lambda r, k: lambda t, T: k*(np.power(np.power(1+(r(T)/k),k*T)/np.power(1+(r(t)/k),k*t),1/(k*(T-t)))-1)

"""
Get the Instantaneous Forward Curve From the Spot Curve.
The Latex Formula is 

F(T) = \frac{partial r}{partial T}
"""
InstantaneousFrowardFromSpotCurve = lambda r: lambda T: derivative(r, T, dx=1e-6)


"""
NPV Functions
"""

"""
Get the NPV of an array of cashflows given a Discount Curve
"""
NPVFromDiscountCurve = lambda cfs, Z: fold(lambda p, cf: p+PVFromDiscountCurve(Z)(cf[0], cf[1]), cfs, 0)

"""
Extension of NPVFromDiscountCurve - Uses Spot Curve Directly
"""
NPVFromSpotCurve = lambda cfs, r: NPVFromDiscountCurve(cfs, DiscountCurveFromSpotCurve(r))

"""
Get the NPV of an array of cashflows given a yeild
"""
NPVFromYield = lambda cfs, y: fold(lambda p, cf: p+cf[1]/((1+y)**(cf[0])), cfs,0)

"""
Get the IRR of an array of cashflows given a price
Intput:
    Cashflows: an array of cashflows with each element as tuple
                    (time, cash amount)
    price: current price of cashflows

Usage:
par = 100
maturity = 2
coupon_rate = 0.03
frequency = 2

cfs =  BondParametersToCashFlows(par, maturity, coupon_rate, frequency)
print IRRFromPrice(cfs, 100)

Output:   
    0.030225

"""
IRRFromPrice = lambda cashflows, price: newton(lambda yeild: NPVFromYield(cashflows, yeild) - price, 0) 


"""
Bond Functions
"""

"""
Get an array of Cashflows from the bond paramters
Input:
    Par: The facev value of the Bond
    Maturity: The Maturity of Bond In Years
    Coupon Rate: Coupon Rate Sepcified
    Frequency: Number of times Bond Pays each year
                1 - Annual
                2 - SemiAnnual
    Time:Current Time
Usage:

par = 100
maturity = 2
coupon_rate = 0.03
frequency = 2

print BondParametersToCashFlows(par, maturity, coupon_rate, frequency)

Output:
    [(0.5, 1.5), (1.0, 1.5), (1.5, 1.5), (2, 101.49999999999999)]
"""
def BondParametersToCashFlows(par, maturity, coupon_rate, frequency, new_issue=False):
    time = maturity - (1/frequency)
    cash_flow_array = [(maturity, par*(1+coupon_rate/frequency))]
    while time > 0:
        cash_flow_array.insert(0, (time, par*coupon_rate/frequency))
        time -= 1/frequency
    if new_issue:
	t1, cf1 = cash_flow_array[0]
	cf1 = par*(np.power(1+coupon_rate/frequency, frequency*t1)-1+np.isclose(t1,maturity))
	cash_flow_array[0] = (t1,cf1)
    return cash_flow_array


"""
Get the Yeild of a Bond given the Bond parameters and the price
Input:
    Par: The facev value of the Bond
    Maturity: The Maturity of Bond In Years
    Coupon Rate: Coupon Rate Sepcified
    Frequency: Number of times Bond Pays each year
                1 - Annual
                2 - SemiAnnual
    Time: Current Time
    Price: Current price of the Bond
"""
def YieldFromBondParameters(par, maturity, coupon_rate, frequency, time, price):
    cashflows = BondParametersToCashFlows( par, maturity, coupon_rate, frequency, time)
    _yield = IRRFromPrice( cashflows, price)
    return _yield


"""
Par Yeild Functions
"""
"""
Get the Par Yeild Curve for Semi Coupons Bonds

def SemiCouponParYeildCurveFromDiscountCurve(Z):
    def _y(T):
        denom = 0
        numer = 2*(1-Z(T))
        while(T>0):
            denom += Z(T)
            T -= 0.5
        return numer / denom
    return _y
"""
def SemiCouponParYeildCurveFromDiscountCurve(Z):
    def _y(T):
        par, freq, mat = 1, 2, T
        def _f(c):
            cfs = BondParametersToCashFlows(par, mat, c, freq, True)
            npv = NPVFromDiscountCurve(cfs, Z)
            return np.power(npv - par, 2)
        return newton(_f, 0)
    return _y

def SemiCouponParYeildCurveFromSpotCurve(r):
    Z = DiscountCurveFromSpotCurve(r)
    return SemiCouponParYeildCurveFromDiscountCurve(Z)

def SemiCouponParYeildCurveFromSpotCurve_k(r, k):
    Z = DiscountCurveFromSpotCurve_k(r, k)
    return SemiCouponParYeildCurveFromDiscountCurve(Z)

#def ForwardSemiCouponParYeildCurveFromDiscountCurve_N(Z, N):
#    def _y(T):
#        if (T<N):
#            return 0
#        else:
#            denom = 0
#            numer = 2*(Z(N)-Z(T))
#            while((T-N)>0):
#                denom += Z(T)
#                T -= 0.5
#            return numer / denom
#    return _y
def ForwardSemiCouponParYeildCurveFromDiscountCurve_N(Z, N):
    def _y(T):
        if (T<N):
            return 0
        else:
            par, freq, mat = 1, 2, T
            def _f(c):
                cfs = BondParametersToCashFlows(par, mat, c, freq, True)
                cfs_adj = [(a0-N, a1) for a0, a1 in cfs if a0 > N]
                npv = NPVFromDiscountCurve(cfs_adj, Z)
                return np.power(npv - par, 2)
            return newton(_f, 0)
    return _y




"""
Spot Rate Curve Functions
"""


def SvenssonSpotCurve(beta_0, beta_1, beta_2, beta_3, tau_1, tau_2):
    def _r(T):
        _T1 = T/tau_1
        _T2 = T/tau_2
        _e1 = np.exp(-_T1)
        _e2 = np.exp(-_T2)
        return beta_0 + beta_1*(1-_e1)/_T1 + beta_2*((1-_e1)/_T1 - _e1) + beta_3*((1-_e2)/_T2 - _e2)
    return _r


def NelsonSiegelFiveSpotCurve(beta_0, beta_1, beta_2, tau_1, tau_2):
    def _r(T):
        _T1 = T/tau_1
        _T2 = T/tau_2
        _e1 = np.exp(-_T1)
        _e2 = np.exp(-_T2)
        return beta_0 + beta_1*(1-_e1)/_T1 + beta_2*((1-_e2)/_T2 - _e2)
    return _r
    

def NelsonSiegelSpotCurve(beta_0, beta_1, beta_2, tau_1):
    def _r(T):
        _T1 = T/tau_1
        _e1 = np.exp(-_T1)
        return beta_0 + beta_1*(1-_e1)/_T1 + beta_2*((1-_e1)/_T1 - _e1)
    return _r


"""
Duration and Convexity Functions
"""

def MacaulayDurationFromDiscountCurve(cfs, Z):
    P = NPVFromDiscountCurve(cfs, Z)
    PV = PVFromDiscountCurve(Z)
    numer = fold( lambda tot, cf: tot+cf[0]*PV(cf[0], cf[1]), cfs, 0)
    return numer/P
