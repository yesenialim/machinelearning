# from math import exp
# def future_discrete_value(x,r,n):
#     return x*(1+r)**n
#
# def present_discrete_value(x,r,n):
#     return x*(1+r)**-n
#
# def future_continuous_value(x,r,t):
#     return x*exp(r*t)
#
# def present_continuous_value(x, r, t):
#     return x*exp(-r*t)
#
#
# if __name__ == '__main__':
#     x=100
#     r=0.05
#     n=5
#
# print('the future discrete value %f' %future_discrete_value(x,r,n))
# print('the present discrete value %f' %present_discrete_value(x,r,n))
# print('the future discrete value %f' %future_continuous_value(x,r,n))
# print('the present discrete value %f' %present_continuous_value(x,r,n))

class zeroCouponBonds:
    def __init__(self,principal,interest,maturity):
        self.principal=principal
        self.interest=interest/100
        self.maturity=maturity
    def presentValue(self,x,n):
        return x/(1+self.interest)**n
    def calculate_price(self):
        return self.presentValue(self.principal,self.maturity)

if __name__ == '__main__':
    bond=zeroCouponBonds(1000,4,2)
    print("price of the bond in $ %.2f" %bond.calculate_price())

class couponBonds:
    def __init__(self,principal,rate,interest_rate,maturity):
        self.principal=principal
        self.rate=rate/100
        self.interest_rate=interest_rate/100
        self.maturity=maturity
    def presentValue(self,x,n):
        return x/(1+self.interest_rate)**n
    def calculatePrice(self):
        price=0
        #discount coupon price
        for t in range(1,self.maturity+1):
            price=price+self.presentValue(self.principal*self.rate,t)
        #discount principle amount
        price=price+ self.presentValue(self.principal,self.maturity)
        return price
if __name__== '__main__' :
    bond=couponBonds(1000,10,4,3)
    print("price of the bond in $ %.2f" % bond.calculatePrice())

##continuous
from math import exp
class couponBonds:
    def __init__(self,principal,rate,interest_rate,maturity):
        self.principal=principal
        self.rate=rate/100
        self.interest_rate=interest_rate/100
        self.maturity=maturity

    def presentValue(self, x, n):
        return x * exp(-self.interest_rate * n)
    def calculatePrice(self):
        price=0
        #discount coupon price
        for t in range(1,self.maturity+1):
            price=price+self.presentValue(self.principal*self.rate,t)
        #discount principle amount
        price=price+ self.presentValue(self.principal,self.maturity)
        return price
if __name__== '__main__' :
    bond=couponBonds(1000,10,4,3)
    print("price of the bond in $ %.2f" % bond.calculatePrice())
