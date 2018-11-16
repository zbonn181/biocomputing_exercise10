# I am putting all packages and reading in the data here so each problem
# can be specifically only the script for that given task
import pandas as pd
import numpy as np
import scipy
import scipy.integrate as spint
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from plotnine import *

data = pd.read_csv('data.txt',header=0,sep=',')

# Problem 1
ggplot(data, aes(x='x',y='y'))+geom_point()+theme_classic()

def ishump(p,obs):
    B0 = p[0]
    B1 = p[1]
    B2 = p[2]
    sigma = p[3]
    pred = B0+B1*data.x+B2*data.x**2
    nll = -1*norm(pred,sigma).logpdf(data.y).sum()
    return nll
def islinear(p,obs):
    B0 = p[0]
    B1 = p[1]
    sigma = p[2]
    pred = B0+B1*data.x
    nll = -1*norm(pred,sigma).logpdf(data.y).sum()
    return nll

HumpGuess = np.array([1,1,1,1])
LinearGuess = np.array([1,1,1])
FitHump = minimize(ishump,HumpGuess,method='Nelder-Mead',args='data')
FitLinear = minimize(islinear,LinearGuess,method='Nelder-Mead', args='data')
teststat = 2*(FitLinear.fun-FitHump.fun)
df = len(FitHump.x)-len(FitLinear.x)
p_value = 1-stats.chi2.cdf(teststat,df)

if p_value <= 0.05:
    print("Quadratic approximation is more accurate for this particular data")
else:
    print('Linear approximation is more accurate for this particular data')

# Problem 2

def LotVoltSim(y,t0,R1,R2,a11,a12,a21,a22):
    N1 = y[0]
    N2 = y[1]
    dN1dt = R1*(1-N1*a11-N2*a12)*N1
    dN2dt = R2*(1-N2*a22-N1*a21)*N2
    return [dN1dt,dN2dt]

# The criteria for the coexistence of two species in a Lotka-Volterra Model are:
# 1.  a12 < a11
# 2.  a21 < a22 

# First Model - (neither of the two coexistence criteria are satisfied)

times = range(0,100)
y0 = [1,5]
params = (0.6,0.6,0.1,0.5,0.5,0.1)
sim = spint.odeint(func=LotVoltSim,y0=y0,t=times,args=params)
simDF = pd.DataFrame({"t":times,"Species 1":sim[:,0],"Species 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Species 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Species 2"),color='red')+theme_classic()

# Second Model - (first criterion is not satisfied, but second is)
times = range(0,100)
y0 = [1,5]
params = (0.6,0.6,0.1,0.5,0.1,0.5)
sim = spint.odeint(func=LotVoltSim,y0=y0,t=times,args=params)
simDF = pd.DataFrame({"t":times,"Species 1":sim[:,0],"Species 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Species 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Species 2"),color='red')+theme_classic()

# Third Model - (second criterion is not satisfied, but first is)
times = range(0,100)
y0 = [1,5]
params = (0.6,0.6,0.5,0.1,0.5,0.1)
sim = spint.odeint(func=LotVoltSim,y0=y0,t=times,args=params)
simDF = pd.DataFrame({"t":times,"Species 1":sim[:,0],"Species 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Species 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Species 2"),color='red')+theme_classic()

# Fourth Model - (both coexistence criteria are satisfied)
times = range(0,100)
y0 = [1,5]
params = (0.6,0.6,0.5,0.1,0.1,0.5)
sim = spint.odeint(func=LotVoltSim,y0=y0,t=times,args=params)
simDF = pd.DataFrame({"t":times,"Species 1":sim[:,0],"Species 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Species 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Species 2"),color='red')+theme_classic()