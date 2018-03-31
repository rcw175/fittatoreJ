from numpy import*
from scipy import optimize, misc
from matplotlib.pyplot import *
%matplotlib inline
import pandas as pd

def fittatoreJ(fu, x, y, dx, dy, init, labels, ran=False, doppio=False):
    '''
    labels = {'x':, 'y':, 'file':}
    fu(x, *par)
    '''
    if ran==False:
        a,b = 0, x.size
    else:
        a,b = ran
    par, dev = optimize.curve_fit(fu, x[a:b], y[a:b], sigma=dy[a:b], p0=init, absolute_sigma=True)
    par = par; dev = diag(dev)
    if doppio==True:
        dyi = sqrt(dy[a:b]**2 + misc.derivative(fu, x[a:b], args=par, dx=0.0001)**2*1**2)
        par, dev = optimize.curve_fit(fu, x[a:b], y[a:b], sigma=dyi, p0=init, absolute_sigma=True)
        par = par; dev = diag(dev)
    X2 = {'obs':sum((y[a:b]-fu(x[a:b],*par))**2/dy[a:b]**2), 'exp':x[a:b].size-par.size}
    N=0
    figure(N)
    errorbar(x[a:b],y[a:b],yerr=dy[a:b],xerr=dx[a:b], fmt=',', capsize=3)
    plot(linspace(min(x[a:b]), max(x[a:b]),10000), fu(linspace(min(x[a:b]), max(x[a:b]),10000), *par), ',')
    xlabel(labels['x'])
    ylabel(labels['y'])
    savefig('figs/'+labels['file']+'.pdf')
    savefig('figs/'+labels['file']+'.png')
    
    figure(N+100)
    plot((y[a:b]-fu(x[a:b],*par))/dy[a:b], '.')
    plot(array([0,b-a]), array([0,0]))
    ylabel('Residui')
    savefig('figs/'+labels['file']+'_res.pdf')
    savefig('figs/'+labels['file']+'_res.png')
    
    return (array([par,dev]).T, X2)

def mediapesata(x,dx):
    return(sum(x/dx**2)/sum(1/dx**2), sqrt(1/sum(1/dx**2)))
