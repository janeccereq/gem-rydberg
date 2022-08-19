# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:20:37 2020

@author: mateu
"""


import numpy as np


def supergauss(X, cen=0, w=0.5, pow=2, dtype=np.cdouble):
    return np.exp(-((X - cen)/w)**pow, dtype=dtype)

def square(X):
    return ((-1)**np.floor(2*X) + 1)/2

def saw(X):
    return (2*(X-np.floor(X+1/2))+1)/2

class pulse:
    'generator impulsów róznych'
    def __init__(self,x):
        self.x=x
        self.cen=0
        self.w=0
        self.pow=2
        self.dtype=np.cdouble
        self.chirp=0
        self.freq=0
    def tofile(self, fname):
        x=(self.x - self.cen)
        e=(x/self.w)**self.pow+1j*self.freq*x+1j*self.chirp*x*x
        return np.exp(e, dtype=self.dtype).tofile(fname)

def ramparam(g, Delta, delta):
    'raman param calc from'
    Gammac = 2 * np.pi * 5.9
    gammac = 2 * np.pi * 10e-3
    i=1j
    class RunParams(xmds.ParamsClass):
        # było:
        # dE_dz = - i * g * (conj(rho12) * Omega + E)/(2*Delta+i*Gammac) * exp(-(z-0.5)*(z-0.5)/(sa*sa));
        # sprzęgamy E
        # drho12_dt = (0.5*i*Omega*conj(E))/(2*Delta - i*Gammac) +
        #  rho12*(-4*i*delta*Delta - 2*Delta*gammac - 2*delta*Gammac + gammac*Gammac*i
        #  + i*Omega*conj(Omega))/(2*(2*Delta - i*Gammac))
        #  + i*fi*sw*rho12 + i*grad*rho12 + i*grzt*rho12;

        CErho = np.conj(-i*g/(2*Delta+i*Gammac))
        CrhoE = 0.5j/(2*Delta - i*Gammac)
        SCpl = 1j/(2*(2*Delta - i*Gammac))
    return RunParams()

def rect(x, T):
    'prostokąt o okresie T'
    lx, ly = x.shape
    tab = np.zeros([lx, ly])
    for i in range(ly):
        for j in range(lx):
            if (i % T >= T / 2):
                tab[j, i] = 1
    return tab

def rect_sum(x, T, N):
    'prostokąt jako suma N cosinusów o okresie T'
    lx, ly = x.shape
    tab = np.ones([lx, ly]) * 0.5    
    for i in range(N):
        tab += 2 / ((2 * i + 1) * np.pi) * np.cos((2 * i + 1) * 2 * np.pi * x / T + ((-1) ** i - 1) * np.pi / 2)
    return tab

def integrals(sid, r, compileParams, savepath):
    Efinal = np.sum(np.absolute(r.E[:,-1])**2) * compileParams.tmax/compileParams.tsteps
    E = np.sum(np.absolute(r.E[:,0])**2) * compileParams.tmax/compileParams.tsteps
    rho12final = np.sum(np.absolute(r.rho12[-1,:])**2 * sid.n_at)  * 2 * compileParams.zradious/compileParams.zsteps
    result = Efinal + rho12final - E
    eff = rho12final/E
    rho12kz0_final = np.sum(np.absolute(r.rho12kz[-1,compileParams.zsteps//2-5:compileParams.zsteps//2+5])**2 * sid.n_at[compileParams.zsteps//2-5:compileParams.zsteps//2+5])  * 2 * compileParams.zradious * 1.5626 ** 2 * 10/compileParams.zsteps
    rho12kz0_mid = np.sum(np.absolute(r.rho12kz[compileParams.tsteps//2,compileParams.zsteps//2-5:compileParams.zsteps//2+5])**2 * sid.n_at[compileParams.zsteps//2-5:compileParams.zsteps//2+5])  * 2 * compileParams.zradious * 1.5626 ** 2 * 10/compileParams.zsteps    
    consts = [["c_rect", sid.c_rect], ["c_cpl1", sid.c_cpl1], ["c_cpl2", sid.c_cpl2], ['E final', Efinal], ["rho12kz0-mid", rho12kz0_mid], ['rho12 final', rho12final], ['E in', E], ["sum", result], ["efficiency", eff], ["rho12kz0-final", rho12kz0_final]]
    print (consts)
    file1 = open(savepath+"/integrals.txt","a")
    for i in range(len(consts)):
        file1.write(consts[i][0])
        file1.write(" ")
        file1.write(str(consts[i][1]))
        file1.write("\n")
    file1.write("\n")
    file1.close() 
    
def find_kzmax(r, compileParams):
    return (np.where(np.abs(r.rho12kz[(compileParams.tsteps//2):,:]) == np.abs(r.rho12kz[(compileParams.tsteps//2):,:]).max())[1][0] - (compileParams.zsteps//2)) * 2 * 1.5625

def save_txt(sid, r, compileParams, savepath):
    integrals(sid, r, compileParams, savepath)
    np.savetxt(savepath+'Efinal.txt', r.E.view(float))
    np.savetxt(savepath+'rho12final.txt', r.rho12.view(float))
    np.savetxt(savepath+'rho12kzfinal.txt', r.rho12kz.view(float))