#%%
%matplotlib qt5
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np

#%%
nsteps = 1000
dR = 0.0001
ro = np.linspace(-0.02,0.02,1024)
dro = ro[1] - ro[0]
kro = np.fft.fftfreq(len(ro), d=dro) * 2 * np.pi

l = 0.000795
k0 = 2 * np.pi / l
rb = 0.003
zR = k0 * rb ** 2 / 2

CD = 1 / 2 / k0
dif = np.exp(-CD * 1j * kro * kro * dR)
potro = 1j * kro

EE0 = -1j / zR * (1 - ro ** 2 * k0 / 2 / zR) * np.exp(-ro ** 2 * k0 / 2 / zR)

EE = [EE0,]
EE3 = EE0
for step in range(nsteps):
    print(step)
    EE1f = np.fft.fft(EE3)
    EE1f *= dif
    EE2 = np.fft.ifft(EE1f)
    EE2f = np.fft.fft(EE2)
    EE2f2 = EE2f * potro
    EE22 = np.fft.ifft(EE2f2)
    EE3 = EE2 + 1j * CD * dR * EE22 / ro
    EE.append(EE3)


Z = np.linspace(dR, dR * len(EE), len(EE))
EG = []
for z in Z:
    l = 0.000795
    k0 = 2 * np.pi / l
    w0 = rb
    zR = k0 * w0 ** 2 / 2
    q = z + 1j * zR
    sig = 1 / (2j * (z - 1j * zR))
    E = -4j * zR * sig ** 2 * (1 - sig * ro ** 2 * k0) * np.exp(-sig * ro ** 2 * k0)
    EG.append(E)

#%%
from slicer_compare import SliceDisplay
sd = SliceDisplay()
sd.setresult(np.array(EE), np.array(EG))

# %%
