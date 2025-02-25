#%%
%matplotlib qt5
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np

#%%
r = np.linspace(-1,1,2048)
dr= r[1] - r[0]

kr = np.fft.fftfreq(len(r), dr) * 2 * np.pi

rb = 0.05
coeff = np.sqrt(12)/rb
Vr = coeff/(1+(r/rb)**6)
EE0 = np.ones_like(r) + 0j

Rsteps = 2000
dR = 0.0001
ODL = 5 / rb
OmG = 0.2
CVR = -1j
CVr = 0
Cr =  4j/coeff

kr2 = -kr * kr
dif = np.exp(-Cr * kr * kr * dR)

dVr = -6*r**5*rb**6*coeff/(r**6+rb**6)**2
ddVr = 6*r**4*rb**6*(7*r**6-5*rb**6)*coeff/(r**6+rb**6)**3

loc = np.exp(-(CVR * Vr - CVr * ddVr) * dR)

#%%
EE = [EE0,]
EE3 = EE0
for step in range(Rsteps):
    print(step)
    EEVr = EE3 * Vr
    EEVrf = np.fft.fft(EEVr)
    EEVrf2 = EEVrf * np.exp(CVr * kr2 * dR)
    EE1f = EEVrf2 - EEVrf + np.fft.fft(EE3)
    EE1f *= dif
    EE2 = np.fft.ifft(EE1f)
    EE2Vr = EE2 * dVr
    EE2Vrf = np.fft.fft(EE2Vr)
    EE2Vrf2 = EE2Vrf * np.exp(-2 * 1j * CVr * kr * dR)
    EE2f = EE2Vrf2 - EE2Vrf + EE1f
    EE2 = np.fft.ifft(EE2f)
    EE3 = EE2 * loc
    
    EE.append(EE3)

#%%
R = np.linspace(0, dR * Rsteps, Rsteps)
import matplotlib.pyplot as plt
plt.title('$|EE|^2$')
plt.xlabel('r [$mm$]')
plt.ylabel('R [$mm$]')
cmap = plt.pcolormesh(r, R, np.abs(EE[1:])**2, cmap='BrBG')
plt.colorbar(cmap)
plt.xlim(-4*rb, 4*rb)
plt.ylim(0, 0.12)
plt.show()

#%%
cmap = plt.pcolormesh(r, R, np.angle(EE[1:]), cmap="twilight_shifted")
plt.colorbar(cmap)
plt.xlim(-4*rb, 4*rb)
plt.ylim(0, 0.12)
