#%%
%matplotlib qt5
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np

#%%
save_every = 10

r = np.linspace(-1,1,512) 
ro = np.linspace(-0.5,0.5,512)
dr = r[1] - r[0]
dro = ro[1] - ro[0]

rm, rom = np.meshgrid(r, ro, indexing='ij')
rom = rom*0
kr = np.fft.fftfreq(len(r), dr) * 2 * np.pi
kro = np.fft.fftfreq(len(ro), dro) * 2 * np.pi

Kr, Kro = np.meshgrid(kr, kro, indexing='ij')

rb = 0.05
coeff = np.sqrt(12)/rb
k0 = 2 * np.pi / 0.000795
Vr = 1/(1+((rm**2+(rom)**2)/rb**2)**3)
EE0 = np.ones_like(rm) + 0j

Rsteps =2000
dR = 0.005

ODL = 5 / rb
OmG = 0.2
CVR = -1j
CVr = 0
Cr =  4j/coeff
CD = 1 / 2 / k0 * 0
kr2 = -Kr * Kr
dif = np.exp(-Cr * Kr * Kr * dR - CD * 1j * Kro * Kro * dR)

potVr = np.exp(CVr * kr2 * dR)
potdVr = np.exp(-2 * 1j * CVr * Kr * dR)
potro = 1j * Kro



# dVr = 12*1j*rb**6*rm*(rm**2+rom**2)**2/(rb**6-2*1j*(rm**2+rom**2)**3)**2
# ddVr = 12*1j*((rm**2+rom**2)**2+4*rm**2*(rm**2+rom**2))/rb**6/(1-2*1j*(rm**2+rom**2)**3/rb**6)**2-288*rm**2*(rm**2+rom**2)**4/rb**12/(1-2*1j*(rm**2+rom**2)**3/rb**6)**3

dVr = -6*rm*coeff*(rm**2+rom**2)**2 / (rb**6*(1+(rm**2+rom**2)**3/rb**6)**2)
ddVr = 6*rb**6*coeff*(rm**2+rom**2)*(7*rm**8+20*rm**6*rom**2+18*rm**4*rom**4-rom**2*(rb**6+rom**6)+r**2*(-5*rb**4+4*rom**6)) / (rm**6+rb**6+3*rm**4*rom**2+3*rm**2*rom**4+rom**6)**3

loc = np.exp(-(CVR * Vr - CVr * ddVr) * dR)
#%%
EE = [EE0,]
EE4 = EE0
for step in range(Rsteps):
    print(step)
    EEVr = EE4 * Vr
    EEVrf = np.fft.fftn(EEVr)
    EEVrf2 = EEVrf * potVr
    EE1f = EEVrf2 - EEVrf + np.fft.fftn(EE4)
    EE1f *= dif
    EE2 = np.fft.ifftn(EE1f)
    EE2Vr = EE2 * dVr
    EE2Vrf = np.fft.fftn(EE2Vr)
    EE2Vrf2 = EE2Vrf * potdVr
    EE2f = EE2Vrf2 - EE2Vrf + EE1f
    EE2 = np.fft.ifftn(EE2f)
    EE2f2 = EE2f * potro
    EE22 = np.fft.ifftn(EE2f2)
    EE3 = EE2 + 1j * CD / ro * dR * EE22
    EE4 = EE3 * loc
    
    if step % save_every == 0:
        EE.append(EE4)

#%%
from slicer import SliceWindow
sw = SliceWindow()
sw.setdata(np.abs(np.array(EE))**2)
#sw.setdata(np.angle(np.array(EE)))

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title(r'$|EE|^2 | _{\rho=0}$')
ax1.set_xlabel('r [$mm$]')
ax1.set_ylabel('R [$mm$]')
ax1.pcolormesh(np.angle(np.array(EE)[1:,:,len(ro) // 2]), shading='auto', cmap='twilight')

ax2.set_title(r'$|EE|^2 | _{r=0}$')
ax2.set_xlabel(r'$\rho$ [$mm$]')
ax2.set_ylabel('R [$mm$]')
ax2.pcolormesh( np.angle(np.array(EE)[1:,len(r) // 2,:]), shading='auto', cmap='twilight')

ax3.set_title(r'$|EE|^2 | _{R=0}$')
ax3.set_xlabel(r'$\rho$ [$mm$]')
ax3.set_ylabel('r [$mm$]')
cmap = ax3.pcolormesh(np.angle(np.array(EE)[len(EE) // 2,:,:]), shading='auto', cmap='twilight')
plt.colorbar(cmap)
fig.show()
fig.canvas.draw()

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title(r'$|EE|^2 | _{\rho=0}$')
ax1.set_xlabel('r [$mm$]')
ax1.set_ylabel('R [$mm$]')
ax1.pcolormesh(np.abs(np.array(EE)[1:,:,len(ro) // 2])**2, shading='auto', cmap='seismic')

ax2.set_title(r'$|EE|^2 | _{r=0}$')
ax2.set_xlabel(r'$\rho$ [$mm$]')
ax2.set_ylabel('R [$mm$]')
ax2.pcolormesh( np.abs(np.array(EE)[1:,len(r) // 2,:])**2, shading='auto', cmap='seismic')

ax3.set_title(r'$|EE|^2 | _{R=0}$')
ax3.set_xlabel(r'$\rho$ [$mm$]')
ax3.set_ylabel('r [$mm$]')
cmap = ax3.pcolormesh(np.abs(np.array(EE)[len(EE) // 2,:,:])**2, shading='auto', cmap='seismic')
plt.colorbar(cmap)
fig.show()
fig.canvas.draw()
