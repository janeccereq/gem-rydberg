"""
symulacja EIT 2 fotony
porównanie wyników dla rozwiązywania pełnych równań i przybliżonego równania na EE
"""
# %% 
%matplotlib qt5
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np
import xmds
#%% kompilacja xmds
class compileParams(xmds.ParamsClass):
    'parametry podawane przed kompilacją'
    tsteps = 100
    tsamples = 100
sim = xmds.Simulation().write_xmds('2eit1t_rydberg_compare.tmpl.xmds', compileParams().toDict())
sim.compile()

# %% przygotowanie pól wejściowych
from matplotlib import pyplot as plt

savepath = str(xmds.root_path/'kz0/par')
class SimInputData(xmds.SimInputData):
    tmax = 4; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 256
    z = np.linspace(-zradious, zradious, num=zsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    path = xmds.init_path()

    n0 = 7e3
    Cpl0 = 20
    lightv = -1.4
    grv = -Cpl0 ** 2 * lightv / (Cpl0 ** 2 + n0)

    n_at = n0*np.ones_like(z)
    n_at.tofile(str(path / 'n_at.bin'))

    E_init = np.ones_like(z) + 0j #supergauss(z, 0, 0.5)
    EE_init = np.outer(E_init, E_init)
    EE_init.tofile(str(path / 'EE_init.bin'))

    Cpl_t = Cpl0 * np.ones_like(t) + 0j
    Cpl_t.tofile(str(path / 'Cpl_t_init.bin'))

    Z, T = np.meshgrid(z, t)

    Mod_zt = 0j * (-0.3<Z)*(Z<0.3) * (0.9 < T)*(T < 1.1)
    Mod_zt.transpose().tofile(str(path / 'Mod_zt.bin'))

    rho12_init = -E_init / Cpl0
    SS_init = np.outer(rho12_init, rho12_init)
    SS_init.tofile(str(path / 'SS_init.bin'))

    pol_init = -1j * (grv / 2 / n0) ** 0.5 * np.diff(E_init, n = 1, append = 0)
    PP_init = np.outer(pol_init, pol_init)
    PP_init.tofile(str(path / 'PP_init.bin'))

    EP_init = np.outer(E_init, pol_init)
    EP_init.tofile(str(path / 'EP_init.bin'))
    ES_init = np.outer(E_init, rho12_init)
    ES_init.tofile(str(path / 'ES_init.bin'))
    PS_init = np.outer(pol_init, rho12_init)
    PS_init.tofile(str(path / 'PS_init.bin'))


sid = SimInputData() 
#%% uruchomienie symulacji
def runParams():
    class RunParams(xmds.ParamsClass):
        GammaP = 100 + 0j
        lightv = sid.lightv
        grv = sid.grv
        d6 = 0.05
        VR = sid.Cpl0 ** 2 / GammaP + 0j
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
    return RunParams()
rpars=runParams()
sim.run(rpars.toDict(True), '2eit1t')
#%% wczytanie wyników
r = xmds.load_sim(-1)
#%% odczytanie EE na antydiagonali, odpowiadającemu EE(r=z1-z2)
n = r.EE.shape[1]
m = r.EE.shape[0]
rfull = np.linspace(-2 * sid.zradious, 2 * sid.zradious, n)
Rfull = np.linspace(-2 * sid.zradious, 2 * sid.zradious, m)
EEfull = np.zeros((m,n), dtype = complex)
for j in range(m):
    for i in range(n):
        EEfull[j, i] = r.EE[j, i, n-1-i]
#%% okienko do oglądania wyników
import pyqtgraph as pg
import numpy as np
import importlib
from ogladnew import Oglad
oglad = Oglad()
oglad.init_win().set_result(r)
oglad.win.activateWindow()
#%%
fig = plt.figure()
fig.clear()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title(r'$|EE|^2$')
ax1.set_xlabel(r'$z_1$ [$mm$]')
ax1.set_ylabel(r'$z_2$ [$mm$]')
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.pcolormesh(r.z1, r.z2, np.abs(r.EE[-1])**2, shading='auto', cmap='Reds')

ax2.set_title(r'$|SS|^2$')
ax2.set_xlabel(r'$z_1$ [$mm$]')
ax2.set_ylabel(r'$z_2$ [$mm$]')
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.pcolormesh(r.z1, r.z2, np.abs(r.SS[-1])**2, shading='auto', cmap='Reds')

ax3.set_title(r'$|PP|^2$')
ax3.set_xlabel(r'$z_1$ [$mm$]')
ax3.set_ylabel(r'$z_2$ [$mm$]')
ax3.set_xlim(-1,1)
ax3.set_ylim(-1,1)
ax3.pcolormesh(r.z1, r.z2, np.abs(r.PP[-1])**2, shading='auto', cmap='Reds')

fig.show()
fig.canvas.draw()
#%% PRZYBLIŻENIE
r = np.linspace(-2,2,2048)
dr= r[1] - r[0]

kr = np.fft.fftfreq(len(r), dr) * 2 * np.pi

rb = rpars.d6
Vr = 1/(1-2*1j*(r/rb)**6)
EE0 = np.ones_like(r) + 0j

Rmax = sid.tmax * sid.grv
Rsteps = 2000
dR = Rmax / Rsteps
ODL = -2 * sid.n0 / sid.lightv / np.abs(rpars.GammaP)
OmG = sid.Cpl0 / np.abs(rpars.GammaP)
CVR = ODL
CVr = OmG ** 2 * 4 / ODL
Cr =  4 / ODL

kr2 = -kr * kr
dif = np.exp(-Cr * kr * kr * dR)

dVr = 12*1j*rb**6*r**5/(rb**6-2*1j*r**6)**2
ddVr = 60*1j*r**4/rb**6/(1-2*1j*(r/rb)**6)**2-288*r**10/rb**12/(1-2*1j*(r/rb)**6)**3

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

#%% porównanie końcowych wyników
fig = plt.figure("EE")
fig.clear();
plt.plot(r, np.abs(EE[-1]), label='approx')
plt.plot(rfull, np.abs(EEfull), label='full')
plt.xlabel('r')
plt.ylabel('|EE|')
plt.legend()

#%%

fig = plt.figure()
fig.clear()
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title(r'$|EE|^2$ approximate')
ax1.set_xlabel('r [$mm$]')
ax1.set_ylabel('R [$mm$]')
ax1.pcolormesh(np.abs(EE)**2, cmap='Reds')

ax2.set_title(r'$|EE|^2$ full')
ax2.set_xlabel('r [$mm$]')
ax2.set_yticks([])
ax2.pcolormesh(rfull, Rfull[1:], np.abs(EEfull[1:])**2, shading='auto', cmap='Reds')


fig.show()
fig.canvas.draw()
#%%