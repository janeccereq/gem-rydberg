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
sim = xmds.Simulation().write_xmds('rydberg1d.tmpl.xmds', compileParams().toDict())
sim.compile()

# %% przygotowanie pól wejściowych
import dill

class SimInputData(xmds.SimInputData):
    tmax = 4; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 256
    z = np.linspace(-zradious, zradious, num=zsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    path = xmds.init_path()

    n0 = 7e3
    Cpl0 = 20
    c0 = -1.4
    grv = -Cpl0 ** 2 * c0 / (Cpl0 ** 2 + n0)

    n_at = n0*np.ones_like(z)

    E_init = np.ones_like(z) + 0j
    EE_init = np.outer(E_init, E_init)

    Cpl_t = Cpl0 * np.ones_like(t) + 0j

    Z, T = np.meshgrid(z, t)

    Mod_zt = 0j * (-0.3<Z)*(Z<0.3) * (0.9 < T)*(T < 1.1)

    rho12_init = -E_init / Cpl0
    SS_init = np.outer(rho12_init, rho12_init)

    pol_init = -1j * (grv / 2 / n0) ** 0.5 * np.diff(E_init, n = 1, append = 0)
    PP_init = np.outer(pol_init, pol_init)

    EP_init = np.outer(E_init, pol_init)
    ES_init = np.outer(E_init, rho12_init)
    PS_init = np.outer(pol_init, rho12_init)

    def save(self, timestamp):
        with open('xmds_out/rydberg1d.Atimestamp_'+timestamp+'.pkl', 'wb') as file:
            dill.dump(self, file)

        path = xmds.init_path() / timestamp
        path.mkdir(parents=True, exist_ok=True)
        sid.EE_init.tofile(str(path / 'EE_init.bin'))       
        sid.Cpl_t.tofile(str(path / 'Cpl_t_init.bin'))
        sid.n_at.transpose().tofile(str(path / 'n_at.bin'))
        sid.Mod_zt.transpose().tofile(str(path / 'Mod_zt.bin'))
        sid.SS_init.tofile(str(path / 'SS_init.bin'))
        sid.PP_init.tofile(str(path / 'PP_init.bin'))
        sid.EP_init.tofile(str(path / 'EP_init.bin'))
        sid.ES_init.tofile(str(path / 'ES_init.bin'))
        sid.PS_init.tofile(str(path / 'PS_init.bin'))

sid = SimInputData() 

#%% uruchomienie symulacji
def runParams():
    class RunParams(xmds.ParamsClass):
        GammaP = 100 + 0j
        c0 = sid.c0
        grv = sid.grv
        c6 = 2.5e-7
        Vmax = np.pi / sid.tmax * sid.tsteps
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
    return RunParams()
rpars=runParams()
sid.save(rpars.Atimestamp)
sim.run(rpars.toDict(True), 'rydberg1d')

#%% wczytanie wyników
sid, r = xmds.load_output(-1, 'rydberg1d')

#%% odczytanie EE na antydiagonali, odpowiadającemu EE(r=z1-z2)
n = r.EE.shape[1]
m = r.EE.shape[0]
rfull = np.linspace(-2 * sid.zradious, 2 * sid.zradious, n)
Rfull = np.linspace(-2 * sid.zradious, 2 * sid.zradious, m)
EEfull = np.zeros((m,n), dtype = complex)
for j in range(m):
    for i in range(n):
        EEfull[j, i] = r.EE[j, i, n-1-i]
        
#%% przybliżenie
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

#%%
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