"""
testowa symulacja EIT 2 fotony
"""
# %% inicjalizacja matplotlib z intefejsem niezależnych okienek
%gui qt5
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.use('Qt5Agg')
#kompilacja xmds
%load_ext autoreload
%autoreload 2
import numpy as np
import xmds
import dill
#%%
class compileParams(xmds.ParamsClass):
    'parametry podawane przed kompilacją'
    tsteps = 100
    tsamples = 100
sim = xmds.Simulation().write_xmds('2eit1t_gem.tmpl.xmds', compileParams().toDict())
sim.compile()

# %% przygotowanie pól wejściowych - przypadek równoległy
import os
from xmdswrapper.raphys import supergauss, square, ramparam, rect, rect_sum, save_txt
from xmdswrapper.raplots import plot_zt, reimplot, in1dfigures, save_all
from scipy import signal
from matplotlib import pyplot as plt

class SimInputData(xmds.SimInputData):
    tmax = 4; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 128
    z = np.linspace(-zradious, zradious, num=zsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    alfa = 2 * np.pi * 0.04 / 5 # MHz / us
    beta = 2 * np.pi * 1.7 * 64 # MHz / cm
    OmegaRc = 2 * np.pi * 4.7 # MHz

    OD = 76
    Gamma = 2 * np.pi * 6.066
    Delta = 2 * np.pi * 70
    Cpl0 = OmegaRc
    c0 = -2
    n0 = OD * Gamma #*np.abs(c0)


    bond1 = 1/(1+np.exp(100*(z-0.9)))
    bond = 10000 * (1 - np.outer(bond1, bond1))
    
    E_init = 0j * z
    EE_init = np.outer(E_init, E_init)

    Cpl_t = Cpl0 * np.ones_like(t) + 0j
    
    Z, T = np.meshgrid(z, t)

    g_at = n0 ** 0.5 * supergauss(Z, 0, 0.6, dtype=np.complex128)
    
    gradB = beta * 1j * Z

    Mod_zt = gradB
    
    rho12_init = 0j * z #-E_init / Cpl0
    SS_init = np.outer(rho12_init, rho12_init)
    SS_init = np.load('SS_write_beta683.61Omega29.53.npy')

    pol_init = 0j * z #-1j * (grv / 2 / n0) ** 0.5 * np.diff(E_init, n = 1, append = 0)
    PP_init = np.outer(pol_init, pol_init)
    
    EP_init = np.outer(E_init, pol_init)
    ES_init = np.outer(E_init, rho12_init)
    PS_init = np.outer(pol_init, rho12_init)
    

    def save(self, timestamp):
        with open('xmds_out/2eit1t.Atimestamp_'+timestamp+'.pkl', 'wb') as file:
            dill.dump(self, file)

        path = xmds.init_path() / timestamp
        path.mkdir(parents=True, exist_ok=True)
        sid.bond.tofile(str(path / 'bond.bin'))
        sid.EE_init.tofile(str(path / 'EE_init.bin'))       
        sid.Cpl_t.tofile(str(path / 'Cpl_t_init.bin'))
        sid.g_at.transpose().tofile(str(path / 'g_at_zt_init.bin'))
        sid.Mod_zt.transpose().tofile(str(path / 'Mod_zt.bin'))
        sid.SS_init.tofile(str(path / 'SS_init.bin'))
        sid.PP_init.tofile(str(path / 'PP_init.bin'))
        sid.EP_init.tofile(str(path / 'EP_init.bin'))
        sid.ES_init.tofile(str(path / 'ES_init.bin'))
        sid.PS_init.tofile(str(path / 'PS_init.bin'))

sid = SimInputData()

#% % uruchomienie symulacji
#%%
def runParams():
    class RunParams(xmds.ParamsClass):
        GammaP = 2 * sid.Gamma + 4j * sid.Delta
        c0 = sid.c0
        c6 = 2.5e-7
        Vmax = np.pi / sid.tmax * sid.tsteps
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
    return RunParams()
rpars=runParams()
sid.save(rpars.Atimestamp)
sim.run(rpars.toDict(True), '2eit1t')

#%%
sid, r = xmds.load_output(-1, '2eit1t')
#%%
from oglad import Oglad
oglad = Oglad()
oglad.init_win().set_result(r)
oglad.win.activateWindow()

#%% pole na detektorze
z1 = 118
zout = np.linspace(-2*sid.z[z1],2*sid.z[z1],2*z1)
Ez1 = r.EE[:,z1,:z1]
Ez2 = r.EE[:,:z1,z1]
Eout = []
tout = []
for i in range(sid.tsteps):
    Eout.append(np.append(Ez1[i], np.flip(Ez2[i])))
    tout.append(sid.t[i]+np.abs(zout)/np.abs(sid.c0))
#%%
plt.title(r'$|EE|^2_{out}$')
plt.xlabel(r'$t_1-t_2$ [$\mu s$]')
plt.ylabel(r't [$\mu s$]')
plt.ylim(0,2)
plt.pcolormesh(zout / np.abs(sid.c0), tout, np.abs(Eout)**2, cmap="Reds")
plt.show()

#%% wydajność
dz = 2 * sid.zradious / sid.zsteps
E_init = np.sum(np.abs(sid.EE_init)**2) * dz ** 2
E_saved = np.sum(np.abs(sid.n0 ** 0.5 * np.outer(supergauss(sid.z, 0, 0.6, dtype=np.complex128),supergauss(sid.z, 0, 0.6, dtype=np.complex128)))**2*np.abs(r.SS[-1])**2) * dz ** 2
eff_w = E_saved / E_init
print (eff_w)
#%%
dz = 2 * sid.zradious / sid.zsteps
dt = np.max(tout) / len(tout)
eff_r = np.sum(np.abs(Eout)**2 * np.abs(sid.c0) * dt * dz) / np.sum(np.abs(sid.n0 ** 0.5 * supergauss(sid.z, 0, 0.6, dtype=np.complex128))**2*np.abs(r.SS[0])**2 * dz ** 2)
print (eff_r)
#%%
np.save('SS_write_beta'+f'{sid.beta:.2f}', r.SS[-1])
#%%
plt.title('|SS|')
plt.xlabel(r'$z_1$')
plt.ylabel(r'$z_2$')
plt.pcolormesh(r.z1, r.z2,  np.abs(r.SS[1]))
plt.show()
#%%
