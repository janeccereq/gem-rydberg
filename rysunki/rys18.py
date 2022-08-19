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

#%%
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

#%%