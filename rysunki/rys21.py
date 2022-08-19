# %% 
%matplotlib qt5
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np
import xmds
import dill
from phys import supergauss

#%% kompilacja xmds
class compileParams(xmds.ParamsClass):
    'parametry podawane przed kompilacją'
    tsteps = 100
sim = xmds.Simulation().write_xmds('gem_rydberg1d.tmpl.xmds', compileParams().toDict())
sim.compile()

# %% przygotowanie pól wejściowych
class SimInputData(xmds.SimInputData):
    tmax = 1; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 128
    z = np.linspace(-zradious, zradious, num=zsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    gradBoff = 2
    gradBon = 3
    modon = 2
    modoff = 3

    alfa = 2 * np.pi * 0.04 / 5 # MHz / us
    beta = 2 * np.pi * 1.7 * 8 # MHz / cm
    OmegaC = 2 * np.pi * 4.7 / 2 # MHz

    OD = 76
    Gamma = 2 * np.pi * 6.066
    Delta = 2 * np.pi * 70
    Cpl0 = OmegaC
    c0 = -2
    n0 = 3e3

    bond1 = 1/(1+np.exp(100*(z-0.9)))
    bond = 10000 * (1 - np.outer(bond1, bond1))
    
    E_init = supergauss(z, -0.92, 0.05, dtype = np.complex128)
    EE_init = np.outer(E_init, E_init)

    Cpl_t = Cpl0 * np.ones_like(t) + 0j
    
    Z, T = np.meshgrid(z, t)

    g_at = n0 ** 0.5 * supergauss(Z, 0, 0.6, dtype=np.complex128) * (T > 0.03)
    
    gradB = -beta * 1j * Z

    Mod_zt = gradB
    
    rho12_init = 0j * z
    SS_init = np.outer(rho12_init, rho12_init)

    pol_init = 0j * z
    PP_init = np.outer(pol_init, pol_init)
    
    EP_init = np.outer(E_init, pol_init)
    ES_init = np.outer(E_init, rho12_init)
    PS_init = np.outer(pol_init, rho12_init)
    

    def save(self, timestamp):
        with open('xmds_out/gem_rydberg1d.Atimestamp_'+timestamp+'.pkl', 'wb') as file:
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

#%% uruchomienie symulacji
def runParams():
    class RunParams(xmds.ParamsClass):
        GammaP = sid.Gamma / 2 + 1j * sid.Delta
        c0 = sid.c0
        c6 = 2.5e-7
        Vmax = np.pi / sid.tmax * sid.tsteps
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
    return RunParams()
rpars=runParams()
sid.save(rpars.Atimestamp)
sim.run(rpars.toDict(True), 'gem_rydberg1d')

#%% wczytanie wyników
sid, r = xmds.load_output(-1, 'gem_rydberg1d')
np.save('SS.npy', r.SS[-1])

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title(r'$|SS|^2 |_{t=0.1 \mu s}$')
ax1.set_xlabel(r'$z_1$ [$mm$]')
ax1.set_ylabel(r'$z_2$ [$mm$]')
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.pcolormesh(r.z1, r.z2, np.abs(r.SS[20])**2, shading='auto', cmap='Reds')

ax2.set_title(r'$|SS|^2 |_{t=0.4 \mu s}$')
ax2.set_xlabel(r'$z_1$ [$mm$]')
ax2.set_ylabel(r'$z_2$ [$mm$]')
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.pcolormesh(r.z1, r.z2, np.abs(r.SS[40])**2, shading='auto', cmap='Reds')

ax3.set_title(r'$|SS|^2 |_{t=1 \mu s}$')
ax3.set_xlabel(r'$z_1$ [$mm$]')
ax3.set_ylabel(r'$z_2$ [$mm$]')
ax3.set_xlim(-1,1)
ax3.set_ylim(-1,1)
ax3.pcolormesh(r.z1, r.z2, np.abs(r.SS[-1])**2, shading='auto', cmap='Reds')

fig.show()
fig.canvas.draw()

#%%