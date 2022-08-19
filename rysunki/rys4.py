#%% importy i inicjalizacja matplotlib z intefejsem niezależnych okienek
%gui qt5
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
    tsteps = 300  
sim = xmds.Simulation().write_xmds('gem1d.tmpl.xmds', compileParams() .toDict())
sim.compile()

#%% przygotowanie pól wejściowych
class SimInputData(xmds.SimInputData):
    tmax = 5; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 256
    z = np.linspace(-zradious, zradious, num=zsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    beta = 200
    OmegaC = 4

    gradBoff = 2
    gradBon = 3

    n_at = supergauss(z, w=0.7, pow=4, dtype=np.double)

    E_init = E_init = supergauss(t, 0.5, 0.1, dtype = np.complex128) + supergauss(t, 1, 0.1, dtype = np.complex128) + supergauss(t, 1.5, 0.1, dtype = np.complex128)

    Cpl = OmegaC * (supergauss(t, cen = gradBoff/2, w = gradBoff/2, pow = 4, dtype = np.complex128) + supergauss(t, cen = (tmax+gradBon)/2, w = (tmax-gradBon)/2, pow = 4, dtype = np.complex128))

    Z, T = np.meshgrid(z, t)

    gradB = beta * 1j * (Z * (T < gradBoff) - Z * (T > gradBon))
    mod = 0j
    Mod = gradB + mod

    rho12_init = 0j * z


    def save(self, timestamp):
        with open('xmds_out/gem1d.Atimestamp_'+timestamp+'.pkl', 'wb') as file:
            dill.dump(self, file)
        path = xmds.init_path() / timestamp
        path.mkdir(parents=True, exist_ok=True)
        sid.n_at.tofile(str(path / 'n_at.bin'))
        sid.E_init.tofile(str(path / 'E_init.bin'))       
        sid.Cpl.tofile(str(path / 'Cpl.bin'))
        sid.Mod.transpose().tofile(str(path / 'Mod.bin'))
        sid.rho12_init.tofile(str(path / 'rho12_init.bin'))


sid = SimInputData()

# %% uruchomienie symulacji
def runParams():
    class RunParams(xmds.ParamsClass):
        CErho = 5 + 0j
        CrhoE = -5 + 0j
        SCpl = 0j
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
    return RunParams()
rpars=runParams()
sid.save(rpars.Atimestamp)
sim.run(rpars.toDict(True), 'gem1d')

#%% wczytanie wyników
sid, r = xmds.load_output(-1, 'gem1d')

#%% rysowanie
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title(r'$|E|^2$')
ax1.set_xlabel('z [cm]')
ax1.set_ylabel('t [$\mu$s]')
ax1.set_xlim(-1,1)
ax1.set_ylim(0,5)
ax1.pcolormesh(r.z, r.t, np.abs(r.E)**2, shading='auto', cmap='Reds')

ax2.set_title(r'$|\rho_{1,2}|^2$')
ax2.set_xlabel('z [cm]')
ax2.set_yticks([])
ax2.set_xlim(-1,1)
ax2.pcolormesh(r.z, r.t,np.abs(r.rho12)**2, shading='auto', cmap='Reds')

ax3.set_title(r'$|\widetilde{\rho}_{1,2}|^2$')
ax3.set_xlabel(r'$kz [\frac{1}{cm}]$')
ax3.set_yticks([])
ax3.pcolormesh(r.kz, r.t, np.abs(r.rho12kz)**2, shading='auto', cmap='Reds')

fig.show()
fig.canvas.draw()

# %%
