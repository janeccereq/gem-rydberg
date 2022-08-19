#%% ustaw interfejs
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
sim = xmds.Simulation().write_xmds('gem2d.tmpl.xmds', compileParams() .toDict())
sim.compile()

#%% przygotowanie pól wejściowych
class SimInputData(xmds.SimInputData):
    tmax = 60; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 500
    xradious = 0.1; xsteps = 30
    z = np.linspace(-zradious, zradious, num=zsteps)
    x = np.linspace(-xradious, xradious, num=xsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    OmegaS = 2 * np.pi * 4.7 # MHz
    OmegaC = 2 * np.pi * 4.7 # MHz

    alfa = 2 * np.pi * 0.04 # MHz / us
    beta = 2 * np.pi * 1.7 # MHz / cm

    mot_x = xradious / 3
    mot_z = zradious * 0.95

    gradBoff = 28
    gradBon = 32
    modon = 28
    modoff = 32
    
    Gamma = 2 * np.pi * 6.066
    Delta = 2 * np.pi * 70
    OD = 76
    
    n_at_z = supergauss(z, w=mot_z, pow=4, dtype=np.double)
    n_at_x = supergauss(x, w=mot_x, pow=2, dtype=np.double)
    n_at = np.outer(n_at_z,n_at_x)
    
    E_init_t = np.sin(2 * np.pi * t / 4) * (t < 24) * (t > 2) * np.exp(1j* np.pi * (-1) ** (t // 2) * (t < 24))
    E_init_x = supergauss(x, 0, 0.1, dtype = np.complex128)
    E_init = OmegaS * np.outer(E_init_x, E_init_t)
    
    Cpl_t = supergauss(t, cen = gradBoff/2, w = gradBoff/2, pow = 2, dtype = np.complex128) + supergauss(t, cen = (tmax+gradBon)/2, w = (tmax-gradBon)/2, pow = 4, dtype = np.complex128)
    Cpl_x = supergauss(x, cen = 0, w = 0.1, pow = 4, dtype = np.complex128)
    Cpl = OmegaC * np.outer(Cpl_x, Cpl_t)
    
    rho12_init_x=0j * x
    rho12_init_z=0j * z
    rho12_init = np.outer(rho12_init_z,rho12_init_x)
    
    X, Z, T = np.meshgrid(x, z, t)
    gradB = -beta * 1j * (Z * (T < gradBoff) - Z * (T > gradBon))
    mod = - 1j * beta ** 2 / (2 * alfa) * Z ** 2 / 4  * (T > modon) * (T < modoff) + 1j * (alfa * (T-12)) * (T < 24)
    Mod = gradB + mod


    def save(self, timestamp):
        with open('xmds_out/gem2d.Atimestamp_'+timestamp+'.pkl', 'wb') as file:
            dill.dump(self, file)
        path = xmds.init_path() / timestamp
        path.mkdir(parents=True, exist_ok=True)
        sid.n_at.tofile(str(path / 'n_at.bin'))
        sid.E_init.tofile(str(path / 'E_init.bin'))       
        sid.Cpl.tofile(str(path / 'Cpl.bin'))
        sid.Mod.tofile(str(path / 'Mod.bin'))
        sid.rho12_init.tofile(str(path / 'rho12_init.bin'))


sid = SimInputData()

#%% uruchomienie symulacji
def runParams():
    class RunParams(xmds.ParamsClass):
        kE = 2*np.pi/0.795e-4
        kEx = 0
        CErho = sid.OD * sid.Gamma * 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        CrhoE = 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        Srho = 0j#-1j / (4 * sid.Delta + 2 * 1j * sid.Gamma)
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
        xradious = sid.xradious
        xsteps = sid.xsteps
    return RunParams()
rpars=runParams()
sid.save(rpars.Atimestamp)
sim.run(rpars.toDict(True), 'gem2d')

#%% wczytanie wyników
sid, r = xmds.load_output(-1, 'gem2d')

#%% przygotowanie pól wejściowych
class SimInputData(xmds.SimInputData):
    tmax = 60; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 500
    xradious = 0.1; xsteps = 30
    z = np.linspace(-zradious, zradious, num=zsteps)
    x = np.linspace(-xradious, xradious, num=xsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    OmegaS = 2 * np.pi * 4.7 # MHz
    OmegaC = 2 * np.pi * 4.7 # MHz

    alfa = 2 * np.pi * 0.04 # MHz / us
    beta = 2 * np.pi * 1.7 # MHz / cm

    mot_x = xradious / 3
    mot_z = zradious * 0.95

    gradBoff = 28
    gradBon = 32
    modon = 28
    modoff = 32
    
    Gamma = 2 * np.pi * 6.066
    Delta = 2 * np.pi * 70
    OD = 76
    
    n_at_z = supergauss(z, w=mot_z, pow=4, dtype=np.double)
    n_at_x = supergauss(x, w=mot_x, pow=2, dtype=np.double)
    n_at = np.outer(n_at_z,n_at_x)
    
    E_init_t = supergauss(t, 8, 1, dtype = np.complex128) + supergauss(t, 16, 1, dtype = np.complex128)
    E_init_x = supergauss(x, 0, 0.1, dtype = np.complex128)
    E_init = OmegaS * np.outer(E_init_x, E_init_t)
    
    Cpl_t = supergauss(t, cen = gradBoff/2, w = gradBoff/2, pow = 2, dtype = np.complex128) + supergauss(t, cen = (tmax+gradBon)/2, w = (tmax-gradBon)/2, pow = 4, dtype = np.complex128)
    Cpl_x = supergauss(x, cen = 0, w = 0.1, pow = 4, dtype = np.complex128)
    Cpl = OmegaC * np.outer(Cpl_x, Cpl_t)
    
    rho12_init_x=0j * x
    rho12_init_z=0j * z
    rho12_init = np.outer(rho12_init_z,rho12_init_x)
    
    X, Z, T = np.meshgrid(x, z, t)
    gradB = -beta * 1j * (Z * (T < gradBoff) - Z * (T > gradBon))
    mod = - 1j * beta ** 2 / (2 * alfa) * Z ** 2 / 4  * (T > modon) * (T < modoff) + 1j * (alfa * (T-12)) * (T < 24)
    Mod = gradB + mod


    def save(self, timestamp):
        with open('xmds_out/gem2d.Atimestamp_'+timestamp+'.pkl', 'wb') as file:
            dill.dump(self, file)
        path = xmds.init_path() / timestamp
        path.mkdir(parents=True, exist_ok=True)
        sid.n_at.tofile(str(path / 'n_at.bin'))
        sid.E_init.tofile(str(path / 'E_init.bin'))       
        sid.Cpl.tofile(str(path / 'Cpl.bin'))
        sid.Mod.tofile(str(path / 'Mod.bin'))
        sid.rho12_init.tofile(str(path / 'rho12_init.bin'))


sid = SimInputData()

#%% uruchomienie symulacji
def runParams():
    class RunParams(xmds.ParamsClass):
        kE = 2*np.pi/0.795e-4
        kEx = 0
        CErho = sid.OD * sid.Gamma * 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        CrhoE = 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        Srho = 0j#-1j / (4 * sid.Delta + 2 * 1j * sid.Gamma)
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
        xradious = sid.xradious
        xsteps = sid.xsteps
    return RunParams()
rpars=runParams()
sid.save(rpars.Atimestamp)
sim.run(rpars.toDict(True), 'gem2d')

#%% wczytanie wyników
sid2, r2 = xmds.load_output(-1, 'gem2d')

#%% rysowanie
import matplotlib
arr = np.zeros(300)
arr[150:] += 1 #tablica, żeby wyciąć część pola z odczytu

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
plt.subplots_adjust(wspace=0, hspace=0)

ax1.set_xticks([])
ax1.set_xlim(0,60)
ax1.set_ylim(0,900)
ax1.plot(r.t, np.abs(sid2.E_init[15, :300])**2 + 10 * np.abs(r2.E[:,-1,15]*arr)**2, 'r')
ax1.vlines(30, 0, 900, color = 'black', linestyles='dashed')

ax2.set_xticks([])
ax2.set_ylabel(r'$k_z [\frac{2 \pi}{mm}]$')
ax2.set_ylim(-5,0)
ax2.pcolormesh(r.t,  1/20/np.pi * r.kz,np.abs(r2.rho12kz[:,:,15].T)**2, shading='auto', cmap = 'Reds', norm=matplotlib.colors.LogNorm(), vmin = 0.0001, vmax = 0.01)

ax3.set_xticks([])
ax3.set_xlim(0,60)
ax3.set_ylim(0,900)
ax3.plot(r.t, np.abs(sid.E_init[15, :300])**2 + np.abs(r.E[:,-1,15]*arr)**2, 'r')

ax4.set_ylabel(r'$k_z [\frac{2 \pi}{mm}]$')
ax4.set_xlabel('t [$\mu$s]')
ax4.set_ylim(-5,0)
ax4.pcolormesh(r.t,  1/20/np.pi * r.kz,np.abs(r.rho12kz[:,:,15].T)**2, shading='auto', cmap = 'Reds', norm=matplotlib.colors.LogNorm(), vmin = 0.0001, vmax = 0.01)

fig.show()
fig.canvas.draw()

# %%