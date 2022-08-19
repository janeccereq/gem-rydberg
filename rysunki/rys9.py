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
    zradious = 1; zsteps = 250
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
    
    E_init_t=supergauss(t, 14, 0.5, dtype = np.complex128)
    E_init_x = supergauss(x, 0, 0.1, dtype = np.complex128)
    E_init = OmegaS * np.outer(E_init_x, E_init_t)
    
    Cpl_t = supergauss(t, cen = gradBoff/2, w = gradBoff/2, pow = 2, dtype = np.complex128) + supergauss(t, cen = (tmax+gradBon)/2, w = (tmax-gradBon)/2, pow = 4, dtype = np.complex128)
    Cpl_x = supergauss(x, cen = 0, w = 0.1, pow = 4, dtype = np.complex128)
    Cpl = OmegaC * np.outer(Cpl_x, Cpl_t)
    
    rho12_init_x=0j * x
    rho12_init_z=0j * z
    rho12_init = np.outer(rho12_init_z,rho12_init_x)
    
    X, Z, T = np.meshgrid(x, z, t)
    
    image = (-1)**np.floor(5*Z[:,:,0])/2
    
    gradB = -beta * 1j * (Z * (T < gradBoff) - Z * (T > gradBon))
    mod = 1j * alfa * (T-(tmax+gradBon)/2) * (T > gradBon) - 1j * image[:,:,np.newaxis] *  (T > modon) * (T < modoff)
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

# %%
from unwrap import unwrap
E = np.pad(r.E[round(sid.tsteps * sid.gradBon / sid.tmax):, -1, :], ((0, 0), (0, 0)),  'constant', constant_values=(0))
freq = np.fft.fftshift(np.fft.fftfreq(E.shape[0], sid.tmax/sid.tsteps), axes = 0) * 2 * np.pi / sid.beta
Ef = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E, axes = 0), axis = 0), axes = 0)
phase = unwrap(np.angle(Ef))
plt.plot(np.abs(r.E[:,-1,15]))
plt.show()
plt.plot(np.abs(Ef[:,15]))
plt.show()

#%%
plt.imshow(np.abs(Ef))
# %%
