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
    tmax = 5; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 120
    xradious = 0.1; xsteps = 100
    z = np.linspace(-zradious, zradious, num=zsteps)
    x = np.linspace(-xradious, xradious, num=xsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    OmegaC = 2 * np.pi * 4.7 * 8 # MHz

    alfa = 2 * np.pi * 0.04 / 5 # MHz / us
    beta = 2 * np.pi * 1.7 * 8 # MHz / cm

    mot_x = xradious / 3
    mot_z = zradious * 0.95

    gradBoff = 2
    gradBon = 3
    modon = 2
    modoff = 3
    
    Gamma = 2 * np.pi * 6.066
    Delta = 2 * np.pi * 70
    OD = 100
    
    n_at_z = supergauss(z, w=mot_z, pow=4, dtype=np.double)
    n_at_x = supergauss(x, w=mot_x, pow=2, dtype=np.double)
    n_at = np.outer(n_at_z,n_at_x)
    
    E_init_t=supergauss(t, 1, 0.2, dtype = np.complex128)
    E_init_x=supergauss(x, 0, 0.1, dtype = np.complex128)
    E_init = np.outer(E_init_x,E_init_t)
    
    Cpl_t = supergauss(t, cen = gradBoff/2, w = gradBoff/2, pow = 2, dtype = np.complex128) + supergauss(t, cen = (tmax+gradBon)/2, w = (tmax-gradBon)/2, pow = 4, dtype = np.complex128)
    Cpl_x = supergauss(x, cen = 0, w = 0.1, pow = 4, dtype = np.complex128)
    Cpl = OmegaC * np.outer(Cpl_x, Cpl_t)
    
    rho12_init_x=0j * x
    rho12_init_z=0j * z
    rho12_init = np.outer(rho12_init_z,rho12_init_x)
    
    X, Z, T = np.meshgrid(x, z, t)
    gradB = beta * 1j * (Z * (T < gradBoff) - Z * (T > gradBon))
    mod = 0j
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

# %% rysowanie
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

ax1.set_title(r'$|E|^2$')
ax1.set_xlabel('z [cm]')
ax1.set_ylabel('t [$\mu$s]')
ax1.set_xlim(-1,1)
ax1.set_ylim(0,5)
ax1.pcolormesh(r.z, r.t, np.abs(r.E[:,:,sid.xsteps // 2])**2, shading='auto', cmap='Reds')

ax2.set_title(r'$|\rho_{1,2}|^2$')
ax2.set_xlabel('z [cm]')
ax2.set_xlim(-1,1)
ax2.set_yticks([])
ax2.pcolormesh(r.z, r.t, np.abs(r.rho12[:,:,sid.xsteps // 2])**2, shading='auto', cmap='Reds')

ax3.set_xlabel('x [cm]')
ax3.set_ylabel('t [$\mu$s]')
ax3.set_xlim(-0.1, 0.1)
ax3.set_ylim(0,5)
ax3.pcolormesh(r.x, r.t,np.abs(r.E[:,sid.zsteps // 2,:])**2, shading='auto', cmap='Reds')

ax4.set_xlabel('x [cm]')
ax4.set_yticks([])
ax4.set_xlim(-0.1,0.1)
ax4.pcolormesh(r.x, r.t,np.abs(r.rho12[:,sid.zsteps // 2,:])**2, shading='auto', cmap='Reds')

ax5.set_xlabel('z [cm]')
ax5.set_ylabel('x [cm]')
ax5.set_xlim(-1,1)
ax5.set_ylim(-0.1,0.1)
ax5.pcolormesh(r.z, r.x, np.abs(r.E[60,:,:].T)**2, shading='auto', cmap='Reds')

ax6.set_xlabel('z [cm]')
ax6.set_yticks([])
ax6.set_xlim(-1,1)
ax6.pcolormesh(r.z, r.x, np.abs(r.rho12[60,:,:].T)**2, shading='auto', cmap='Reds')

fig.show()
fig.canvas.draw()

# %%