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
    zradious = 1; zsteps = 128
    z = np.linspace(-zradious, zradious, num=zsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    alfa = 2 * np.pi * 0.04 / 5 # MHz / us
    beta = 2 * np.pi * 1.7 * 8 # MHz / cm
    OmegaRc = 2 * np.pi * 4.7 

    gradBoff = 2
    gradBon = 3
    modon = 2
    modoff = 3

    Gamma = 2 * np.pi * 6.066
    Delta = 2 * np.pi * 70
    OD = 1500

    n_at = supergauss(z, w=0.7, pow=4, dtype=np.double)

    E_init = supergauss(t, 0.5, 0.05, dtype = np.complex128)

    Cpl = OmegaRc * (supergauss(t, cen = gradBoff/2, w = gradBoff/2, pow = 4, dtype = np.complex128) + supergauss(t, cen = (tmax+gradBon)/2, w = (tmax-gradBon)/2, pow = 4, dtype = np.complex128))

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
        CErho = sid.OD * sid.Gamma * 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        CrhoE = 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        SCpl = 0j#-1j / (4 * sid.Delta + 2 * 1j * sid.Gamma)
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
    return RunParams()
rpars=runParams()
sid.save(rpars.Atimestamp)
sim.run(rpars.toDict(True), 'gem1d')

#%% wczytanie wyników
sid, r = xmds.load_output(-1, 'gem1d')

#%%
class SimInputData(xmds.SimInputData):
    z, t = compileParams().grid_zt()
    path = xmds.init_path()

    alfa = 2 * np.pi * 0.04 / 5 # MHz / us
    beta = 200 # MHz / cm
    OmegaRc = 4

    gradBoff = 2
    gradBon = 3
    modon = 2
    modoff = 3

    Gamma = 0.1
    Delta = 0.2
    OD = 10

    n_at = supergauss(z, w=0.7, pow=4, dtype=np.double)
    #n_at = supergauss(z, cen = -0.3, w=0.5, pow=4, dtype=np.double) + supergauss(z, cen = 0.3, w=0.5, pow=4, dtype=np.double)
    n_at.tofile(str(path / 'n_at.bin'))

    #E_init = 0j * t #supergauss(t, 1, 0.1, dtype = np.complex128) #+ supergauss(t, 2, 0.1, dtype = np.complex128) + supergauss(t, 3, 0.1, dtype = np.complex128)  #* np.exp(+(t - 1.25)* 3j)
             # + 0j * supergauss(t, 1.25, 0.01) * np.exp(-(t - 1.25) * 3j))
    E_init = supergauss(t, 0.5, 0.4, dtype = np.complex128)
    E_init.tofile(str(path / 'E0_t_init.bin'))

    Cpl_t = OmegaRc * (supergauss(t, cen = gradBoff/2, w = gradBoff/2, pow = 4, dtype = np.complex128) + supergauss(t, cen = (g.tmax+gradBon)/2, w = (g.tmax-gradBon)/2, pow = 4, dtype = np.complex128))
    #Cpl_t = 1 * (supergauss(t, cen = 1.25, w = 0.5, pow = 4, dtype = np.complex128) * np.exp(+(t - 1.25) ** 2 * 1j) * (t < 2)
    #         + supergauss(t, cen = 3.75, w = 0.5, pow = 4, dtype = np.complex128) * np.exp(-(t - 3.75) ** 2 * 1j) * (t > 3))
    #Cpl_t = OmegaRc * np.ones_like(t) + 0j
    Cpl_t.tofile(str(path / 'Cpl_t_init.bin'))

    Z, T = np.meshgrid(z, t)

    gradB = beta * 1j * (Z * (T < gradBoff) - Z * (T > gradBon))
    #gradB = beta * 1j * Z

    mod = 0j#- 1j * beta ** 2 / (2 * alfa) * Z ** 2 / 10 / 16 * (T > modon) * (T < modoff) + 1j * (alfa * (T-1)) * (T < 2)
    Mod_zt = gradB + mod
    Mod_zt.transpose().tofile(str(path / 'Mod_zt.bin'))

    #rho12_init = supergauss(z, 0, 0.25, dtype = np.complex128) * np.exp(-1j * beta * z * 2)
    rho12_init = 0j * z
    rho12_init.tofile(str(path / 'rho12_init.bin'))


sid = SimInputData()

# %% uruchomienie symulacji
def runParams():
    class RunParams(xmds.ParamsClass):
        CErho = sid.OD * sid.Gamma * 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        CrhoE = 1j / (4 * sid.Delta - 2 * 1j * sid.Gamma)
        SCpl = 0j#-1j / (4 * sid.Delta + 2 * 1j * sid.Gamma)
    return RunParams()
rpars=runParams()
r2=sim.run_load(rpars, 'gem1d')


#%% rysowanie
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.set_title(r'$|\Omega_S|^2$')
ax1.set_xticks([])
ax1.set_ylabel('t [$\mu$s]')
ax1.set_ylim(0,5)
ax1.pcolormesh(r.z, r.t, np.abs(r.E)**2, shading='auto', cmap='Reds')

ax2.set_title(r'$|\rho_{g,h}|^2$')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.pcolormesh(r.z, r.t,np.abs(r.rho12)**2, shading='auto', cmap='Reds')

ax3.set_title(r'$|\widetilde{\rho}_{g,h}|^2$')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.pcolormesh(r.kz, r.t,np.abs(r.rho12kz)**2, shading='auto', cmap='Reds')

ax4.set_xlabel('z [cm]')
ax4.set_ylabel('t [$\mu$s]')
ax4.set_xlim(-1,1)
ax4.set_ylim(0,5)
ax4.pcolormesh(r.z, r.t, np.abs(r2.E)**2, shading='auto', cmap='Reds')

ax5.set_xlabel('z [cm]')
ax5.set_xlim(-1,1)
ax5.set_yticks([])
ax5.pcolormesh(r.z, r.t,np.abs(r2.rho12)**2, shading='auto', cmap='Reds')

ax6.set_xlabel(r'kz [$\frac{1}{cm}$]')
ax6.set_yticks([])
ax6.pcolormesh(r.kz, r.t,np.abs(r2.rho12kz)**2, shading='auto', cmap='Reds')

fig.show()
fig.canvas.draw()




#%%
fig = plt.figure("E")
fig.clear()
plt.title(r'|E|')
plt.xlabel(r'z [cm]')
plt.ylabel(r't [$\mu$s]')
plt.pcolormesh(sid.z, sid.t[2:], np.abs(r.E), shading = 'auto')
fig.show()
fig = plt.figure("rho12")
fig.clear()
plt.title(r'|S|')
plt.xlabel(r'z [cm]')
plt.ylabel(r't [$\mu$s]')
plt.pcolormesh(sid.z, sid.t[2:], np.abs(r.rho12), shading = 'auto')
fig.show()
fig = plt.figure("rho12kz")
fig.clear()
plt.title(r'|$\mathcal{F}$(S)|')
plt.xlabel(r'kz [$\frac{1}{cm}$]')
plt.ylabel(r't [$\mu$s]')
plt.pcolormesh(np.fft.fftfreq(len(sid.z)) * 2 * np.pi, sid.t[2:], np.abs(np.fft.fft(r.rho12)), shading = 'auto')
fig.show()

#%%
fig = plt.figure("rho12out")
fig.clear()
plt.title(r'|S(t=$t_{max}$)|')
plt.xlabel(r'z [cm]')
plt.ylabel(r't [$\mu$s]')
plt.plot(sid.z, np.abs(r.rho12[-1]))
plt.show()

#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(3, 2)
plt.subplots_adjust(wspace=0, hspace=0)

ax1.set_xticks([])
ax1.pcolormesh(r.z, r.t, np.abs(r.E), shading='auto', cmap='Reds')

ax2.set_xticks([])
ax2.set_ylabel(r'$kz [\frac{2 \pi}{mm}]$')
ax2.pcolormesh(r.z, r.t, np.abs(r.E), shading='auto')

ax3.set_xticks([])

ax3.pcolormesh(r.z, r.t, np.abs(r.E), shading='auto')

ax4.set_ylabel(r'$kz [\frac{2 \pi}{mm}]$')
ax4.set_xlabel('t [$\mu$s]')
ax4.pcolormesh(r.z, r.t, np.abs(r.E), shading='auto')

fig.show()
fig.canvas.draw()
#%%