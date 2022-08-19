"""
ładowanie i wyświetlanie xmds-a

wykorzystujemy smpl2dtmpl.xmds
    - patrz równania dE_dz oraz drho12_dt
    - dyfrakcja patrz Lxx[E]
    - symulacja odbywa sie w ukłądzie poruszajacym się ze słabym impulsem
        - gdy kEx!=0 układ ten porusza sie w bok i oś z jest nachylona
    - dopasowanie fazowe jest idealne, aby to zepsuć można dopisać człon typu mod*rho12 do drho12_dt

stałe przed kompilacją:
    tstep, tmax
    x/zsteps, x/zradious - siatka od -z do z

parametry:
    CErho - sprzężenie E <- rho
    CrhoE - sprzężenie rho <- E
    Srho - stark od lasera cpl
    kE - długość wektora falowego słabego światła w jednostkach odpowiednich dla jednostek x/z
    kEx - rzut w/w na oś x


początkowo ładujemy z pliku:
    E0_xt - słabe pole padajace z zewnatrz na atomy
    Cpl_t - zależnosć czasową silnego lasera
    Mod_zt - zależność modulacji fazowej (starkowskiej/gem) od z oraz t, drho12/dt=Mod*rho12
    n_at - gęstość chmury atomów od z
    rho12_init - wejściowa fala spinowa od z

interfejs:
    moduł Oglad, show_results

"""
#%% ustaw interfejs
%gui qt5
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import xmdswrapper.xmds2d as xmds
#import dill
#from unwrap import unwrap
from xmdswrapper.raplots2d import plot_rho12, plot_rho12_kz, plot_n_at, plot_E_init, plot_E, plot_Mod, plot_R
from xmdswrapper.utils import fourier, png_load
from xmdswrapper.raphys import supergauss

#%% kompilacja xmds
class compileParams(xmds.ParamsClass):
    'parametry podawane przed kompilacją'
    tsteps=600
    xsteps=256
    zsteps=200
    tmax=100 # mus
    zradious=1 # cm
    xradious=0.045 # cm
g = compileParams()
sim=xmds.Simulation().write_xmds('xmdswrapper/smpl2dtemplmodzx.xmds', g.toDict())
sim.compile()

#%% parametry symulacji
class simInputData:
    Gamma = 2 * np.pi * 6.066
    Delta = 2 * np.pi * 70
    OD = 76
    
    z = np.linspace(-g.zradious, g.zradious, num = g.zsteps)
    x = np.linspace(-g.xradious, g.xradious, num = g.xsteps)
    t = np.linspace(0, g.tmax * (1 + 2 / g.tsteps), num = g.tsteps + 2)
    
    n_at_z = supergauss(z, w = mot_z, pow = 4, dtype = np.double)
    n_at_x = supergauss(x, w = mot_x, pow = 4, dtype = np.double)
    n_at = np.outer(n_at_z, n_at_x)
    
    E_init_t = supergauss(t, 2, 0.3, dtype = np.complex128)
    E_init_x = supergauss(x, 0, 3, dtype = np.complex128)
    E_init = np.outer(OmegaRs * E_init_x, E_init_t)
    
    Cpl_t = supergauss(t, cen = 16, w = 16, pow = 2, dtype = np.complex128) + supergauss(t, cen = 70, w = 30, pow = 4, dtype = np.complex128)
    Cpl_x = supergauss(x, cen = 0, w = 3, pow = 4, dtype = np.complex128)
    Cpl = OmegaRc * np.outer(Cpl_x, Cpl_t)
    
    rho12_init_x = 0j * x
    rho12_init_z = 0j * z
    rho12_init = np.outer(rho12_init_z, rho12_init_x)
    
    X, Z, T = np.meshgrid(x, z, t)
    gradB = beta * 1j * (Z * (T < gradBoff) - Z * (T > gradBon))
    #image = png_load(g, "qot256x200.png")
    mod = -1j * beta ** 2 / (2 * alfa) * Z ** 2 *  (T > modon) * (T < modoff)  #+ 1j * (alfa * (T-16)) * (T < gradBoff) 

    
    def save(self, timestamp):
        with open('xmds_out/smpl2d.Atimestamp_'+timestamp+'.pkl', 'wb') as file:
            dill.dump(self, file)

class runParams(xmds.ParamsClass):
    "parametry przy uruchomieniu"
    def __init__(self, Gamma, Delta, OD):
        xmds.ParamsClass.__init__(self)
        self.kE = round(2*np.pi/0.795e-4, 8)
        self.kEx = round(0, 8)
        self.CErho_re = round(np.real(OD * Gamma * 1j / (4 * Delta - 2 * 1j * Gamma)), 8)
        self.CErho_im = round(np.imag(OD * Gamma * 1j / (4 * Delta - 2 * 1j * Gamma)), 8)
        self.CrhoE_re = round(np.real(1j / (4 * Delta - 2 * 1j * Gamma)), 8)
        self.CrhoE_im = round(np.imag(1j / (4 * Delta - 2 * 1j * Gamma)), 8)
        self.Srho_re = 0#round(np.real(1j / (4 * Delta + 2 * 1j * Gamma)), 8)
        self.Srho_im = 0#round(np.imag(1j / (4 * Delta + 2 * 1j * Gamma)), 8)

#%% uruchomienie symulacji
sid = simInputData()
rP = runParams(sid.Gamma, sid.Delta, sid.OD)
xmds.load_input(sid)
sim.run(rP.toDict(), './smpl2d')
sid.save(rP.Atimestamp)

#%% wczytanie wyników
import matplotlib.pyplot as plt
g, sid, r = xmds.load_output(-1)

#%% prezentacja wyników
import show_results
ex = show_results.Explorer(r)
ex.show()

#%% korekcja dyfrakcji
Ef, phase, freq = fourier(g, sid, r)

fig = plt.figure()
fig.clear()
plt.pcolormesh(sid.x, freq, np.abs(Ef) ** 2)
plt.xlim(-g.xradious,g.xradious)
plt.ylim(-g.zradious,g.zradious)
plt.xlabel('x')
plt.ylabel('z')
fig.show()
fig.canvas.draw()


zmin = int(freq.shape[0] / 2 - freq.shape[0] * 0.3 / (2 * np.max(freq)))
zmax = int(freq.shape[0] / 2 + freq.shape[0] * 0.3 / (2 * np.max(freq)))
xmin = int(sid.x.shape[0] / 2 - sid.mot_x / g.xradious * g.xsteps / 2)
xmax = int(sid.x.shape[0] / 2 + sid.mot_x / g.xradious * g.xsteps / 2)

freq2 = freq[zmin:zmax]
x2 = sid.x[xmin:xmax]
phase2 =  phase[zmin:zmax,xmin:xmax]


fig = plt.figure()
fig.clear()
plt.pcolormesh(x2, freq2, phase2, shading='auto')
plt.xlabel('x')
plt.ylabel('z')
fig.show()
fig.canvas.draw()

Eff = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Ef, axes = 1), axis = 1), axes = 1)
ffreq = np.fft.fftshift(np.fft.fftfreq(Eff.shape[1], 2 * g.xradious / g.xsteps)) * 2 * np.pi
for i in range(Eff.shape[0]):
    Eff[i,:] = Eff[i,:] * np.exp(1j * ffreq ** 2  / 2 / rP.kE * (g.zradious + freq[i]))
Ef= np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Eff, axes = 1), axis = 1), axes = 1)
phase = unwrap(np.angle(Ef))
phase2 =  phase[zmin:zmax,xmin:xmax]


fig = plt.figure()
fig.clear()
plt.pcolormesh(sid.x, freq, np.abs(Ef) ** 2, shading='auto')
plt.xlim(-g.xradious,g.xradious)
plt.ylim(-g.zradious,g.zradious)
plt.xlabel('x')
plt.ylabel('z')
fig.show()
fig.canvas.draw()

fig = plt.figure()
fig.clear()
plt.pcolormesh(x2, freq2, phase2, shading='auto')
plt.xlabel('x')
plt.ylabel('z')
fig.show()
fig.canvas.draw()


#%% korekcja dyfrakcji 2
E = np.pad(r.E[g.tsteps * 4 // 10:, -1, :], ((0, 0), (0, 0)),  'constant', constant_values=(0))
freq = np.fft.fftshift(np.fft.fftfreq(E.shape[0], g.tmax/g.tsteps), axes = 0) * 2 * np.pi / sid.beta
x, y = np.meshgrid(sid.x, freq)
Ef = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E, axes = 0), axis = 0), axes = 0)
phase = unwrap(np.angle(Ef))
'''
fig = plt.figure()
fig.clear()
plt.pcolormesh(x, y, np.abs(Ef) ** 2)
plt.ylim(-1,1)
fig.show()
fig.canvas.draw()
'''

zmin = int(freq.shape[0] / 2 - freq.shape[0] * 0.3 / (2 *np.max(freq)))
zmax = int(freq.shape[0] / 2 + freq.shape[0] * 0.3 / (2 *np.max(freq)))
xmin = int(sid.x.shape[0] / 2 - sid.mot_x / g.xradious * g.xsteps / 2)
xmax = int(sid.x.shape[0] / 2 + sid.mot_x / g.xradious * g.xsteps / 2)

newfreq = np.linspace(0, freq[-1] - freq[0] + g.zradious, freq.shape[0])


freq2 = freq[zmin:zmax]
x2 = sid.x[xmin:xmax]
phase2 =  phase[zmin:zmax,xmin:xmax]# - np.average(phase[freq.shape[0] * 2 // 5:freq.shape[0] * 3 // 5, :])
k0 = (1 - 0 *np.outer(supergauss(freq, w=0.7, pow=4, dtype=np.double), sid.n_at_x)) * runParams.kE
k0f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(1/(2*k0), axes = 1), axis = 1), axes = 1)
Eff = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Ef, axes = 1), axis = 1), axes = 1)
ffreq = np.fft.fftshift(np.fft.fftfreq(Eff.shape[1], 2 * g.xradious / g.xsteps)) * 2 * np.pi
kffreq = k0f
for i in range(kffreq.shape[0]):
    kffreq[i] *= ffreq ** 2
for i in range(Eff.shape[0]):
#    Eff[i,:] = Eff[i,:] * np.exp(1j * ffreq ** 2  / 2 / runParams.kE * (g.zradious + freq[i]))
#    Eff[i,:] = Eff[i,:] * np.exp(1j * ((g.zradious + freq[0]) * 1/(2 * runParams.kE) * ffreq[:] ** 2 + np.trapz(1/(2 * k0[:i,:]) * ffreq[:] ** 2, freq[:i], axis = 0)))
    Eff[i,:] = Eff[i,:] * np.exp(1j * (np.trapz(1/(2 * k0[:i,:]) * ffreq[:] ** 2, freq[:i], axis = 0) - 1.6 * ffreq ** 2  / 2 / runParams.kE * g.zradious))

#zrad 1 -  - 1.6 * ffreq ** 2  / 2 / runParams.kE * g.zradious
#zrad 1.5 - - 1.6 * ffreq ** 2  / 2 / runParams.kE * g.zradious * 0.5
#zrad 1.2 - - 1.6 * ffreq ** 2  / 2 / runParams.kE * g.zradious * 0.5

Ef= np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Eff, axes = 1), axis = 1), axes = 1)
phase = unwrap(np.angle(Ef))
phase2 =  phase[zmin:zmax,xmin:xmax]


fig = plt.figure()
fig.clear()
plt.pcolormesh(x, y, np.abs(Ef) ** 2, shading='auto')
plt.ylim(-g.zradious,g.zradious)
fig.show()
fig.canvas.draw()

fig = plt.figure()
fig.clear()
plt.pcolormesh(x2, freq2, phase2, shading='auto')
fig.show()
fig.canvas.draw()


