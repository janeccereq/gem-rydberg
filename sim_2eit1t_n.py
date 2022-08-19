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
from xmdswrapper import xmds
#%%
class compileParams(xmds.ParamsClass):
    'parametry podawane przed kompilacją'
    tsteps = 100
    tsamples = 100
    #tmax = 2
    #zradious = 1; zsteps = 128
sim = xmds.Simulation().write_xmds('2eit1tnt.tmpl.xmds', compileParams().toDict())
sim.compile()
# %% przygotowanie pól wejściowych - przypadek równoległy
import os
from xmdswrapper.raphys import supergauss, square, ramparam, rect, rect_sum, save_txt
from xmdswrapper.raplots import plot_zt, reimplot, in1dfigures, save_all
from scipy import signal
from matplotlib import pyplot as plt

savepath = str(xmds.root_path/'kz0/par')
class SimInputData(xmds.SimInputData):
    tmax = 2; tsteps = compileParams.tsteps
    zradious = 1; zsteps = 128
    z = np.linspace(-zradious, zradious, num=zsteps)
    t = np.linspace(0, tmax * (1 + 2 / tsteps), num=tsteps + 2)

    path = xmds.init_path()

    n0 = 50e3
    Cpl0 = 100
    lightv = -1.4
    grv0 = -Cpl0 ** 2 * lightv / (Cpl0 ** 2 + n0)

    #n_at = n0*np.ones_like(z)
    #n_at.tofile(str(path / 'n_at.bin'))
    n_at = n0 * 1/(1+np.exp(-7*(t-1))) + 0j * (t > 0.5)
    n_at.tofile(str(path / 'n_at_t_init.bin'))

    E_init = supergauss(z, 0, 0.5)
    EE_init = np.outer(E_init, E_init)
    EE_init.tofile(str(path / 'EE_init.bin'))

    Cpl_t = Cpl0 * np.ones_like(t) + 0j * t
    Cpl_t.tofile(str(path / 'Cpl_t_init.bin'))

    grv = -Cpl_t ** 2 * lightv / (Cpl_t ** 2 + n_at)
    grv.tofile(str(path / 'grv_t_init.bin'))

    Z, T = np.meshgrid(z, t)

    Mod_zt = 0j * (-0.3<Z)*(Z<0.3) * (0.9 < T)*(T < 1.1)
    Mod_zt.transpose().tofile(str(path / 'Mod_zt.bin'))

    rho12_init = -E_init / Cpl0
    SS_init = np.outer(rho12_init, rho12_init)
    SS_init.tofile(str(path / 'SS_init.bin'))

    pol_init = -1j * (grv0 / 2 / n0) ** 0.5 * np.diff(E_init, n = 1, append = 0)
    PP_init = np.outer(pol_init, pol_init)
    PP_init.tofile(str(path / 'PP_init.bin'))

    EP_init = np.outer(E_init, pol_init)
    EP_init.tofile(str(path / 'EP_init.bin'))
    ES_init = np.outer(E_init, rho12_init)
    ES_init.tofile(str(path / 'ES_init.bin'))
    PS_init = np.outer(pol_init, rho12_init)
    PS_init.tofile(str(path / 'PS_init.bin'))


sid = SimInputData()
#% %
fig = plt.figure("sid E(z)")
fig.clear();
plt.plot(sid.z, np.abs(sid.E_init))
#fig.show()
#fig.canvas.draw()
fig = plt.figure("sid n(t) Cpl(t)")
fig.clear()
plt.plot(sid.t, sid.n_at/ np.max(sid.n_at))
plt.plot(sid.t, np.abs(sid.Cpl_t))
#fig.show()
#fig.canvas.draw()
# fig = plt.figure("sid Mod(z,t)")
# plt.imshow(np.imag(sid.Mod_zt),aspect='auto', origin='lower', interpolation='none')
#% %
fig = plt.figure("|EE(z,z)|")
plt.imshow(np.abs(sid.EE_init),aspect='auto', origin='lower', interpolation='none')
#% % uruchomienie symulacji
#%%
def runParams():
    class RunParams(xmds.ParamsClass):
        GammaP = 100 + 0j
        lightv = sid.lightv
        d6 = 0.1
        VR = 1 + 0j
        tmax = sid.tmax
        zradious = sid.zradious
        zsteps = sid.zsteps
    return RunParams()
rpars=runParams()
sim.run(rpars.toDict(True), '2eit1t')
#r=sim.run_load(rpars, '2eit1t')
#%%
opts = dict(extent=(r.z1[0],r.z1[-1], r.t[0],r.t[-1]), aspect='auto', origin='lower', interpolation='none')
fig = plt.figure("|EE(z,z)|")
plt.imshow(np.abs(r.EE[:,range(64),range(64)]), **opts)

fig = plt.figure("|SS(z,z)|")
plt.imshow(np.abs(r.SS[:,range(64),range(64)]), **opts)

#%%
r = xmds.load_sim(-1)

import pyqtgraph as pg
import numpy as np
import importlib
from ogladnew import Oglad
# if 'oglad' in dir():
#     # jeśli okienko już otware - przeładuj
#     import ogladnew as oglad_module
#     importlib.reload(oglad_module)
#     if oglad.__class__!=Oglad:
#         print('replacing class',oglad.__class__,Oglad)
#         oglad.__class__ = Oglad
#     else:
#         print('equal classes',oglad.__class__, Oglad)
# else:
#     # jeśli okianka jeszcze nie było - skonstruuj
#     oglad=Oglad()
oglad = Oglad()
# w kązdym przypadku inicjalizuj interfejs
oglad.init_win().set_result(r)
oglad.win.move(4000,400) # wyślij okienko zawsze w takie samo miejsce na ekranie
oglad.win.activateWindow()

#%%
from slicer import Oglad
oglad = Oglad()
# w kązdym przypadku inicjalizuj interfejs
oglad.init_win().set_result(r)
oglad.win.move(4000,400) # wyślij okienko zawsze w takie samo miejsce na ekranie
oglad.win.activateWindow()

#%%
r=sim.load(rpars)
#%%
fig = plt.figure("|EE(z,z)|")
plt.imshow(np.abs(r.PP[2,:,:]),aspect='auto', origin='lower', interpolation='none')
#%%
fig = plt.figure("|EE(z,z)|")
plt.imshow(np.abs(r.EE[:,range(64),range(64)]),aspect='auto', origin='lower', interpolation='none')
 #%%
fig = plt.figure("|SS(z,z)|")
plt.imshow(np.abs(r.SS[75,:,:]),aspect='auto', origin='lower', interpolation='none')
np.max(np.abs(r.SS[1,:,:]))
#%%
fig = plt.figure("|SS(z,z)|")
plt.imshow(np.abs(r.SS[:,range(64),range(64)]),aspect='auto', origin='lower', interpolation='none')

#%%
fig = plt.figure("|EE(z,z)|")
n = r.EE.shape[1]
E2 = np.zeros(n, dtype = complex)
for i in range(n):
    E2[i] = r.EE[-1, i, n-1-i]
plt.plot(np.abs(E2))

#%%
r.SS[0,:,:]
#%%
plot_zt(r,'abs S', savepath)
plot_zt(r,'abs P', savepath)
plot_zt(r,'abs E', savepath)
#%%
reimplot(r.t, r.E[:,0]*np.exp(-0.0j*r.t), 'abs-angle-E-in', savepath)
reimplot(r.t, r.E[:,-1]*np.exp(-0.0j*r.t), 'abs-angle-E-final', savepath)
#plot_zt(r,'abs Skz', savepath,'kz t')
#%%
np.argmin(abs(sid.z))

#%%

#%%