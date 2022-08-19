#%%
import numpy as np

r = np.linspace(-1,1,2048)
dr= r[1] - r[0]

kr = np.fft.fftfreq(len(r), dr) * 2 * np.pi

rb = 0.05
Vr = 1/(1-2*1j*(r/rb)**6)
EE0 = np.ones_like(r) + 0j

Rsteps = 2000
dR = 0.0005
ODL = 5 / rb
OmG = 0.2
CVR = ODL
CVr = OmG ** 2 * 4 / ODL
Cr =  4 / ODL

kr2 = -kr * kr
dif = np.exp(-Cr * kr * kr * dR)

dVr = 12*1j*rb**6*r**5/(rb**6-2*1j*r**6)**2
ddVr = 60*1j*r**4/rb**6/(1-2*1j*(r/rb)**6)**2-288*r**10/rb**12/(1-2*1j*(r/rb)**6)**3

loc = np.exp(-(CVR * Vr - CVr * ddVr) * dR)
#%%
EE = [EE0,]
EE3 = EE0
for step in range(Rsteps):
    print(step)
    EEVr = EE3 * Vr
    EEVrf = np.fft.fft(EEVr)
    EEVrf2 = EEVrf * np.exp(CVr * kr2 * dR)
    EE1f = EEVrf2 - EEVrf + np.fft.fft(EE3)
    EE1f *= dif
    EE2 = np.fft.ifft(EE1f)
    EE2Vr = EE2 * dVr
    EE2Vrf = np.fft.fft(EE2Vr)
    EE2Vrf2 = EE2Vrf * np.exp(-2 * 1j * CVr * kr * dR)
    EE2f = EE2Vrf2 - EE2Vrf + EE1f
    EE2 = np.fft.ifft(EE2f)
    EE3 = EE2 * loc
    
    EE.append(EE3)

#%%
R = np.linspace(-dR * Rsteps / 2, dR * Rsteps / 2, Rsteps)
import matplotlib.pyplot as plt
plt.title('$|EE|^2$')
plt.xlabel('r [$mm$]')
plt.ylabel('R [$mm$]')
plt.pcolormesh(r, R, np.abs(EE[1:])**2, cmap='Reds')
plt.show()
#%%
plt.plot(r, np.real(EE[-1]))
# %%
V = np.array(EE)[-1]

V2 = np.ones((len(V) * 2, len(V) * 2), dtype = complex)
for i in range(len(V2)):
    if i <= len(V) // 2:
        V2[i,:len(V) // 2 + i] = V[len(V) // 2 - i:]
    elif i+len(V) // 2 <= len(V2):
        V2[i,i-len(V)//2:i+len(V)//2] = V[:]
    else:
        V2[i, i-len(V)//2:] = V[:len(V2)-i+len(V)//2]
#%%
E2 = np.load('E2.npy')
plt.plot(r, np.abs(EE[600]), label='approx')
z = np.linspace(-2,2,len(E2))
plt.plot(z, E2, label='full')
plt.xlabel('r')
plt.ylabel('|EE|')
plt.legend()
#%%
import scipy
a = scipy.io.mmread('ryd2.mtx')
plt.pcolormesh(np.abs(a.T)**2, cmap='Reds')
r2 = np.linspace(-1,1, a.shape[0])
R2 = np.linspace(-0.5,0.5, a.shape[1])

#%%

fig = plt.figure()
fig.clear()
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title(r'$|EE|^2$ Python')
ax1.set_xlabel('r [$mm$]')
ax1.set_ylabel('R [$mm$]')
ax1.pcolormesh(r, R, np.abs(EE[1:])**2, cmap='Reds')

ax2.set_title(r'$|EE|^2$ Mathematica')
ax2.set_xlabel('r [$mm$]')
ax2.set_yticks([])
ax2.pcolormesh(r2, R2,np.abs(a.T)**2, shading='auto', cmap='Reds')


fig.show()
fig.canvas.draw()
# %%
