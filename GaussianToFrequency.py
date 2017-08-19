'''
Part of the code is drawn from the Jake VanderPlas (University of Washington's eScience Institute) fft implementation
http://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
'''

# Session One Fast Fourier Transform

import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

L = 20
n = 128

x2 = np.linspace(-L/2, L/2, n+1)
# Spatial domain
x = x2[:n]
u = np.exp(-x**2)
ut = np.fft.fft(u)

#frequency domain
k = [(2 * np.pi / L) * ii for jj in (np.arange(0,n/2), np.arange(-n/2, 0)) for ii in jj]

g = np.cosh(x)**(-1)  # sech function
# Derivative of sech
gd = (-np.cosh(x)**(-1)) * np.tanh(x)
# Second Derivative of sech
g2d = (1./np.cosh(x)) - 2 * (1./np.cosh(x))**3

gt = np.fft.fft(g)
gds = np.fft.ifft(([1j*k[i] for i in range(len(k))])*gt)
g2ds = np.fft.ifft(([1j*k[i]**2 for i in range(len(k))])*gt)
'''
timeit dft_slow(x)
timeit fft(x)
timeit np.fft.fft(x)
'''
plt.figure(1)

plt.subplot(511)
plt.plot(x, u, color='lightblue', linewidth=1)

plt.subplot(512)
plt.plot(x, ut, color='blue', linewidth=1)

plt.subplot(513)
plt.plot(x, np.fft.fftshift(np.abs(ut)), color='green', linewidth=1)
# Spectral content of Gaussian
plt.subplot(514)
plt.plot(np.fft.fftshift(k), np.fft.fftshift(np.abs(ut)), color='green', linewidth=1)

plt.subplot(515)
plt.plot(x, gd, 'r', x, gds, 'g^')

plt.xlim(-L/2, L/2)
plt.show()




'''
Here are implementation of fft that it is worth reading, however, I am not using it in the main code
I prefer use the default numpy fft
'''

'''
x = np.random.random(1024)
np.allclose(fft(x), np.fft.fft(x))
'''
def dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return dft_slow(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])


def fft_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] / 2]
        X_odd = X[:, X.shape[1] / 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()


# Session Two / Filtering

T = 30      # Time domain
n = 512     # Sampling

t = np.linspace(-T/2, T/2, n+1)
time = t[:n]
#t = t2[:n]
# Actual frequency component   Cos0T , Cos1T, Cos2T ...
#freq = [(2 * np.pi / T) * ii for jj in (np.arange(0,n/2), np.arange(-n/2, 0)) for ii in jj]
signal = np.cosh(time)**(-1)  # sech function
freq = np.fft.fftfreq(signal.size, d=time[1]-time[0])
fourier = np.fft.fft(signal)      # Heisenberg Uncertainty Principle is associated with fft

# Adding white noise to the signal
# if we add only real noise it will be symmetric in time domain
# if we add only imaginary noise it will be asymmetric
noise_coefficient = 10
noise = np.random.normal(0, 1, n) + 1j*np.random.normal(0, 1, n)
fourier_noise = fourier + noise_coefficient * noise
signal_noise = np.fft.ifft(fourier_noise)

# Gaussian filter design
#fourier[(freq<0.6)] = 0     high-pass
filter = [np.exp(-freq[ii]**2) for ii in range(len(freq))]
fourier_noise_filter = filter * fourier_noise

signal_noise_filter = np.fft.ifft(fourier_noise_filter)
threshold = [0.5 for ii in range(len(time))]

freq_shift = np.fft.fftshift(freq)
plt.figure(2)

plt.subplot(411)
plt.plot(time, signal, color='lightblue', linewidth=1)
plt.plot(time, signal_noise, color='lightgreen', linewidth=1)

plt.subplot(412)
plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier)), color='blue', linewidth=1)
plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier_noise)), color='green', linewidth=1)

plt.subplot(413)
plt.plot(time, signal_noise_filter, color='black', linewidth=1)
#plt.plot(t, un, color='lightgreen', linewidth=1)
plt.plot(time, threshold, color='red', linewidth=1)

plt.subplot(414)
plt.plot(freq_shift, np.fft.fftshift(np.abs(filter)), color='blue', linewidth=1)
plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier_noise))/np.max(np.fft.fftshift(np.abs(fourier_noise))), color='green', linewidth=1)
plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier_noise_filter))/np.max(np.fft.fftshift(np.abs(fourier_noise_filter))), color='black', linewidth=1)

plt.xlim(-T/2, T/2)
plt.show()


# Session 3 Averaging
#Frequency domain
noise_coefficient = 20
average_vec = np.zeros(n)
average_vec2 = np.zeros(n)
realizations = 30
plt.figure(3)
for i in range(realizations):
    noise = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)
    fourier_noise = fourier + noise_coefficient * noise
    average_vec = average_vec + fourier_noise
    #signal_noise = np.fft.ifft(fourier_noise)
    average_vec2 = np.fft.fftshift(np.abs(average_vec)) / i
    plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier_noise)), color='lightgreen', linewidth=0.5)
    plt.plot(freq_shift, average_vec2, color='blue', linewidth=1)
    plt.pause(0.5)
    plt.clf()
average_vec = np.fft.fftshift(np.abs(average_vec)) /realizations


# Averaging in Time domain

fig = plt.figure(figsize=plt.figaspect(0.5))

slice = np.arange(0, 10, 0.5)
time_mesh, slice_mesh = np.meshgrid(time, slice)
freq_mesh, slice_mesh = np.meshgrid(freq, slice)
signal_mesh= (np.cosh(time_mesh - 10 * np.sin(slice_mesh))**(-1)) * np.exp(1*0*time_mesh)
################
# First subplot
################
ax = fig.add_subplot(2, 1, 1, projection='3d')
surf = ax.plot_surface(time_mesh, slice_mesh, signal_mesh, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

#################
# Second subplot
#################
fourier_mesh = np.zeros(np.shape(signal_mesh))
for i in range(len(slice)):
    fourier_mesh[i, :] = np.fft.fftshift(np.abs(np.fft.fft(signal_mesh[i, :])))

ax = fig.add_subplot(2, 1, 2, projection='3d')
#surf1 = ax.plot_surface(np.fft.fftshift(time_mesh), slice_mesh, fourier_mesh, cmap=cm.coolwarm,
 #                      linewidth=0, antialiased=False)
ax.plot_wireframe(np.fft.fftshift(freq_mesh), slice_mesh, fourier_mesh, rstride=10, cstride=10)
# Customize the z axis.
ax.set_zlim(-1.01, 70.01)
ax.set_ylim(-1.01, 10.01)
ax.set_xlim(-8.01, 8.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

plt.figure(5)
plt.subplot(411)
plt.plot(time, signal, color='lightblue', linewidth=1)
plt.plot(time, signal_noise, color='lightgreen', linewidth=1)

plt.subplot(412)
plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier)), color='blue', linewidth=1)
plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier_noise)), color='green', linewidth=1)

plt.subplot(413)
plt.plot(freq_shift, np.fft.fftshift(np.abs(fourier_noise)), color='green', linewidth=1)
plt.plot(freq_shift, average_vec, color='blue', linewidth=1)

plt.xlim(-T/2, T/2)
plt.show()