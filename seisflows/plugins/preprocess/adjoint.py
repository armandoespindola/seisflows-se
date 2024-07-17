#!/usr/bin/env python3
"""
Adjoints used by the 'default' preprocess class use to generate adjoint sources

All functions defined have four required positional arguments

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
"""
import numpy as np
from scipy.signal import hilbert as analytic

from seisflows.tools.math import hilbert
from seisflows.plugins.preprocess import misfit


def waveform(syn, obs, *args, **kwargs):
    """
    Waveform difference from Tromp et al 2005 Eq 9

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    """
    
    wadj = syn - obs

    #import matplotlib.pyplot as plt
    #plt.plot(obs,'k')
    #plt.plot(syn,'r')
    #plt.show()

    return wadj



def se_waveform(syn, obs, se_t, se_td, se_tse,
                se_dt, nt_se,freq, freq_idx,
                rdi, fft_stf,gamma,t0_array,Wp):
    """
    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    """
    import matplotlib.pyplot as plt
    from scipy.fft import fft,fftfreq,ifft
    residual = syn - obs
    #residual *= Wp
    #residual[abs(obs) == 0.0] = 0.0 
    nt = se_t
    fft_wadj = np.zeros(nt_se, dtype=complex)
    omega = 2.0 * np.pi * freq
    residual *= -1j * np.conj(fft_stf) * np.exp(1j * omega * se_td * se_dt) * np.exp(gamma * t0_array) #* t0_array
    fft_wadj[freq_idx] = residual
    fft_wadj[-freq_idx] = np.conj(residual)
    wadj = np.real(ifft(fft_wadj))

    
    wadj = np.tile(wadj, int(np.ceil(nt / nt_se)))[:nt]
    wadj *= np.exp(-1.0 * gamma * (np.arange(len(wadj)) * se_dt))
    ntaper = np.int(0.025 * nt) # 5% taper
    wadj[-ntaper:] *= np.hanning(2 * ntaper)[ntaper:]
    
    # plt.figure()
    # plt.plot(wadj,'b-')
    # plt.show()

    return wadj



# Exponential phase adjoint source
def se_phase(syn, obs, se_t, se_td, se_tse,
                se_dt, nt_se,freq, freq_idx,
                rdi, fft_stf,gamma,t0_array,Wp):
    """
    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    """
    import matplotlib.pyplot as plt
    from scipy.fft import fft,fftfreq,ifft

    ratio = np.divide(syn, obs, out=np.zeros_like(syn), where=np.abs(obs)!=0)
    #plt.figure()
    #plt.plot(np.angle(ratio),'g')
    #plt.show()
    ratio *= Wp

#    ratio[np.angle(ratio) > np.pi / 2.0] = 0.0
    residual = np.sin(np.angle(ratio)) 
    nt = se_t
    fft_wadj = np.zeros(nt_se, dtype=complex)
    omega = 2.0 * np.pi * freq

    amp_syn = np.abs(syn)
    #amp_syn[amp_syn < np.max(amp_syn) * 1e-2] = 0.0 
    phase = np.angle(syn)

    residual = residual *  np.conj(fft_stf) * syn 
    residual = np.divide(residual, amp_syn**2.0, out=np.zeros_like(residual), where=amp_syn!=0)

    residual *= np.exp(1j * omega * se_td * se_dt) * np.exp(gamma * t0_array) #* t0_array
    fft_wadj[freq_idx] = residual
    fft_wadj[-freq_idx] = np.conj(residual)
    wadj = np.real(ifft(fft_wadj))
    #wadj[:] = 1.0     
    wadj = np.tile(wadj, int(np.ceil(nt / nt_se)))[:nt]
    #wadj *= np.exp(-1.0 * gamma * (np.arange(len(wadj)) * se_dt + se_td * se_dt))
    wadj *= np.exp(-1.0 * gamma * (np.arange(len(wadj)) * se_dt)) # + se_td * se_dt))
    ntaper = np.int(0.025 * nt) # 5% taper
    wadj[-ntaper:] *= np.hanning(2 * ntaper)[ntaper:]

    # plt.figure()
    # plt.plot(np.abs(residual),'b')
    # plt.figure()
    # plt.plot(Wp,'r')
    # plt.show()

    
    # plt.figure()
    # plt.plot(wadj,'b')
    # plt.show()

    return wadj






def envelope(syn, obs, nt, dt, eps=0.05, *args, **kwargs):
    """
    Waveform envelope difference from Yuan et al. 2015 Eq. 16

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    env_syn = abs(analytic(syn))
    env_obs = abs(analytic(obs))

    env_tmp = (env_syn - env_obs) / (env_syn + eps * env_syn.max())

    wadj = env_tmp * syn - np.imag(analytic(env_tmp * np.imag(analytic(syn))))

    return wadj


def instantaneous_phase(syn, obs, nt, dt, eps=0.05, *args, **kwargs):
    """
    Instantaneous phase difference from Bozdag et al. 2011 Eq. 27

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    r = np.real(analytic(syn))
    i = np.imag(analytic(syn))
    phi_syn = np.arctan2(i, r)

    r = np.real(analytic(obs))
    i = np.imag(analytic(obs))
    phi_obs = np.arctan2(i, r)

    phi_rsd = phi_syn - phi_obs
    env_syn = abs(analytic(syn))
    env_max = max(env_syn**2.)

    wadj_1 = phi_rsd * np.imag(analytic(syn)) / (env_syn ** 2. + eps * env_max) 
    wadj_2 = np.imag(analytic(phi_rsd * syn / (env_syn**2. + eps * env_max)))
 
    wadj = wadj_1 + wadj_2

    return wadj


def traveltime(syn, obs, nt, dt, *args, **kwargs):
    """
    Cross-correlation traveltime from Tromp et al. 2005 Eq. 45

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    wadj = np.zeros(nt)

    wadj[1:-1] = (syn[2:] - syn[0:-2]) / (2. * dt)
    wadj *= 1. / (sum(wadj * wadj) * dt)

    wadj *= misfit.traveltime(syn, obs, nt, dt)
    return wadj


def traveltime_inexact(syn, obs, nt, dt, *args, **kwargs):
    """
    A faster cc traveltime function but possibly innacurate

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    wadj = np.zeros(nt)

    wadj[1:-1] = (syn[2:] - syn[0:-2]) / (2. * dt)
    wadj *= 1. / (sum(wadj * wadj) * dt)

    wadj *= misfit.traveltime_inexact(syn, obs, nt, dt)

    return wadj


def amplitude(syn, obs, nt, dt, *args, **kwargs):
    """
    Cross-correlation amplitude difference

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    wadj = 1. / (sum(syn * syn) * dt) * syn
    wadj *= misfit.amplitude(syn, obs, nt, dt)

    return wadj


def envelope2(syn, obs, nt, dt, eps=0., *args, **kwargs):
    """
    Envelope amplitude ratio from Yuan et al. 2015 Eq. B-2

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    raise NotImplementedError


def envelope3(syn, obs, nt, dt, eps=0., *args, **kwargs):
    """
    Envelope lag from Yuan et al. 2015 Eq. B-2, B-5

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    env_syn = abs(analytic(syn))
    env_obs = abs(analytic(obs))

    env_rat = np.zeros(nt)
    env_rat[1:-1] = (env_syn[2:] - env_syn[0:-2]) / (2. * dt)
    env_rat[1:-1] /= env_syn[1:-1]
    env_rat *= misfit.envelope3(syn, obs, nt, dt)

    wadj = -env_rat * syn + hilbert(env_rat * hilbert(env_syn))

    return wadj


def instantaneous_phase2(syn, obs, nt, dt, eps=0., *args, **kwargs):
    """
    Alterative instantaneous phase function

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    esyn = abs(analytic(syn))
    eobs = abs(analytic(obs))

    esyn1 = esyn + eps * max(esyn)
    eobs1 = eobs + eps * max(eobs)
    esyn3 = esyn ** 3 + eps * max(esyn ** 3)

    diff1 = (syn / esyn1) - (obs / eobs1)
    diff2 = (hilbert(syn) / esyn1) - (hilbert(obs) / eobs1)

    part1 = diff1 * hilbert(syn) ** 2 / esyn3 
    part2 = diff2 * syn * hilbert(syn) / esyn3
    part3 = diff1 * syn * hilbert(syn) / esyn3 - diff2 * syn ** 2 / esyn3

    wadj = part1 - part2 + hilbert(part3)

    return wadj


def displacement(syn, obs, nt, dt, *args, **kwargs):
    """
    Displacement waveform for migration
    """
    return obs


def velocity(syn, obs, nt, dt, *args, **kwargs):
    """
    Velocity waveform for migration, taking derivative of obs
    """
    adj = np.zeros(nt)
    adj[1:-1] = (obs[2:] - obs[0:-2]) / (2. * dt)

    return adj


def acceleration(syn, obs, nt, dt, *args, **kwargs):
    """
    Acceleration waveform for migration, second derivative of obs
    Use finite difference to differentiate observatio nwaveform
    """
    adj = np.zeros(nt)
    adj[1:-1] = (-obs[2:] + 2. * obs[1:-1] - obs[0:-2]) / (2. * dt)

    return adj

