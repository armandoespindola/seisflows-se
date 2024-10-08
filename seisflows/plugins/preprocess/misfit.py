#!/usr/bin/env python3
"""
Misfit functions used by the 'default' preprocess class use to quantify 
differences between data and synthetics. 

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


def waveform(syn, obs, nt, dt, *args, **kwargs):
    """
    Direct waveform differencing

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    wrsd = syn - obs

    return np.sqrt(np.sum(wrsd * wrsd * dt))


def envelope(syn, obs, nt, dt, *args, **kwargs):
    """
    Waveform envelope difference from Yuan et al. 2015 Eq. 9

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

    # Residual of envelopes
    env_rsd = env_syn - env_obs

    return np.sqrt(np.sum(env_rsd * env_rsd * dt))


def instantaneous_phase(syn, obs, nt, dt, *args, **kwargs):
    """
    Instantaneous phase difference from Bozdag et al. 2011

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

    return np.sqrt(np.sum(phi_rsd * phi_rsd * dt))


def traveltime(syn, obs, nt, dt, *args, **kwargs):
    """
    Cross-correlation traveltime 

    :type syn: np.array
    :param syn: synthetic data array
    :type obs: np.array
    :param obs: observed data array
    :type nt: int
    :param nt: number of time steps in the data array
    :type dt: float
    :param dt: time step in sec
    """
    cc = abs(np.convolve(obs, np.flipud(syn)))

    return (np.argmax(cc) - nt + 1) * dt


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
    it = np.argmax(syn)
    jt = np.argmax(obs)

    return (jt - it) * dt


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
    ioff = (np.argmax(cc) - nt + 1) * dt

    if ioff <= 0:
        wrsd = syn[ioff:] - obs[:-ioff]
    else:
        wrsd = syn[:-ioff] - obs[ioff:]

    return np.sqrt(np.sum(wrsd * wrsd * dt))


def envelope2(syn, obs, nt, dt, *args, **kwargs):
    """
    Envelope amplitude ratio from Yuan et al. 2015 Eq. B-1

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

    raise NotImplementedError


def envelope3(syn, obs, nt, dt, eps=0., *args, **kwargs):
    """
    Envelope cross-correlation lag from Yuan et al. 2015, Eq. B-4

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

    return Traveltime(env_syn, env_obs, nt, dt)


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
    env_syn = abs(analytic(syn))
    env_obs = abs(analytic(obs))

    env_syn1 = env_syn + eps * max(env_syn)
    env_obs1 = env_obs + eps * max(env_obs)

    diff = (syn / env_syn1) - (obs / env_obs1)

    return np.sqrt(np.sum(diff * diff * dt))



def se_waveform(syn,obs):
    residual = syn - obs
    misfit = np.sum(np.multiply(residual,np.conj(residual)))
    return np.sqrt(np.real(misfit))

def se_phase(syn,obs):
    # Exponential Phase Misfit
    # ratio = syn / obs
    ratio = np.divide(syn, obs, out=np.zeros_like(syn), where=np.abs(obs)!=0)
    angle_unwrap = np.unwrap(np.angle(ratio))
    residual = np.sqrt(2) * np.sin(0.50 * angle_unwrap)
    misfit = np.sqrt(np.sum(np.multiply(residual,residual)))
    return misfit


def se_amplitude(syn,obs):
    # Exponential Phase Misfit
    # ratio = syn / obs
    amp_syn = np.abs(syn)
    amp_obs = np.abs(obs)
    
    ratio = np.divide(amp_syn, amp_obs, out=np.ones_like(amp_syn), where=amp_obs!=0)
    residual = np.log(ratio)
    misfit = 0.5 * sum(np.multiply(residual,residual))
    return misfit



def se_amp_phase(syn,obs):
    amp_misfit = se_amplitude(syn,obs)
    phase_misfit = se_phase(syn,obs)
    misfit = amp_misfit + phase_misfit
    return misfit

    
    
def displacement(*args, **kwargs):
    return Exception("This function can only used for migration.")


def velocity(*args, **kwargs):
    return Exception("This function can only used for migration.")


def acceleration(*args, **kwargs):
    return Exception("This function can only used for migration.")

