import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# from MDCTfb import *


def psycho_acoustic_loss(y_pred, y_true, fs=44100, N=1024, nfilts=64, quality=100):
    mT_pred, mTbarkquant_pred = compute_masking_threshold(
        y_pred.squeeze(), fs, N, nfilts, quality
    )
    mT_true, mTbarkquant_true = compute_masking_threshold(
        y_true.squeeze(), fs, N, nfilts, quality
    )

    mse = F.mse_loss(mTbarkquant_pred, mTbarkquant_true)
    return mse


def compute_masking_threshold(x, fs, N, nfilts=64, quality=100):
    maxfreq = fs / 2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfft = 2 * N  # number of fft subbands

    W = mapping2barkmat(fs, nfilts, nfft)
    W_inv = mappingfrombarkmat(W, nfft)
    spreadingfunctionBarkdB = f_SP_dB(maxfreq, nfilts)
    spreadingfuncmatrix = spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts)

    # Compute STFT
    ys = torch.stft(x, n_fft=2 * N, return_complex=True)
    ys = ys * torch.sqrt(torch.tensor(2 * N / 2)) / 2 / 0.375

    M = ys.shape[1]  # number of blocks in the signal
    mT = torch.zeros((N + 1, M))
    mTbarkquant = torch.zeros((nfilts, M))

    # print("ys", ys.shape)
    # print("torch.abs(ys)", torch.abs(ys).shape)
    # print("torch.abs(ys[:, m])", torch.abs(ys[:, 0]).shape)
    # print("W", W.shape)

    for m in range(M):  # M: number of blocks
        mXbark = mapping2bark(torch.abs(ys[:, m]), W, nfft)
        mTbark = maskingThresholdBark(
            mXbark, spreadingfuncmatrix, alpha, fs, nfilts
        ) / (quality / 100)

        mTbarkquant[:, m] = torch.round(torch.log2(mTbark) * 4)
        mTbarkquant[:, m].clamp_(min=0)

        mTbarkdequant = torch.pow(2, mTbarkquant[:, m] / 4)
        mT[:, m] = mappingfrombark(mTbarkdequant, W_inv, nfft)

    return mT, mTbarkquant


def gaussian_spreading_function(nfilts, sigma):
    x = np.linspace(-nfilts, nfilts, 2 * nfilts)
    spreadingfunctionBark = np.exp(-(x**2) / (2 * sigma**2))
    spreadingfunctionBarkdB = 20 * np.log10(spreadingfunctionBark + np.finfo(float).eps)
    return spreadingfunctionBarkdB


def hyperbolic_spreading_function(nfilts, a, b):
    x = np.linspace(-nfilts, nfilts, 2 * nfilts)
    spreadingfunctionBark = 1 / (1 + np.exp(-a * (x - b)))
    spreadingfunctionBarkdB = 20 * np.log10(spreadingfunctionBark + np.finfo(float).eps)
    return spreadingfunctionBarkdB


def f_SP_dB(maxfreq, nfilts):
    maxbark = hz2bark(maxfreq)
    spreadingfunctionBarkdB = torch.zeros(2 * nfilts)
    spreadingfunctionBarkdB[0:nfilts] = torch.linspace(-maxbark * 27, -8, nfilts) - 23.5
    spreadingfunctionBarkdB[nfilts : 2 * nfilts] = (
        torch.linspace(0, -maxbark * 12.0, nfilts) - 23.5
    )
    return spreadingfunctionBarkdB


def spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts):
    spreadingfunctionBarkVoltage = 10.0 ** (spreadingfunctionBarkdB / 20.0 * alpha)
    spreadingfuncmatrix = torch.zeros(nfilts, nfilts)
    for k in range(nfilts):
        spreadingfuncmatrix[k, :] = spreadingfunctionBarkVoltage[
            (nfilts - k) : (2 * nfilts - k)
        ]
    return spreadingfuncmatrix


def maskingThresholdBark(mXbark, spreadingfuncmatrix, alpha, fs, nfilts):
    mTbark = torch.mm(mXbark**alpha, spreadingfuncmatrix**alpha)
    mTbark = mTbark ** (1.0 / alpha)

    maxfreq = fs / 2.0
    maxbark = hz2bark(maxfreq)
    step_bark = maxbark / (nfilts - 1)
    barks = torch.arange(0, nfilts) * step_bark
    f = bark2hz(barks) + 1e-6

    LTQ = torch.clip(
        (
            3.64 * (f / 1000.0) ** -0.8
            - 6.5 * torch.exp(-0.6 * (f / 1000.0 - 3.3) ** 2.0)
            + 1e-3 * ((f / 1000.0) ** 4.0)
        ),
        -20,
        120,
    )
    mTbark = torch.max(mTbark, 10.0 ** ((LTQ - 60) / 20))
    return mTbark


def hz2bark(f):
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f)
    Brk = 6.0 * torch.arcsinh(f / 600.0)
    return Brk


def bark2hz(Brk):
    if not isinstance(Brk, torch.Tensor):
        Brk = torch.tensor(Brk)
    Fhz = 600.0 * torch.sinh(Brk / 6.0)
    return Fhz


def mapping2barkmat(fs, nfilts, nfft):
    maxbark = hz2bark(fs / 2)
    nfreqs = nfft // 2
    step_bark = maxbark / (nfilts - 1)
    binbark = hz2bark(torch.linspace(0, (nfft / 2), int(nfft / 2) + 1) * fs / nfft)

    W = torch.zeros(nfilts, nfft)
    for i in range(nfilts):
        W[i, 0 : int(nfft / 2) + 1] = (torch.round(binbark / step_bark) == i).float()
    return W


def mapping2bark(mX, W, nfft):
    nfreqs = int(nfft / 2)
    mX = mX[:-1].unsqueeze(0)
    mXbark = (torch.mm((mX[:nfreqs].abs()) ** 2.0, W[:, :nfreqs].T)) ** 0.5
    return mXbark


def mappingfrombarkmat(W, nfft):
    nfreqs = int(nfft / 2)
    W_inv = torch.mm(
        torch.diag(1.0 / (torch.sum(W, dim=1) + 1e-6)).sqrt(), W[:, 0 : nfreqs + 1]
    ).T
    return W_inv


def mappingfrombark(mTbark, W_inv, nfft):
    mTbark = mTbark.unsqueeze(0)
    nfreqs = int(nfft / 2)
    mT = torch.mm(mTbark, W_inv[:, :nfreqs].T)
    return mT
