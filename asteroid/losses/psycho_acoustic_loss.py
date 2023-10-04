import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def psycho_acoustic_loss(ys_pred, ys_true, fs=44100, N=1024, nfilts=64, quality=100):
    """
    ys_pred: [batch_size, channels, N+1, frame]
    ys_true: [batch_size, channels, N+1, frame]
    """
    # Check the number of channels (either 1 for mono or 2 for stereo)
    channels = ys_pred.shape[1]

    if channels not in [1, 2]:
        raise ValueError(
            f"Unsupported number of channels: {channels}, only mono and stereo are supported"
        )

    # Function to compute MSE loss for a single channel
    def compute_channel_loss(y_pred_channel, y_true_channel, use_quantization=True):
        mT_pred, mTbarkquant_pred = compute_masking_threshold(
            y_pred_channel, fs, N, nfilts, quality
        )
        mT_true, mTbarkquant_true = compute_masking_threshold(
            y_true_channel, fs, N, nfilts, quality
        )

        if use_quantization:
            return F.mse_loss(mTbarkquant_pred, mTbarkquant_true)
        else:
            return F.mse_loss(mT_pred, mT_true)

    if channels == 1:
        # Mono audio
        mse_loss = compute_channel_loss(ys_pred, ys_true)
    else:
        # Stereo audio
        mse_left = compute_channel_loss(ys_pred[:, 0, :, :], ys_true[:, 0, :, :])
        mse_right = compute_channel_loss(ys_pred[:, 1, :, :], ys_true[:, 1, :, :])
        mse_loss = (mse_left + mse_right) / 2  # Average loss across channels
    print("psycho acoustic mse loss", mse_loss.item())
    return mse_loss


def get_analysis_params(fs, N, nfilts=64):
    maxfreq = fs / 2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfft = 2 * N  # number of fft subbands

    W = mapping2barkmat(fs, nfilts, nfft)
    spreadingfunctionBarkdB = f_SP_dB(maxfreq, nfilts)
    spreadingfuncmatrix = spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts)

    return W, spreadingfuncmatrix, alpha


def compute_masking_threshold(ys, fs, N, nfilts=64, quality=100):
    W, spreadingfuncmatrix, alpha = get_analysis_params(fs, N, nfilts)
    W = W.to(ys.device)
    ys = ys.squeeze(1)
    M = ys.shape[1]  # number of blocks in the signal

    # Compute mXbark for all frames at once
    mXbark = mapping2bark(torch.abs(ys), W, 2 * N)

    # Compute mTbark for all frames at once
    mTbark = maskingThresholdBark(mXbark, spreadingfuncmatrix, alpha, fs, nfilts) / (
        quality / 100
    )

    # Vectorized operations to avoid loop
    mTbarkquant = torch.round(torch.log2(mTbark) * 4)
    mTbarkquant = torch.clamp(mTbarkquant, min=0)
    mTbarkdequant = torch.pow(2, mTbarkquant / 4).to(mXbark.device)

    # Compute mT for all frames at once
    W_inv = mappingfrombarkmat(W, 2 * N)
    mT = mappingfrombark(mTbarkdequant, W_inv, 2 * N)

    return mT, mTbarkquant


def compute_STFT(x, N):
    ys = torch.stft(x, n_fft=2 * N, return_complex=True)
    ys = ys * torch.sqrt(torch.tensor(2 * N / 2)) / 2 / 0.375
    return ys


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
    spreadingfuncmatrix = spreadingfuncmatrix.to(mXbark.device)
    mTbark = torch.matmul(mXbark**alpha, spreadingfuncmatrix**alpha)
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
    ).to(mXbark.device)
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
    """
    mX [batch_size, N+1, frame]
    W [M, 2*N] M=64
    """

    nfreqs = int(nfft / 2)

    # Removing the last frequency band and squaring the magnitude
    mX = mX[:, :-1, :] ** 2
    mX_transposed = mX.transpose(-1, -2)  # Shape should now be [batch size, n, 1024]

    # Performing the matrix multiplication and square root operation
    mXbark = (
        torch.matmul(mX_transposed, W[:, :nfreqs].T) ** 0.5
    )  # Shape should be [batch size, n, 64]
    return mXbark


def mappingfrombarkmat(W, nfft):
    nfreqs = int(nfft / 2)
    W_inv = torch.mm(
        torch.diag(1.0 / (torch.sum(W, dim=1) + 1e-6)).sqrt(), W[:, 0 : nfreqs + 1]
    ).T
    return W_inv


def mappingfrombark(mTbark, W_inv, nfft):
    """
    mTbark [batch_size, N, M]
    W_inv [N, M]
    """

    nfreqs = int(nfft / 2)
    mT = torch.matmul(mTbark, W_inv[:, :nfreqs].T)

    return mT  # Keeping shape [batch size, N, N]


def plot_results(ys, fs, N, nfilts=64, quality=100):
    mT, mTbarkquant = compute_masking_threshold(ys, fs, N, nfilts, quality)

    # Convert STFT magnitude to dB for visualization
    ys_dB = 20 * torch.log10(torch.abs(ys) + 1e-6)
    # Convert masking threshold to dB for visualization
    mT_dB = 20 * torch.log10(mTbarkquant + 1e-6)

    print("mt_dB", mT_dB)
    print("ys_dB", ys_dB)

    # Frequency and Time vectors for plotting
    f = np.linspace(0, fs / 2, ys.shape[0])
    t = np.linspace(0, ys.shape[0], ys.shape[1])

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot Spectrogram
    plt.subplot(3, 1, 1)
    plt.pcolormesh(t, f, ys_dB.numpy(), shading="gouraud", vmin=0, vmax=60)
    plt.colorbar(label="dB")
    plt.title("Spectrogram")
    plt.ylabel("Frequency (Hz)")

    # Plot Spectrum and Masking Threshold of Middle Frame
    middle_frame_idx = len(t) // 2
    plt.subplot(3, 1, 2)
    print("ys", ys_dB[:, middle_frame_idx].numpy())
    print("mt", mT_dB[:, middle_frame_idx].numpy())
    plt.plot(
        f, ys_dB[:, middle_frame_idx].numpy(), color="blue", label="Spectrum", alpha=0.7
    )
    plt.plot(
        f,
        mT_dB[:, middle_frame_idx].numpy(),
        color="red",
        label="Masking Threshold",
        alpha=0.7,
    )
    plt.legend()
    plt.title(f"Spectrum and Masking Threshold at t = {t[middle_frame_idx]:.2f} s")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    # Overlay bark scale center frequencies in blue
    W, _, alpha = get_analysis_params(fs, N, nfilts)
    bark_center_freqs = bark2hz(np.linspace(0, hz2bark(fs / 2), W.shape[0]))
    # for freq in bark_center_freqs:
    #     plt.axhline(y=freq, color="blue", linewidth=0.5, alpha=0.7)

    # Plot Spreading Function
    spreadingfunctionBarkdB = f_SP_dB(fs / 2, W.shape[0])
    # print("spreadingfunctionBarkdB", spreadingfunctionBarkdB)
    spreadingfunctionBarkVoltage = 10.0 ** (spreadingfunctionBarkdB / 20.0 * alpha)

    plt.subplot(3, 1, 3)
    plt.plot(spreadingfunctionBarkVoltage.numpy())
    x_length = len(spreadingfunctionBarkVoltage)
    plt.axvline(x=x_length // 2, color="red", linestyle="--")
    plt.axvline(x=x_length - 1, color="red", linestyle="--")
    plt.title("Spreading Function")
    plt.xlabel("Bark Scale")
    plt.ylabel("Amplitude (Voltage)")

    plt.tight_layout()
    plt.savefig("spectrogram_with_masking_threshold.png")
