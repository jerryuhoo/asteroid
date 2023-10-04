import torchaudio
import torch
from asteroid.losses.psycho_acoustic_loss import (
    psycho_acoustic_loss,
    plot_results,
    compute_STFT,
)

waveform, sample_rate = torchaudio.load("test48khz.wav")
y_true = waveform[0]

noise = 0.01 * torch.randn(y_true.size())
y_noisy = y_true + noise

y_random1 = torch.randn(y_true.size())
y_random2 = torch.randn(y_true.size())

ys_noisy = compute_STFT(y_noisy, N=1024).unsqueeze(0).unsqueeze(0)
ys_true = compute_STFT(y_true, N=1024).unsqueeze(0).unsqueeze(0)
ys_random1 = compute_STFT(y_random1, N=1024).unsqueeze(0).unsqueeze(0)
ys_random2 = compute_STFT(y_random2, N=1024).unsqueeze(0).unsqueeze(0)

print("ys_noisy", ys_noisy.shape)

loss_noisy = psycho_acoustic_loss(ys_noisy, ys_true, fs=sample_rate)
print(f"Psychoacoustic Loss (with noise): {loss_noisy.item()}")


loss_self = psycho_acoustic_loss(ys_true, ys_true, fs=sample_rate)
loss_random1 = psycho_acoustic_loss(ys_random1, ys_true, fs=sample_rate)
loss_random2 = psycho_acoustic_loss(ys_random2, ys_true, fs=sample_rate)

print(f"Psychoacoustic Loss (self): {loss_self.item()}")
print(f"Psychoacoustic Loss (random noise 1): {loss_random1.item()}")
print(f"Psychoacoustic Loss (random noise 2): {loss_random2.item()}")

# y_true = y_true.unsqueeze(0)

# N = 1024
# ys_true = compute_STFT(y_true, N)
# plot_results(ys_true, sample_rate, N)
