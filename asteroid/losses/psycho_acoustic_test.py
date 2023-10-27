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
# shape: [batch size, channels, N+1, frame]

# Concatenating all ys_pred inputs along a new batch dimension
ys_pred = torch.cat(
    [ys_noisy, ys_true, ys_random1, ys_random2], dim=0
)  # Shape: [4, ...other dims...]
ys_pred = ys_noisy
# print(ys_pred.shape)

# Replicating ys_true to have the same batch size as ys_pred
ys_true_batch = ys_true.repeat(1, 1, 1, 1)  # Shape: [4, ...other dims...]
# print(ys_true_batch.shape)
# Compute loss for all batch entries at once
loss = psycho_acoustic_loss(ys_pred, ys_true_batch, fs=sample_rate)

# Printing individual loss values
# print(f"Psychoacoustic Loss (with noise): {loss[0].item()}")
# print(f"Psychoacoustic Loss (self): {loss[1].item()}")
# print(f"Psychoacoustic Loss (random noise 1): {loss[2].item()}")
# print(f"Psychoacoustic Loss (random noise 2): {loss[3].item()}")
print(f"Psychoacoustic Loss: {loss.item()}")
# y_true = y_true.unsqueeze(0)

N = 1024
ys_true = compute_STFT(y_true, N)
ys_true = ys_true.unsqueeze(0).unsqueeze(0)
print(ys_true.shape)
plot_results(ys_true, sample_rate, N)
