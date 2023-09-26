import torchaudio
import torch
from asteroid.losses.psycho_acoustic_loss import psycho_acoustic_loss, plot_results

waveform, sample_rate = torchaudio.load("test48khz.wav")
y_true = waveform[0]

noise = 0.01 * torch.randn(y_true.size())
y_noisy = y_true + noise

loss_noisy = psycho_acoustic_loss(y_noisy, y_true, fs=sample_rate)
print(f"Psychoacoustic Loss (with noise): {loss_noisy.item()}")

y_random1 = torch.randn(y_true.size())
y_random2 = torch.randn(y_true.size())

# loss_self = psycho_acoustic_loss(y_true, y_true, fs=sample_rate)
# loss_random1 = psycho_acoustic_loss(y_random1, y_true, fs=sample_rate)
# loss_random2 = psycho_acoustic_loss(y_random2, y_true, fs=sample_rate)

# print(f"Psychoacoustic Loss (self): {loss_self.item()}")
# print(f"Psychoacoustic Loss (random noise 1): {loss_random1.item()}")
# print(f"Psychoacoustic Loss (random noise 2): {loss_random2.item()}")

plot_results(y_true, sample_rate, 1024)
