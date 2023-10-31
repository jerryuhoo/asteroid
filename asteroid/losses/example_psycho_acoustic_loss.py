import torch
import torchaudio
from asteroid.losses.psycho_acoustic_loss import (
    psycho_acoustic_loss,
    compute_STFT,
)

# Load audio files
audio_wav, sr_wav = torchaudio.load("audio_original.wav")
audio_mp3, sr_mp3 = torchaudio.load("audio_mp3_align.wav")
audio_quantized, sr_wav = torchaudio.load("audio_quantized.wav")

audio_mp3 = audio_mp3[:, : audio_wav.shape[1]]


# MSE Loss (Time Domain)
loss_mse = torch.nn.MSELoss()
mse_mp3_original = loss_mse(audio_mp3, audio_wav)
print("MSE Loss (mp3 and original):", mse_mp3_original)
mse_quant_original = loss_mse(audio_quantized, audio_wav)
print("MSE Loss (quanitzed and original):", mse_quant_original)

print(
    "ratio:, mse_mp3_original / mse_quant_original",
    mse_mp3_original / mse_quant_original,
)

# MSE Loss (Frequency Domain)
N = 2048
audio_wav_stft = compute_STFT(audio_wav, N=N).unsqueeze(0)
audio_mp3_stft = compute_STFT(audio_mp3, N=N).unsqueeze(0)
audio_quantized_stft = compute_STFT(audio_quantized, N=N).unsqueeze(0)

mse_mp3_original_stft = loss_mse(audio_mp3_stft, audio_wav_stft)
print("STFT MSE Loss (mp3 and original):", mse_mp3_original_stft)
mse_quant_original_stft = loss_mse(audio_quantized_stft, audio_wav_stft)
print("STFT MSE Loss (quanitzed and original):", mse_quant_original_stft)
print(
    "ratio:, mse_mp3_original_stft / mse_quant_original_stft",
    mse_mp3_original_stft / mse_quant_original_stft,
)
print(
    "Note: The lower the ratio means the better the loss captures perceptual differences"
)


# Psycho-acoustic Loss
loss_pal_mp3_original = psycho_acoustic_loss(
    audio_mp3_stft,
    audio_wav_stft,
    fs=44100,
    N=N,
)
print("Psycho-acoustic Loss (mp3 and original):", loss_pal_mp3_original)
loss_pal_quant_original = psycho_acoustic_loss(
    audio_quantized_stft,
    audio_wav_stft,
    fs=44100,
    N=N,
)
print("Psycho-acoustic Loss (quanitzed and original):", loss_pal_quant_original)
print(
    "ratio:, loss_pal_mp3_original / loss_pal_quant_original",
    loss_pal_mp3_original / loss_pal_quant_original,
)
