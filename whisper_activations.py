import numpy as np
import whisper
import torch
import pickle

phoneme_file = "phoneme_segments.pkl"

with open(phoneme_file, "rb") as f:
    df = pickle.load(f)


WHISPER_SAMPLE_RATE = 16000
WHISPER_INPUT_SAMPLES = 30 * WHISPER_SAMPLE_RATE  # 30 seconds*16k = 480000
WHISPER_NUM_FRAMES = 1500
SAMPLES_PER_FRAME = WHISPER_INPUT_SAMPLES // WHISPER_NUM_FRAMES  # 320

def pad_or_truncate(segment, target_len=WHISPER_INPUT_SAMPLES):
    if len(segment) > target_len:
        return segment[:target_len]
    else:
        return np.pad(segment, (0, target_len - len(segment)))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

block_index = 2  # Choose block index as needed

# Load model to the same device
model = whisper.load_model("tiny").to(device)  # small, base, large
# Set model to eval mode
model.eval()

# Store activations per segment
activations_list = []

# Hook dict
activation_dict = {}

def hook_fn(module, input, output):
    activation_dict['mlp'] = output.detach().cpu()  # Move to CPU immediately

# Register hook to desired block's MLP
mlp_module = model.encoder.blocks[block_index].mlp
hook = mlp_module.register_forward_hook(hook_fn)

# Process each row
for i, row in df.iterrows():
    original_segment = np.asarray(row['segment'], dtype=np.float32)
    segment = pad_or_truncate(original_segment)

    # Convert to tensor and move to correct device
    audio_tensor = torch.from_numpy(segment).unsqueeze(0).to(device)  # Shape: (1, T)

    # Compute log-mel (this returns CPU tensor)
    mel = whisper.log_mel_spectrogram(audio_tensor).to(device)  # Move to correct device

    # Forward pass (hook will capture MLP activations)
    with torch.no_grad():
        _ = model.encoder(mel)

    # Store only the relevant slice of activations
    if 'mlp' in activation_dict:
        n_samples = len(original_segment)
        n_frames = n_samples // SAMPLES_PER_FRAME
        activation_slice = activation_dict['mlp'][0, :n_frames, :]
        activations_list.append(activation_slice)
    else:
        activations_list.append(None)  # In case hook fails for some reason

    print(f"Processed segment {i}, key: {row['key'].split('_')}")

# Add new column to dataframe
df[f'activations_block{block_index}_mlp'] = activations_list

activations_list[0].shape


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_waveform_with_activation_stats(
    waveform,
    activations,
    phoneme=None,
    utterance=None,
    sample_rate=16000,
    std_alpha=0.2
):
    """
    Plot waveform with activation statistics (mean, min, max, std) on a secondary y-axis.

    Parameters:
        waveform (np.ndarray): 1D float32 array of waveform samples.
        activations (torch.Tensor): Shape (n_frames, hidden_dim), MLP activations.
        phoneme (str): Optional label for phoneme.
        utterance (str): Optional label for utterance.
        sample_rate (int): Default 16000 Hz.
        std_alpha (float): Transparency for std shading.
    """
    waveform = np.asarray(waveform)
    n_samples = len(waveform)
    n_frames, hidden_dim = activations.shape

    # Time vectors
    time_waveform = np.linspace(0, n_samples / sample_rate, n_samples)
    frame_times = np.linspace(0, n_samples / sample_rate, n_frames)

    # Aggregate stats over hidden dim
    mean_vals = torch.mean(activations, dim=1).numpy()
    min_vals = torch.min(activations, dim=1).values.numpy()
    max_vals = torch.max(activations, dim=1).values.numpy()
    std_vals = torch.std(activations, dim=1).numpy()

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 4))

    # Left y-axis: waveform
    ax1.plot(time_waveform, waveform, color='black', linewidth=1, label='Waveform')
    ax1.set_ylabel("Amplitude", color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Right y-axis: activations
    ax2 = ax1.twinx()
    ax2.plot(frame_times, mean_vals, label='Mean', color='purple', linewidth=2)
    ax2.plot(frame_times, min_vals, label='Min', color='blue', linestyle='dashed')
    ax2.plot(frame_times, max_vals, label='Max', color='red', linestyle='dashed')

    # Shaded std area
    ax2.fill_between(
        frame_times,
        mean_vals - std_vals,
        mean_vals + std_vals,
        color='purple',
        alpha=std_alpha,
        label='±1 STD'
    )

    ax2.set_ylabel("MLP Activation Value", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Title
    title = f"Waveform + MLP Activation Stats"
    if phoneme:
        title += f" – Phoneme: '{phoneme}'"
    if utterance:
        title += f" – Utterance: {utterance}"
    ax1.set_title(title)

    # Legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


row = df.iloc[42]
waveform = np.asarray(row['segment'], dtype=np.float32)
activations = row['activations_block2_mlp']

plot_waveform_with_activation_stats(
    waveform=waveform,
    activations=activations,
    phoneme=row.get('phoneme'),
    utterance=row.get('utterance')
)
