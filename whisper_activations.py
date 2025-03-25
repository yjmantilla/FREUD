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

def plot_waveform_with_activations(
    waveform,
    activations,
    phoneme=None,
    utterance=None,
    sample_rate=16000,
    agg_fn=torch.mean,
    agg_label="mean",
    cmap='plasma'
):
    """
    Plot waveform with vertical lines for activation frames and an activation heatmap underneath.

    Parameters:
        waveform (np.ndarray): 1D float32 array of waveform samples.
        activations (torch.Tensor): Shape (n_frames, hidden_dim), MLP activations.
        phoneme (str): Optional label for phoneme.
        utterance (str): Optional label for utterance.
        sample_rate (int): Default 16000 Hz.
        agg_fn (function): Aggregation function to reduce activations across hidden_dim (e.g., torch.mean).
        agg_label (str): Label to display for aggregation (e.g., 'mean', 'max').
        cmap (str): Matplotlib colormap for heatmap.
    """
    waveform = np.asarray(waveform)
    n_samples = len(waveform)
    n_frames, hidden_dim = activations.shape
    print(n_frames, hidden_dim)

    # Time vectors
    time_waveform = np.linspace(0, n_samples / sample_rate, n_samples)
    frame_times = np.linspace(0, n_samples / sample_rate, n_frames)

    # Aggregate activation per frame
    activation_values = agg_fn(activations, dim=1)
    
    if 'values' in dir(activation_values):
        activation_values = activation_values.values

    # Normalize activation values for colormap
    norm = plt.Normalize(vmin=activation_values.min(), vmax=activation_values.max())
    colors = matplotlib.colormaps[cmap](norm(activation_values))
    #plt.cm.get_cmap(cmap)(norm(activation_values))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={'height_ratios': [3, 0.5]})
    
    # Waveform plot
    ax1.plot(time_waveform, waveform, color='black', linewidth=1)
    for ft, color in zip(frame_times, colors):
        ax1.axvline(x=ft, color=color, alpha=0.3, linewidth=1)

    ax1.set_ylabel("Amplitude")
    title = f"Waveform with MLP activation ({agg_label}) overlay"
    if phoneme:
        title += f" – Phoneme: '{phoneme}'"
    if utterance:
        title += f" – Utterance: {utterance}"
    ax1.set_title(title)
    
    # Heatmap
    heatmap = ax2.imshow(
        activation_values[None, :],  # add fake height dim
        aspect='auto',
        cmap=cmap,
        extent=[0, time_waveform[-1], 0, 1]
    )
    ax2.set_yticks([])
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Activation ({agg_label}) per frame", fontsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label(f"MLP Activation ({agg_label})", rotation=90)

    plt.tight_layout()
    plt.show()

row = df.iloc[42]  # Pick any index
waveform = np.asarray(row['segment'], dtype=np.float32)
activations = row['activations_block2_mlp']  # adjust if you renamed column

plot_waveform_with_activations(
    waveform=waveform,
    activations=activations,
    phoneme=row.get('phoneme'),
    utterance=row.get('utterance'),
    agg_fn=torch.max,       # You could also try torch.max or torch.std
    agg_label="max",        # Label for the aggregation method
    cmap='plasma'            # Try also: 'viridis', 'inferno', 'coolwarm'
)