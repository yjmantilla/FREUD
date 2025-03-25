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
