import nltk
from nltk.corpus import timit
import numpy as np
# Uncomment the following line if you haven't downloaded TIMIT corpus yet
#nltk.download('timit')

# Create a dictionary to store phoneme segments across all utterances
phoneme_segments = {}

utterances = [x.split('.phn')[0] for x in nltk.corpus.timit.fileids() if x.endswith('.phn')]

# Iterate over all fileids (utterances) in the corpus
for utterance in utterances:
    try:
        # Get phonemes and their corresponding time alignments for this utterance
        phonemes = timit.phones(utterance)
        times = timit.phone_times(utterance)
        
        # Get the audio data for the utterance
        audio = timit.audiodata(utterance)
        
        # Extract audio segments for each phoneme in this utterance
        for (phone, start, end), phoneme in zip(times, phonemes):
            # Convert time to sample indices
            start_sample = int(start * 16000)  # TIMIT uses 16kHz sampling rate
            end_sample = int(end * 16000)
            
            # Extract the audio segment for this phoneme
            segment = audio[start_sample:end_sample]
            
            # Store the segment
            if phone not in phoneme_segments:
                phoneme_segments[phone] = []
            phoneme_segments[phone].append((np.array(segment), utterance))  # Store utterance ID with segment
            
        print(f"Processed utterance: {utterance}")
        
    except Exception as e:
        print(f"Error processing utterance {utterance}: {str(e)}")

# Print summary information about all segments
print("\nSummary of all phoneme segments:")
for phone, segments in phoneme_segments.items():
    print(f"Phoneme '{phone}' has {len(segments)} segments from all utterances")

phoneme_segments.keys()

phoneme_segments['h#'][0]