import numpy as np
import requests
import io

# 1. Corrected URLs with dl=1 for direct download
RX_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/ALZQcmH0nHHmylH8NTkU_ns/phase1_timing/snr_5db/sample_002/rx.npy?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
META_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/AJvuR7a2XZetEJruTff6hvc/phase1_timing/snr_5db/sample_002/meta.json?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
# Load waveform data from URL
try:
    r_waveform = requests.get(RX_URL)
    r_waveform.raise_for_status()
    # 2. Added allow_pickle=True to load the file
    rx = np.load(io.BytesIO(r_waveform.content), allow_pickle=True)
    print(f"Waveform loaded: shape={rx.shape}, dtype={rx.dtype}")
except Exception as e:
    print(f"Error loading waveform: {e}")
    exit()

# Load metadata JSON from URL
try:
    r_meta = requests.get(META_URL)
    r_meta.raise_for_status()
    meta = r_meta.json()
    print("Metadata loaded.")
except Exception as e:
    print(f"Error loading or parsing metadata: {e}")
    print("Response content received:")
    print(r_meta.text)
    exit()

# --- Signal Processing Steps ---

# Extract parameters from meta
sps = meta["sps"]
timing_offset = meta["timing_offset"]
clean_bits = np.array(meta["clean_bits"])

# Step 1: Matched filter
matched_filter = np.ones(sps)
filtered = np.convolve(rx, matched_filter, mode='same')

# Step 2: Symbol timing recovery
sampled = filtered[timing_offset::sps]

# Step 3: Carrier Phase Correction
# Estimate the phase by calculating the average angle of the symbols.
phase_offset_estimate = np.angle(np.mean(sampled**4)) / 4
corrected_sampled = sampled * np.exp(-1j * phase_offset_estimate)

# Step 4: Hard decision BPSK demodulation (on the corrected symbols)
decoded_bits = (np.real(corrected_sampled) > 0).astype(int)

# Step 5: Trim bits to match the length of clean_bits
decoded_bits = decoded_bits[:len(clean_bits)]

# Step 6: Calculate BER
bit_errors = np.sum(decoded_bits != clean_bits)
ber = bit_errors / len(clean_bits)
print(f"BER: {ber:.6f} ({bit_errors} errors out of {len(clean_bits)})")

# Step 7: Save decoded bits
np.save('decoded_bitssample002_5db.npy', decoded_bits)
print("Decoded bits saved as 'decoded_bitssample002_5db.npy'.")

# Optional: Show first 20 decoded bits
print("First 20 decoded bits:", decoded_bits[:20])