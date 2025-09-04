import numpy as np
import requests
import io


# --- Helper Function for RRC Filter ---
def generate_rrc_filter(sps, num_taps, beta):
    """
    Generates a Root-Raised Cosine (RRC) filter.

    Args:
        sps (int): Samples per symbol.
        num_taps (int): The length of the filter (must be an odd number).
        beta (float): The roll-off factor (typically between 0.2 and 0.5).

    Returns:
        numpy.ndarray: The RRC filter coefficients.
    """
    t = np.arange(num_taps) - (num_taps - 1) / 2
    t /= sps

    # Handle the special cases for t=0 and t=±sps/(4*beta)
    sinc_term = np.sinc(t)
    cos_term = np.cos(np.pi * beta * t)
    denom_term = 1 - (2 * beta * t) ** 2

    h_rrc = np.zeros(num_taps)

    # Denominator is zero at t = ±1/(2*beta), so handle these points separately
    idx_denom_zero = np.where(np.abs(denom_term) < 1e-8)[0]
    idx_all_others = np.ones(num_taps, dtype=bool)
    idx_all_others[idx_denom_zero] = False

    h_rrc[idx_all_others] = sinc_term[idx_all_others] * cos_term[idx_all_others] / denom_term[idx_all_others]

    # Special case for the center tap t=0
    center_idx = num_taps // 2
    h_rrc[center_idx] = (1 - beta + 4 * beta / np.pi)

    # Special case for t = ±1/(2*beta)
    if idx_denom_zero.size > 0:
        val = beta / np.sqrt(2) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        h_rrc[idx_denom_zero] = val

    return h_rrc


# --- Main Script ---

# Corrected URLs with dl=1 for direct download
RX_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/AOkJlQr_yHLfeAe0iqOZYGM/phase1_timing/snr_10db/sample_002/rx.npy?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
META_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/ADF7oyGMTh3J0b1VMUr-oXI/phase1_timing/snr_10db/sample_002/meta.json?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
# Load waveform data from URL
try:
    r_waveform = requests.get(RX_URL)
    r_waveform.raise_for_status()
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
    exit()

# --- Signal Processing Steps ---

# Extract parameters from meta
sps = meta["sps"]
timing_offset = meta["timing_offset"]
clean_bits = np.array(meta["clean_bits"])

# Step 1: Matched filter (IMPROVED using RRC)
num_taps = 101  # A good filter length, must be odd
beta = 0.35  # Standard roll-off factor
matched_filter = generate_rrc_filter(sps, num_taps, beta)
filtered = np.convolve(rx, matched_filter, mode='same')

# Step 2: Symbol timing recovery
sampled = filtered[timing_offset::sps]

# Step 3: Carrier Phase Correction (IMPROVED using a PLL)
corrected_sampled = np.zeros_like(sampled)
phase_integrator = 0.0
prop_gain = 0.1
int_gain = 0.005

for i, s in enumerate(sampled):
    # Apply phase correction to the current symbol
    corrected_symbol = s * np.exp(-1j * phase_integrator)
    corrected_sampled[i] = corrected_symbol

    # Costas Loop phase error detector for BPSK
    phase_error = np.sign(np.real(corrected_symbol)) * np.imag(corrected_symbol)

    # Update the loop filter (integrator)
    phase_integrator += prop_gain * phase_error + int_gain * phase_error

# Step 4: Hard decision BPSK demodulation (on the corrected symbols)
decoded_bits = (np.real(corrected_sampled) > 0).astype(int)

# Step 5: Trim bits to match the length of clean_bits
decoded_bits = decoded_bits[:len(clean_bits)]

# Step 6: Calculate BER
bit_errors = np.sum(decoded_bits != clean_bits)
ber = bit_errors / len(clean_bits)
print(f"BER with improvements: {ber:.6f} ({bit_errors} errors out of {len(clean_bits)})")

# Step 7: Save decoded bits
np.save('improveddecodedbitssample002_10db.npy', decoded_bits)
print("Decoded bits saved as 'improveddecodedbitssample002_10db.npy.npy'.")