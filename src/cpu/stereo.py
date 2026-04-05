from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt
from .transforms import lr_to_ms, ms_to_lr

def rms(audio):
    return np.sqrt(np.mean(np.square(audio)))

def segment_audio(audio, config):
    segment_length = config.internal_sample_rate  # 1 second segments
    num_full_segments = len(audio) // segment_length
    segments = np.array_split(audio[:num_full_segments * segment_length], num_full_segments)
    return np.array(segments)

def analyze_stereo_width(mid, side):
    mid_energy = np.mean(np.square(mid))
    side_energy = np.mean(np.square(side))
    return side_energy / (mid_energy + side_energy + 1e-8)

def adjust_stereo_balance(mid, side, target_width, config):
    current_width = analyze_stereo_width(mid, side)
    
    width_difference = target_width - current_width
    if abs(width_difference) < 0.05:  # Less than 5% difference
        return mid, side
    
    adjustment_factor = 1 + config.stereo_width_adjustment_factor * width_difference
    
    # Only adjust the side channel
    adjusted_side = side * adjustment_factor
    
    # Ensure RMS remains constant
    original_rms = np.sqrt(np.mean(mid**2 + side**2))
    adjusted_rms = np.sqrt(np.mean(mid**2 + adjusted_side**2))
    rms_correction = original_rms / adjusted_rms
    
    return mid, adjusted_side * rms_correction

def finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config):
    print("Finalizing stereo image...")
    try:
        # Calculate stereo widths
        initial_width = analyze_stereo_width(target_mid, target_side)
        reference_width = analyze_stereo_width(reference_mid, reference_side)
        
        print(f"Initial stereo width: {initial_width:.4f}")
        print(f"Reference stereo width: {reference_width:.4f}")
        
        # Check for near-mono signal
        if initial_width < 0.02:
            print("Input signal is nearly mono. Skipping stereo adjustment.")
            return ms_to_lr(target_mid, target_side)
        
        # Calculate initial RMS
        initial_rms = np.sqrt(np.mean(target_mid**2 + target_side**2))
        
        # Calculate width difference and apply adjustment with upper limit
        width_difference = reference_width - initial_width
        max_adjustment = 0.15  # 15% maximum adjustment
        
        if width_difference > 0:
            adjustment_factor = 1 + min(width_difference / initial_width, max_adjustment)
            adjusted_side = target_side * adjustment_factor
            print(f"Applied {(adjustment_factor - 1) * 100:.1f}% stereo width increase.")
        else:
            print("No stereo width increase needed.")
            adjusted_side = target_side
        
        # Convert to left-right
        result = ms_to_lr(target_mid, adjusted_side)
        
        # RMS matching
        current_rms = np.sqrt(np.mean(result**2))
        rms_adjustment = initial_rms / current_rms
        result *= rms_adjustment
        
        final_mid, final_side = lr_to_ms(result)
        final_width = analyze_stereo_width(final_mid, final_side)
        
        print(f"Initial stereo width: {initial_width:.4f}")
        print(f"Final stereo width: {final_width:.4f}")
        print(f"Stereo image finalization complete. Output max amplitude: {np.max(np.abs(result)):.4f}")
        
        return result
    except Exception as e:
        print(f"Error during stereo image finalization: {str(e)}")
        return ms_to_lr(target_mid, target_side)

def process_band(args):
    mid, side, target_balance, config, band = args
    if band[0] == 0:
        sos = signal.butter(10, band[1], btype='lowpass', fs=config.internal_sample_rate, output='sos')
    else:
        sos = signal.butter(10, band, btype='bandpass', fs=config.internal_sample_rate, output='sos')
    
    band_mid = signal.sosfilt(sos, mid)
    band_side = signal.sosfilt(sos, side)
    
    return adjust_stereo_balance(band_mid, band_side, target_balance, config)

def frequency_band_stereo_adjust(mid, side, target_balance, config):
    bands = [0, 250, 8000, config.internal_sample_rate // 2]  # Reduced to 3 bands
    
    with Pool() as pool:
        results = pool.map(process_band, [(mid, side, target_balance, config, (bands[i], bands[i+1])) for i in range(len(bands) - 1)])
    
    adjusted_mid = sum(result[0] for result in results)
    adjusted_side = sum(result[1] for result in results)
    
    return adjusted_mid, adjusted_side
