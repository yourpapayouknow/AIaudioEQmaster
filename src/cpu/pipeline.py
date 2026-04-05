from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt
from .audio_io import load_audio, save_audio, apply_dither
from .transforms import lr_to_ms, ms_to_lr, oversample, downsample, improved_anti_aliasing_filter, apply_lowpass_filter, add_subtle_mid_channel_saturation, calculate_improved_rms
from .spectral import match_rms_ms, match_frequencies_ms, gradual_level_correction
from .stereo import finalize_stereo_image, analyze_stereo_width
from .dynamics import multi_stage_limiter
from .reference import load_genre_profile, calculate_lufs, create_reference_from_profile, apply_eq_style, low_shelf_tighten

def process_audio(target, reference, step, config, genre_profile=None):
    start_time = time.time()
    print(f"Processing step {step}")
    print(f"Input target shape: {target.shape}, max={np.max(np.abs(target))}, min={np.min(np.abs(target))}")
    
    log_audio_metrics(target, "Target (Before Processing)", config)
    log_audio_metrics(reference, "Reference" if genre_profile is None else "Synthetic Reference", config)
    
    # Calculate and log initial LUFS
    target_lufs = calculate_lufs(target, config.internal_sample_rate)
    print(f"Target LUFS before processing: {target_lufs:.2f}")

    if genre_profile is None:
        reference_lufs = calculate_lufs(reference, config.internal_sample_rate)
        print(f"Reference LUFS: {reference_lufs:.2f}")
    else:
        synthetic_reference_lufs = calculate_lufs(reference, config.internal_sample_rate)
        print(f"Synthetic Reference LUFS: {synthetic_reference_lufs:.2f}")
        print(f"Genre profile LUFS: {genre_profile['lufs']:.2f}")
    
    def calculate_rms(audio):
        return np.sqrt(np.mean(np.square(audio)))

    # Calculate and log initial RMS values
    target_rms = calculate_rms(target)
    if genre_profile is None:
        reference_rms = calculate_rms(reference)
        print(f"Initial RMS - Reference: {reference_rms:.6f}, Target: {target_rms:.6f}")
    else:
        synthetic_reference_rms = calculate_rms(reference)
        print(f"Initial RMS - Synthetic Reference: {synthetic_reference_rms:.6f}, Target: {target_rms:.6f}")
        print(f"Genre Profile Initial RMS: {genre_profile['initial_rms']:.6f}")

    # Store the initial loudness ratio
    initial_loudness_ratio = 1.0
    if config.loudness_option == "dynamic":
        initial_loudness_ratio = 0.80
        print("Applying dynamic loudness: reducing volume by 20%")
    elif config.loudness_option == "soft":
        initial_loudness_ratio = 0.70
        print("Applying soft loudness: reducing volume by 30%")
    elif config.loudness_option == "loud":
        initial_loudness_ratio = 1.20
        print("Applying loud loudness: increasing volume by 20%")
    else:
        print("Applying normal loudness: no adjustment")

    # Apply initial loudness adjustment
    target *= initial_loudness_ratio

    # Recalculate RMS and LUFS after loudness adjustment
    if config.loudness_option != "normal":
        target_rms = calculate_rms(target)
        target_lufs = calculate_lufs(target, config.internal_sample_rate)
        print(f"After loudness adjustment - Target RMS: {target_rms:.6f}, Target LUFS: {target_lufs:.2f}")
    
    # Ensure input audio is in 64-bit float precision
    target = target.astype(np.float64)
    reference = reference.astype(np.float64)
    
    # Oversample
    target = oversample(target, config.oversampling_factor)
    reference = oversample(reference, config.oversampling_factor)
    
    oversampled_rate = config.internal_sample_rate * config.oversampling_factor
    print(f"After oversampling: target_max={np.max(np.abs(target))}")
    
    # Apply anti-aliasing filter
    target = improved_anti_aliasing_filter(target, oversampled_rate)
    reference = improved_anti_aliasing_filter(reference, oversampled_rate)
    print(f"After anti-aliasing: target_max={np.max(np.abs(target))}, reference_max={np.max(np.abs(reference))}")
    
    # Convert to mid-side
    target_mid, target_side = lr_to_ms(target)
    reference_mid, reference_side = lr_to_ms(reference)
    print(f"After mid-side conversion: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
    print(f"reference_mid_max={np.max(np.abs(reference_mid))}, reference_side_max={np.max(np.abs(reference_side))}")
    
    # Apply processing steps
    if step >= 1:
        if genre_profile is None:
            target_mid, target_side = match_rms_ms(target_mid, target_side, reference_mid, reference_side, oversampled_rate, config)
        else:
            target_mid, target_side = match_rms_ms(target_mid, target_side, reference_mid, reference_side, oversampled_rate, config)
        print(f"After RMS matching: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
        
        # Calculate and log RMS after matching
        processed_mid_side = ms_to_lr(target_mid, target_side)
        processed_rms = calculate_improved_rms(processed_mid_side, oversampled_rate, config)
        reference_rms = calculate_improved_rms(reference, oversampled_rate, config)
        print(f"After RMS matching - Reference RMS: {reference_rms:.6f}, Processed RMS: {processed_rms:.6f}")
        
        print("After RMS Matching:")
        log_audio_metrics(ms_to_lr(target_mid, target_side), "Target", config)
    
    if step >= 2:

        # Apply subtle saturation to mid channel
        target_mid = add_subtle_mid_channel_saturation(target_mid, config)
        print(f"After mid channel saturation: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")

        if genre_profile is None:
            target_mid, target_side = match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config)
        else:
            target_mid, target_side = match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config)
        print(f"After frequency matching: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
        
        # Apply high-pass filter to side channel
        target_side = low_shelf_tighten(target_side, config.internal_sample_rate, cutoff_freq=100, gain=0.5, order=4)
        print(f"After side channel high-pass: target_side_max={np.max(np.abs(target_side))}")

        # Apply EQ style after frequency matching
        if config.eq_style != "Neutral":
            print(f"Applying {config.eq_style} EQ style")
            target_mid, target_side = apply_eq_style(target_mid, target_side, config.internal_sample_rate, config.eq_style)
        
        # Apply lowpass filter
        result = ms_to_lr(target_mid, target_side)
        result = apply_lowpass_filter(result, config)
        target_mid, target_side = lr_to_ms(result)
        print("After Lowpass Filter:")
        log_audio_metrics(result, "Target", config)
    
    if step >= 3:
        if genre_profile is None:
            target_mid, target_side = gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config)
        else:
            target_mid, target_side = gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config)
        print(f"After gradual level correction: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
        
        print("After Level Correction:")
        log_audio_metrics(ms_to_lr(target_mid, target_side), "Target", config)
    
    if step >= 4:
        if genre_profile is None:
            result = finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config)
        else:
            result = finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config)
        print(f"After stereo finalization: result_max={np.max(np.abs(result))}")
        
        print("After Stereo Adjustment:")
        log_audio_metrics(result, "Target", config)
    else:
        result = ms_to_lr(target_mid, target_side)
    
    if step >= 5:
        print("Applying final mastering processes...")
        print(f"Before multi-stage limiting: result_max={np.max(np.abs(result)):.4f}")
        
        # Before final limiting, reapply the loudness ratio
        result *= initial_loudness_ratio
        
        result = multi_stage_limiter(result, config)
        print(f"After multi-stage limiting: result_max={np.max(np.abs(result)):.4f}")

    # Apply final hard limiting before downsampling
    result = np.clip(result, -0.95, 0.95)
    print(f"After final hard limiting (before downsampling): result_max={np.max(np.abs(result)):.4f}")

    # Downsample (now outside of step 5)
    result = downsample(result, config.oversampling_factor, oversampled_rate)
    print(f"After downsampling: result_max={np.max(np.abs(result)):.4f}")

    # Add normalization step
    target_peak_db = -0.5
    current_peak_db = 20 * np.log10(np.max(np.abs(result)))
    if current_peak_db < target_peak_db:
        gain_db = target_peak_db - current_peak_db
        gain_linear = 10 ** (gain_db / 20)
        result *= gain_linear
        print(f"Normalized to {target_peak_db} dB. Applied gain: {gain_db:.2f} dB")
    else:
        print(f"Current peak ({current_peak_db:.2f} dB) is already at or above target peak. No additional normalization applied.")

    print(f"Final output: result_max={np.max(np.abs(result)):.4f}")

    # Calculate and log initial LUFS before True Peak limiting
    initial_lufs = calculate_lufs(result, config.internal_sample_rate)
    print(f"LUFS before True Peak limiting: {initial_lufs:.2f}")
    
    # Apply simple dithering before final True Peak limiting
    result = apply_dither(result)
    
    # Calculate True Peak
    true_peak_db = calculate_true_peak(result, config.internal_sample_rate)
    print(f"Initial True Peak (dBTP): {true_peak_db:.2f}")

    # Apply True Peak limiting if necessary
    if true_peak_db > -0.4:
        gain_reduction_db = -0.4 - true_peak_db
        gain_factor = 10 ** (gain_reduction_db / 20)
        result *= gain_factor
        print(f"Applied True Peak limiting. Gain reduction: {gain_reduction_db:.2f} dB")
        
        # Recalculate True Peak after limiting
        true_peak_db = calculate_true_peak(result, config.internal_sample_rate)
        print(f"Final True Peak after limiting (dBTP): {true_peak_db:.2f}")
    else:
        print("True Peak is already below -0.4 dBTP. No additional limiting applied.")

    print(f"Final output: result_max={np.max(np.abs(result)):.4f}")

    # Genre-specific loudness adjustment
    if genre_profile and genre_profile['genre'] in ['Piano', 'Orchestral', 'Speech']:
        if config.loudness_option == "dynamic":
            # Reduce gain to simulate -0.6 dB true peak
            result *= 10 ** (-0.3 / 20)  # Additional -0.3 dB
        elif config.loudness_option == "soft":
            # Reduce gain to simulate -0.9 dB true peak
            result *= 10 ** (-0.6 / 20)  # Additional -0.6 dB

    # Calculate and log final LUFS after all processing
    final_lufs = calculate_lufs(result, config.internal_sample_rate)
    print(f"Final LUFS after all processing: {final_lufs:.2f}")

    print("After Final Processing:")
    log_audio_metrics(result, "Target", config)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time for step {step}: {total_time:.2f} seconds")
    
    return result

def calculate_true_peak(audio, sample_rate):
    # Upsample by a factor of 4 for true peak calculation
    upsampled = signal.resample_poly(audio, 4, 1, axis=-1)
    peak = np.max(np.abs(upsampled))
    true_peak_db = 20 * np.log10(peak)
    return true_peak_db

def log_audio_metrics(audio, name, config):
    print(f"--- {name} Metrics ---")
    print(f"Shape: {audio.shape}")
    print(f"Max amplitude: {np.max(np.abs(audio)):.4f}")
    lufs = calculate_lufs(audio, config.internal_sample_rate)
    print(f"LUFS: {lufs:.2f}")
    
    mid, side = lr_to_ms(audio)
    print(f"Mid RMS: {np.sqrt(np.mean(np.square(mid))):.4f}")
    print(f"Side RMS: {np.sqrt(np.mean(np.square(side))):.4f}")
    
    stereo_width = analyze_stereo_width(mid, side)
    print(f"Stereo Width: {stereo_width:.4f}")
    print("-------------------")

def master_audio(input_file, output_file, config, eq_style, is_preview=False):
    start_time = time.time()
    print(f"Master audio function called with: input_file={input_file}, output_file={output_file}, reference_file={config.reference_file}, eq_style={eq_style}, is_preview={is_preview}")
    config.eq_style = eq_style
    
    load_start = time.time()
    target, sr = load_audio(input_file, config)
    load_end = time.time()
    print(f"Audio loading time: {load_end - load_start:.2f} seconds")
    
    print(f"Original audio length: {len(target[0])} samples")
    print(f"Original audio duration: {len(target[0]) / sr:.2f} seconds")
    
    if is_preview:
        preview_start = time.time()
        preview_duration = 30  # seconds
        preview_samples = min(sr * preview_duration, target.shape[1])
        target = target[:, :preview_samples]
        preview_end = time.time()
        print(f"Preview creation time: {preview_end - preview_start:.2f} seconds")
        print(f"Processing preview: {preview_samples} samples")
        print(f"Preview duration: {preview_samples / sr:.2f} seconds")
    else:
        print(f"Processing full track: {len(target[0])} samples")
    
    if config.genre:
        print(f"Using genre profile: {config.genre}")
        genre_profile = load_genre_profile(config.genre)
        reference, _ = create_reference_from_profile(genre_profile, config)
        log_audio_metrics(reference, "Reference from Genre", config)
    elif config.reference_file:
        print(f"Using reference file: {config.reference_file}")
        reference, _ = load_audio(config.reference_file, config)
        genre_profile = None
    else:
        raise ValueError("Either genre or reference file must be specified")
    
    process_start = time.time()
    processed_audio = process_audio(target, reference, 5, config, genre_profile)
    process_end = time.time()
    print(f"Audio processing time: {process_end - process_start:.2f} seconds")
    
    save_start = time.time()
    save_audio(processed_audio, output_file, sr)    
    save_end = time.time()
    print(f"Audio saving time: {save_end - save_start:.2f} seconds")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Python processing time: {total_time:.2f} seconds")
    print("Mastering completed")
