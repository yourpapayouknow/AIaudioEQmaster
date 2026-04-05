from ..cpu.dynamics import process_limiter_stage


def multi_stage_limiter(audio, config):
    # Reuse original numba limiter core to avoid audible clipping distortion.
    threshold1 = 10 ** (-1.0 / 20)  # gentler first catch
    knee_width1 = 0.12
    attack_time1 = 1.5
    release_time1 = 220.0

    threshold2 = 10 ** (-0.6 / 20)  # safety stage
    knee_width2 = 0.10
    attack_time2 = 3.0
    release_time2 = 750.0

    sample_rate = config.internal_sample_rate * config.oversampling_factor

    result = process_limiter_stage(
        audio,
        threshold1,
        knee_width1,
        attack_time1,
        release_time1,
        sample_rate,
    )
    result = process_limiter_stage(
        result,
        threshold2,
        knee_width2,
        attack_time2,
        release_time2,
        sample_rate,
    )
    return result

