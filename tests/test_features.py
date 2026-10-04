import numpy as np

from earshot.data import features as F


def tone(hz, amp=0.5):
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_shape():
    assert F.log_mel(np.zeros(F.CLIP_SAMPLES)).shape == (F.N_FRAMES, F.N_MELS) == (49, 40)


def test_batched_matches_single():
    x = np.random.default_rng(0).uniform(-1, 1, (3, F.CLIP_SAMPLES)).astype(np.float32)
    batched = F.log_mel(x)
    for i in range(3):
        np.testing.assert_allclose(batched[i], F.log_mel(x[i]), rtol=1e-5, atol=1e-5)


def test_tone_peaks_in_nearest_mel_bin():
    centers = F.mel_to_hz(np.linspace(F.hz_to_mel(F.FMIN), F.hz_to_mel(F.FMAX), F.N_MELS + 2))[1:-1]
    for hz in (300, 1000, 3000, 6000):
        assert F.log_mel(tone(hz)).mean(0).argmax() == np.abs(centers - hz).argmin()


def test_silence_is_log_eps():
    np.testing.assert_allclose(F.log_mel(np.zeros(F.CLIP_SAMPLES)), np.log(F.LOG_EPS), rtol=1e-6)


def test_filterbank_covers_every_bin_in_range():
    fb = F.mel_filterbank()
    hz = np.arange(F.N_FFT // 2 + 1) * F.SAMPLE_RATE / F.N_FFT
    inside = (hz > F.FMIN) & (hz < F.FMAX)
    assert (fb[:, inside].sum(0) > 0).all()
    assert (fb[:, ~inside].sum(0) == 0).all()


def test_fix_length():
    assert F.fix_length(np.ones(12000)).shape == (F.CLIP_SAMPLES,)
    assert F.fix_length(np.ones(17000)).shape == (F.CLIP_SAMPLES,)
    assert F.fix_length(np.ones(12000))[-1] == 0
