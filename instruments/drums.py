import numpy as np
import mido
import io
import wave
import os
import glob
import random
from scipy import signal

SR = 48000

# 全局采样缓存
SAMPLE_CACHE = {}
SAMPLES_LOADED = False


# ==========================================
# 1. DSP 工具箱 (模拟电路建模基础)
# ==========================================

def saturation(x, drive=1.0):
    """模拟磁带/电子管饱和失真，增加厚度"""
    if drive <= 0: return x
    # 使用软拐点 tanh，但限制最大增益防止破音
    return np.tanh(x * drive)


def generate_metallic_noise(n_samples):
    """
    [修复] 生成金属噪声 (TR-808 风格 - 改进版)
    原先使用方波会导致严重的“电流声/漏电声”(Aliasing)。
    现在改用纯正弦波叠加 (Additive Synthesis)，声音更干净、像真实的铜镲。
    """
    t = np.linspace(0, n_samples / SR, n_samples)
    # TR-808 经典频率比率
    freqs = [263, 400, 421, 474, 587, 845] 
    noise = np.zeros(n_samples)
    for f in freqs:
        # [关键修改] 去掉了 np.sign()，不再使用方波，消除滋滋声
        noise += np.sin(2 * np.pi * f * t)
    return noise / len(freqs)


def apply_envelope(wave, decay_rate):
    """应用指数衰减包络"""
    t = np.linspace(0, len(wave) / SR, len(wave))
    return wave * np.exp(-decay_rate * t)


# ==========================================
# 2. 采样加载器 (Sample Loader)
# ==========================================
def load_drum_samples():
    global SAMPLE_CACHE, SAMPLES_LOADED
    if SAMPLES_LOADED: return

    base_dirs = ["assets/drum_samples", "../assets/drum_samples", "./drum_samples"]
    found_root = None
    for d in base_dirs:
        if os.path.exists(d):
            found_root = d
            break

    if not found_root:
        SAMPLES_LOADED = True
        return

    categories = ['kick', 'snare', 'hihat', 'tom', 'crash', 'ride']

    for cat in categories:
        SAMPLE_CACHE[cat] = []
        cat_dir = os.path.join(found_root, cat)
        if not os.path.exists(cat_dir): continue

        files = glob.glob(os.path.join(cat_dir, "*.wav"))
        for fpath in files:
            try:
                with wave.open(fpath, 'rb') as wf:
                    if wf.getnchannels() != 1: pass
                    frames = wf.readframes(wf.getnframes())
                    dtype = np.int16 if wf.getsampwidth() == 2 else np.uint8
                    raw = np.frombuffer(frames, dtype=dtype)

                    if wf.getsampwidth() == 2:
                        audio = raw.astype(np.float32) / 32768.0
                    else:
                        audio = (raw.astype(np.float32) - 128.0) / 128.0

                    fname = os.path.basename(fpath).lower()
                    intensity = 'medium'
                    if 'hard' in fname or 'loud' in fname:
                        intensity = 'hard'
                    elif 'soft' in fname or 'quiet' in fname:
                        intensity = 'soft'

                    SAMPLE_CACHE[cat].append({'data': audio, 'intensity': intensity})
            except:
                pass

    SAMPLES_LOADED = True


def get_sample_processed(category, velocity):
    if category not in SAMPLE_CACHE or not SAMPLE_CACHE[category]:
        return None

    candidates = SAMPLE_CACHE[category]
    target_intensity = 'medium'
    if velocity > 0.85:
        target_intensity = 'hard'
    elif velocity < 0.4:
        target_intensity = 'soft'

    matches = [s for s in candidates if s['intensity'] == target_intensity]
    if not matches: matches = candidates

    selected = random.choice(matches)
    audio = selected['data'].copy()

    # Pitch Jitter
    pitch_shift = np.random.uniform(0.99, 1.01) # 减小抖动范围，更自然
    if pitch_shift != 1.0 and len(audio) > 100:
        indices = np.arange(0, len(audio), pitch_shift)
        indices = indices[indices < len(audio) - 1]
        audio = np.interp(indices, np.arange(len(audio)), audio)

    gain = 0.3 + velocity * 0.7
    return audio * gain


# ==========================================
# 3. 高级模拟合成引擎 (修复电流声)
# ==========================================

def synth_kick_advanced(duration_samples, velocity, brightness):
    """TR-909 风格底鼓 - 优化版"""
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # Pitch Envelope
    freq_env = 50 + 180 * np.exp(-40 * t) # 稍微降低起始频率，更沉
    phase = np.cumsum(freq_env) / SR * 2 * np.pi
    sine_wave = np.sin(phase)

    # Amp Envelope
    amp_env = np.exp(-5 * t)
    body = sine_wave * amp_env

    # [修复] Click 瞬态：降低高频噪声，防止滋滋声
    click_noise = np.random.uniform(-0.5, 0.5, duration_samples) # 降低幅度
    click_env = np.exp(-100 * t) 
    
    # 强力低通滤波 Click
    cutoff = 800 + brightness * 2000
    sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
    click = signal.sosfilt(sos, click_noise) * click_env * 0.4

    mix = body + click * brightness
    mix = saturation(mix, drive=1.8) # 增加饱和度，让Kick更实

    return mix * velocity


def synth_snare_advanced(duration_samples, velocity, brightness):
    """TR-808 风格军鼓"""
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # Tone
    freq_tone = 180 * (1 + 0.05 * np.exp(-15 * t))
    tone_part = np.sin(np.cumsum(freq_tone) / SR * 2 * np.pi)
    tone_env = np.exp(-10 * t)
    tone = tone_part * tone_env

    # Noise (响弦)
    raw_noise = np.random.uniform(-1, 1, duration_samples)
    sos = signal.butter(2, [1000, 6000], 'bp', fs=SR, output='sos') # 提高带通频率，更脆
    snare_wires = signal.sosfilt(sos, raw_noise)
    
    noise_env = 0.6 * np.exp(-25 * t) + 0.4 * np.exp(-8 * t)
    noise = snare_wires * noise_env

    mix = tone * 0.5 + noise * (0.5 + brightness * 0.5)
    return mix * velocity


def synth_hihat_metallic(duration_samples, velocity, open_hat=False, brightness=0.5):
    """
    [修复] 金属质感 Hi-hat
    使用纯正弦波叠加，彻底消除电流声。
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # 1. 生成金属底音 (无电流声版)
    metal_base = generate_metallic_noise(duration_samples)

    # 2. 高通滤波
    cutoff = 7000 if not open_hat else 4000 # 提高Cutoff，声音更细
    cutoff += (brightness - 0.5) * 2000
    sos = signal.butter(4, cutoff, 'hp', fs=SR, output='sos')
    filtered = signal.sosfilt(sos, metal_base)

    # 3. 包络
    decay = 50 if not open_hat else 8
    env = np.exp(-decay * t)

    return filtered * env * velocity * 0.8 # 提高一点音量


def synth_tom_advanced(duration_samples, velocity, freq):
    t = np.linspace(0, duration_samples / SR, duration_samples)
    f_sweep = freq * (1 + 0.6 * np.exp(-18 * t))
    wave = np.sin(np.cumsum(f_sweep) / SR * 2 * np.pi)
    env = np.exp(-5 * t)
    wave = saturation(wave, drive=1.2)
    return wave * env * velocity


# ==========================================
# 4. 主渲染逻辑
# ==========================================

def midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling):
    load_drum_samples()

    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        print(f"MIDI Error: {e}")
        return None, None

    total_time = sum(msg.time for msg in mid) + 3.0
    total_samples = int(total_time * SR)
    if total_samples > SR * 300: total_samples = SR * 300

    mix_buffer = np.zeros(total_samples, dtype=np.float32)
    current_time = 0

    for msg in mid:
        current_time += msg.time
        start_sample = int(current_time * SR)
        if start_sample >= total_samples: break

        if msg.type == 'note_on' and msg.velocity > 0:
            note = msg.note
            vel = (msg.velocity / 127.0) ** pluck_pos

            sample_data = None

            # Kick
            if note in [35, 36]:
                sample_data = get_sample_processed('kick', vel)
                if sample_data is None:
                    sample_data = synth_kick_advanced(int(SR * 0.5), vel, brightness)

            # Snare
            elif note in [38, 40, 37]:
                sample_data = get_sample_processed('snare', vel)
                if sample_data is None:
                    sample_data = synth_snare_advanced(int(SR * 0.35), vel, brightness)

            # Hi-Hat
            elif note in [42, 44]:  # Closed
                sample_data = get_sample_processed('hihat', vel)
                if sample_data is None:
                    sample_data = synth_hihat_metallic(int(SR * 0.15), vel, open_hat=False, brightness=brightness)

            elif note in [46]:  # Open
                sample_data = get_sample_processed('hihat', vel)
                if sample_data is None:
                    sample_data = synth_hihat_metallic(int(SR * 0.8), vel, open_hat=True, brightness=brightness)

            # Toms
            elif note in [41, 43]:  # Low
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.6), vel, 85)
            elif note in [45, 47]:  # Mid
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.5), vel, 130)
            elif note in [48, 50]:  # High
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.4), vel, 190)

            # Cymbals
            elif note in [49, 57, 51, 59]:
                sample_data = get_sample_processed('crash', vel)
                if sample_data is None:
                    # 镲片使用长尾音的金属合成
                    sample_data = synth_hihat_metallic(int(SR * 2.5), vel * 0.8, open_hat=True, brightness=brightness)

            if sample_data is not None:
                end_sample = start_sample + len(sample_data)
                if end_sample > total_samples:
                    sample_data = sample_data[:total_samples - start_sample]
                    end_sample = total_samples

                mix_buffer[start_sample:end_sample] += sample_data

    # ==========================================
    # 5. 总线效果
    # ==========================================

    # 饱和
    if body_mix > 0.0:
        drive = 1.0 + body_mix * 1.5
        mix_buffer = saturation(mix_buffer, drive)

    # EQ
    if brightness > 0.6:
        sos = signal.butter(2, 5000, 'hp', fs=SR, output='sos')
        highs = signal.sosfilt(sos, mix_buffer) * (brightness - 0.6)
        mix_buffer += highs
    elif brightness < 0.4:
        sos = signal.butter(2, 300, 'lp', fs=SR, output='sos')
        lows = signal.sosfilt(sos, mix_buffer) * (0.4 - brightness)
        mix_buffer += lows

    # Reverb
    if reflection > 0.01:
        delay_samps = int(SR * 0.03)
        decay = 0.4
        reverb = np.zeros_like(mix_buffer)
        if len(mix_buffer) > delay_samps * 2:
            reverb[delay_samps:] += mix_buffer[:-delay_samps] * 0.5
            reverb[delay_samps * 2:] += mix_buffer[:-delay_samps * 2] * 0.25
        mix_buffer = mix_buffer * (1 - reflection * 0.4) + reverb * reflection

    # Limiter
    peak = np.max(np.abs(mix_buffer))
    target_peak = 0.95
    if peak > target_peak:
        mix_buffer = mix_buffer / peak * target_peak

    samples_int = (mix_buffer * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    return buf.getvalue(), mix_buffer
