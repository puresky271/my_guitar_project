import numpy as np
import mido
import io
import wave
import os
import glob
import random
from scipy import signal

SR = 48000

SAMPLE_CACHE = {}
SAMPLES_LOADED = False


# ==========================================
# 1. DSP 工具箱
# ==========================================

def saturation(x, drive=1.0):
    """磁带/电子管饱和"""
    if drive <= 0: return x
    return np.tanh(x * drive)


def generate_metallic_noise(n_samples):
    """金属噪声 (纯正弦波叠加，无电流声)"""
    t = np.linspace(0, n_samples / SR, n_samples)
    freqs = [263, 400, 421, 474, 587, 845]
    noise = np.zeros(n_samples)
    for f in freqs:
        noise += np.sin(2 * np.pi * f * t)
    return noise / len(freqs)


def transient_shaper(buf, attack_gain=1.5, sustain_gain=1.0):
    """
    瞬态整形器 (Transient Shaper)
    增强打击乐的 Attack "扎感"，减弱尾巴。
    朋克鼓的核心：每一击都要砸到脸上。
    """
    env = np.abs(buf)

    # 快速攻击包络
    fast_env = np.zeros_like(env)
    fast_coeff = 0.01
    slow_coeff = 0.0001
    for i in range(1, len(env)):
        if env[i] > fast_env[i - 1]:
            fast_env[i] = fast_env[i - 1] + fast_coeff * (env[i] - fast_env[i - 1])
        else:
            fast_env[i] = fast_env[i - 1] + slow_coeff * (env[i] - fast_env[i - 1])

    # 慢速包络
    slow_env = np.zeros_like(env)
    for i in range(1, len(env)):
        slow_env[i] = slow_env[i - 1] + 0.00005 * (env[i] - slow_env[i - 1])

    # 瞬态 mask
    transient_mask = np.clip(fast_env - slow_env, 0, None)
    peak_t = np.max(transient_mask)
    if peak_t > 1e-10:
        transient_mask = transient_mask / peak_t

    gain = transient_mask * attack_gain + (1.0 - transient_mask) * sustain_gain
    return buf * gain


# ==========================================
# 2. 采样加载器
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

        for fpath in glob.glob(os.path.join(cat_dir, "*.wav")):
            try:
                with wave.open(fpath, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    dtype = np.int16 if wf.getsampwidth() == 2 else np.uint8
                    raw = np.frombuffer(frames, dtype=dtype)
                    if wf.getsampwidth() == 2:
                        audio = raw.astype(np.float32) / 32768.0
                    else:
                        audio = (raw.astype(np.float32) - 128.0) / 128.0

                    fname = os.path.basename(fpath).lower()
                    intensity = 'medium'
                    if 'hard' in fname or 'loud' in fname: intensity = 'hard'
                    elif 'soft' in fname or 'quiet' in fname: intensity = 'soft'
                    SAMPLE_CACHE[cat].append({'data': audio, 'intensity': intensity})
            except:
                pass

    SAMPLES_LOADED = True


def get_sample_processed(category, velocity):
    if category not in SAMPLE_CACHE or not SAMPLE_CACHE[category]:
        return None

    candidates = SAMPLE_CACHE[category]
    target = 'hard' if velocity > 0.85 else ('soft' if velocity < 0.4 else 'medium')
    matches = [s for s in candidates if s['intensity'] == target]
    if not matches: matches = candidates

    selected = random.choice(matches)
    audio = selected['data'].copy()

    pitch_shift = np.random.uniform(0.995, 1.005)
    if pitch_shift != 1.0 and len(audio) > 100:
        indices = np.arange(0, len(audio), pitch_shift)
        indices = indices[indices < len(audio) - 1]
        audio = np.interp(indices, np.arange(len(audio)), audio)

    return audio * (0.3 + velocity * 0.7)


# ==========================================
# 3. 改进合成引擎
# ==========================================

def synth_kick_advanced(duration_samples, velocity, brightness):
    """
    底鼓 v3: 更紧凑的 sub、更锐利的 click
    朋克底鼓的核心：快速扫频 + 短包络 = "拳头感"
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # Pitch Envelope (更陡峭的扫频 → 更"拳")
    freq_env = 48 + 200 * np.exp(-50 * t)
    phase = np.cumsum(freq_env) / SR * 2 * np.pi
    body = np.sin(phase)

    # Body Envelope (更紧 → 不拖泥带水)
    amp_env = np.exp(-6.5 * t)
    body = body * amp_env

    # Click 瞬态 (极短的噪声冲击)
    click_noise = np.random.uniform(-0.4, 0.4, duration_samples)
    click_env = np.exp(-150 * t)
    cutoff = 1200 + brightness * 2500
    sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
    click = signal.sosfilt(sos, click_noise) * click_env * 0.5

    # Sub 低频 (50Hz 正弦，增加重量感)
    sub = np.sin(2 * np.pi * 50 * t) * np.exp(-4 * t) * 0.3

    mix = body + click * brightness + sub
    mix = saturation(mix, drive=2.0)

    return mix * velocity


def synth_snare_advanced(duration_samples, velocity, brightness):
    """
    军鼓 v3: 更脆的 attack、双段衰减响弦
    朋克军鼓：高频爆炸 + 快速衰减 = "啪"
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # Tone (200Hz 起，略带扫频)
    freq_tone = 200 * (1 + 0.06 * np.exp(-20 * t))
    tone = np.sin(np.cumsum(freq_tone) / SR * 2 * np.pi) * np.exp(-12 * t)

    # 响弦噪声 (更宽的带通)
    raw_noise = np.random.uniform(-1, 1, duration_samples)
    sos = signal.butter(2, [800, 8000], 'bp', fs=SR, output='sos')
    wires = signal.sosfilt(sos, raw_noise)

    # 双段衰减 (快速瞬态 + 慢速尾巴)
    wire_env = 0.65 * np.exp(-30 * t) + 0.35 * np.exp(-10 * t)
    wires = wires * wire_env

    # 瞬态 Click (用来穿透混合)
    click = np.random.uniform(-0.5, 0.5, duration_samples) * np.exp(-200 * t) * 0.3
    sos_click = signal.butter(2, 5000, 'lp', fs=SR, output='sos')
    click = signal.sosfilt(sos_click, click)

    mix = tone * 0.4 + wires * (0.5 + brightness * 0.4) + click * brightness
    return mix * velocity


def synth_hihat_metallic(duration_samples, velocity, open_hat=False, brightness=0.5):
    """Hi-hat (正弦波叠加，无电流声)"""
    t = np.linspace(0, duration_samples / SR, duration_samples)

    metal_base = generate_metallic_noise(duration_samples)

    cutoff = 7500 if not open_hat else 4500
    cutoff += (brightness - 0.5) * 2000
    sos = signal.butter(4, cutoff, 'hp', fs=SR, output='sos')
    filtered = signal.sosfilt(sos, metal_base)

    decay = 55 if not open_hat else 7
    env = np.exp(-decay * t)

    return filtered * env * velocity * 0.85


def synth_tom_advanced(duration_samples, velocity, freq):
    """通鼓 (带打击瞬态)"""
    t = np.linspace(0, duration_samples / SR, duration_samples)
    f_sweep = freq * (1 + 0.7 * np.exp(-20 * t))
    body = np.sin(np.cumsum(f_sweep) / SR * 2 * np.pi) * np.exp(-5.5 * t)

    # 打击噪声
    attack = np.random.uniform(-0.3, 0.3, duration_samples) * np.exp(-80 * t) * 0.2

    mix = body + attack
    mix = saturation(mix, drive=1.3)
    return mix * velocity


# ==========================================
# 4. 主渲染逻辑
# ==========================================

def midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling,
                  punk_mode=False):
    """
    punk_mode: True=瞬态增强+中频推高+更干混响 (合奏调用时传入)
    向后兼容：默认 False，不影响独奏调用。
    """
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

            # Hi-Hat Closed
            elif note in [42, 44]:
                sample_data = get_sample_processed('hihat', vel)
                if sample_data is None:
                    sample_data = synth_hihat_metallic(int(SR * 0.15), vel, False, brightness)

            # Hi-Hat Open
            elif note in [46]:
                sample_data = get_sample_processed('hihat', vel)
                if sample_data is None:
                    sample_data = synth_hihat_metallic(int(SR * 0.8), vel, True, brightness)

            # Toms
            elif note in [41, 43]:
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.6), vel, 85)
            elif note in [45, 47]:
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.5), vel, 130)
            elif note in [48, 50]:
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.4), vel, 190)

            # Cymbals
            elif note in [49, 57, 51, 59]:
                sample_data = get_sample_processed('crash', vel)
                if sample_data is None:
                    sample_data = synth_hihat_metallic(int(SR * 2.5), vel * 0.8, True, brightness)

            if sample_data is not None:
                end_sample = start_sample + len(sample_data)
                if end_sample > total_samples:
                    sample_data = sample_data[:total_samples - start_sample]
                    end_sample = total_samples
                mix_buffer[start_sample:end_sample] += sample_data

    # ==========================================
    # 5. 总线效果
    # ==========================================

    # 瞬态整形 (朋克模式下更猛)
    if punk_mode:
        mix_buffer = transient_shaper(mix_buffer, attack_gain=1.8, sustain_gain=0.85)
        print("   🥁 鼓组瞬态增强已启用")

    # 饱和 (朋克模式下加大 drive)
    if body_mix > 0.0:
        drive = 1.0 + body_mix * (2.5 if punk_mode else 1.5)
        mix_buffer = saturation(mix_buffer, drive)

    # EQ
    if brightness > 0.6:
        sos = signal.butter(2, 5000, 'hp', fs=SR, output='sos')
        mix_buffer += signal.sosfilt(sos, mix_buffer) * (brightness - 0.6)
    elif brightness < 0.4:
        sos = signal.butter(2, 300, 'lp', fs=SR, output='sos')
        mix_buffer += signal.sosfilt(sos, mix_buffer) * (0.4 - brightness)

    # 朋克模式：中频推高 (让鼓穿透吉他墙)
    if punk_mode:
        b_mid, a_mid = signal.iirpeak(2000, 4, SR)
        mix_buffer += signal.lfilter(b_mid, a_mid, mix_buffer) * 0.15

    # Reverb (朋克模式下混响更短更干)
    rev_amount = reflection * (0.4 if punk_mode else 1.0)
    if rev_amount > 0.01:
        d = int(SR * 0.03)
        if len(mix_buffer) > d * 2:
            reverb = np.zeros_like(mix_buffer)
            reverb[d:] += mix_buffer[:-d] * 0.5
            reverb[d * 2:] += mix_buffer[:-d * 2] * 0.25
            mix_buffer = mix_buffer * (1 - rev_amount * 0.4) + reverb * rev_amount

    # Limiter
    peak = np.max(np.abs(mix_buffer))
    if peak > 0.95:
        mix_buffer = mix_buffer / peak * 0.95

    samples_int = (mix_buffer * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    return buf.getvalue(), mix_buffer