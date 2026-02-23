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
    """磁带饱和"""
    if drive <= 0: return x
    return np.tanh(x * drive)


def generate_metallic_noise(n_samples):
    """
    金属噪声 v2：正弦波叠加 + 高频噪声混合
    纯正弦波太"干净"，真实镲片有大量非谐波噪声成分。
    混入滤波白噪声让音色更接近真实铜合金振动。
    """
    t = np.linspace(0, n_samples / SR, n_samples)
    # TR-808 经典非谐波频率
    freqs = [263, 400, 421, 474, 587, 845]
    tonal = np.zeros(n_samples)
    for f in freqs:
        tonal += np.sin(2 * np.pi * f * t)
    tonal /= len(freqs)

    # 混入高频带通噪声 (6-14kHz)，模拟真实镲片的"沙沙"质感
    noise = np.random.uniform(-1, 1, n_samples)
    sos = signal.butter(2, [6000, 14000], 'bp', fs=SR, output='sos')
    noise = signal.sosfilt(sos, noise)

    # 6:4 混合 — 保留金属音调感的同时加入噪声真实感
    return tonal * 0.6 + noise * 0.4


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
# 3. 合成引擎 (逐个重写)
# ==========================================

def synth_kick(duration_samples, velocity, brightness):
    """
    底鼓 v3 — 三层结构：Sub + Body + Click
    关键改进：
    - Body 衰减从 exp(-5t) → exp(-8t)：更紧凑，不拖泥带水
    - Click 更短更亮 (exp(-200t))：穿透力强
    - 新增独立 Sub 层 (45Hz)：低频厚度和 Body 分离，不互相干扰
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # Layer 1: Body (快速扫频正弦波 — 底鼓的"音调"部分)
    freq_env = 50 + 200 * np.exp(-45 * t)   # 从 250Hz 快速扫到 50Hz
    phase = np.cumsum(freq_env) / SR * 2 * np.pi
    body = np.sin(phase) * np.exp(-8 * t)    # 紧凑衰减

    # Layer 2: Sub (低频正弦，提供"胸口共振"的重量感)
    sub = np.sin(2 * np.pi * 45 * t) * np.exp(-4.5 * t) * 0.35

    # Layer 3: Click (极短噪声脉冲 — 穿透混音的关键)
    click_noise = np.random.uniform(-0.5, 0.5, duration_samples)
    click_env = np.exp(-200 * t)   # 极短：~5ms
    cutoff = 1500 + brightness * 3000
    sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
    click = signal.sosfilt(sos, click_noise) * click_env * 0.6

    mix = body + sub + click * (0.4 + brightness * 0.6)
    mix = saturation(mix, drive=2.0)

    return mix * velocity


def synth_snare(duration_samples, velocity, brightness):
    """
    军鼓 v3 — 三层结构：Tone + Wires + Click
    关键改进：
    - Tone 提高到 200Hz，衰减加快 (exp(-14t))
    - 响弦噪声带拓宽到 800-12kHz (原来只到 6kHz，缺少 "嘶" 的高频)
    - 新增独立 Click 层：1-5kHz 的短脉冲，提供 "啪" 的瞬态
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # Layer 1: Tone (军鼓膜振动)
    freq_tone = 200 * (1 + 0.05 * np.exp(-18 * t))
    tone = np.sin(np.cumsum(freq_tone) / SR * 2 * np.pi) * np.exp(-14 * t)

    # Layer 2: Wires (响弦 — 军鼓最重要的特征)
    raw_noise = np.random.uniform(-1, 1, duration_samples)
    sos = signal.butter(2, [800, 12000], 'bp', fs=SR, output='sos')
    wires = signal.sosfilt(sos, raw_noise)
    # 双段包络：快速 attack (啪) + 慢速尾巴 (沙沙)
    wire_env = 0.6 * np.exp(-28 * t) + 0.4 * np.exp(-9 * t)
    wires = wires * wire_env

    # Layer 3: Click (木棒打击膜面的瞬态)
    click_noise = np.random.uniform(-0.6, 0.6, duration_samples)
    click_env = np.exp(-180 * t)
    sos_click = signal.butter(2, [1000, 5000], 'bp', fs=SR, output='sos')
    click = signal.sosfilt(sos_click, click_noise) * click_env * 0.4

    mix = tone * 0.35 + wires * (0.45 + brightness * 0.35) + click * (0.5 + brightness * 0.5)
    return mix * velocity


def synth_hihat(duration_samples, velocity, open_hat=False, brightness=0.5):
    """
    Hi-hat v3 — 金属音调 + 噪声混合
    关键改进：
    - 不再是纯正弦波（太"电子合成器"了）
    - 混入高频带通噪声，模拟真实铜合金的非谐波振动
    - Closed hat 衰减加快，更 "tik tik" 而不是 "嗡嗡"
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    metal = generate_metallic_noise(duration_samples)

    # 高通滤波（closed hat 更高）
    cutoff = 8000 if not open_hat else 5000
    cutoff += (brightness - 0.5) * 2000
    cutoff = max(cutoff, 3000)
    sos = signal.butter(4, cutoff, 'hp', fs=SR, output='sos')
    filtered = signal.sosfilt(sos, metal)

    # 衰减包络
    decay = 60 if not open_hat else 7    # closed hat 更快衰减
    env = np.exp(-decay * t)

    return filtered * env * velocity * 0.85


def synth_tom(duration_samples, velocity, freq):
    """
    通鼓 v3 — Body + Attack 双层
    新增：短噪声脉冲作为 attack 瞬态
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # Body (扫频正弦)
    f_sweep = freq * (1 + 0.7 * np.exp(-20 * t))
    body = np.sin(np.cumsum(f_sweep) / SR * 2 * np.pi) * np.exp(-5.5 * t)

    # Attack (短噪声脉冲)
    attack_noise = np.random.uniform(-0.4, 0.4, duration_samples)
    attack_env = np.exp(-100 * t)
    sos = signal.butter(2, [200, 4000], 'bp', fs=SR, output='sos')
    attack = signal.sosfilt(sos, attack_noise) * attack_env * 0.3

    mix = body + attack
    mix = saturation(mix, drive=1.3)
    return mix * velocity


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
                    sample_data = synth_kick(int(SR * 0.45), vel, brightness)

            # Snare
            elif note in [38, 40, 37]:
                sample_data = get_sample_processed('snare', vel)
                if sample_data is None:
                    sample_data = synth_snare(int(SR * 0.30), vel, brightness)

            # Hi-Hat Closed
            elif note in [42, 44]:
                sample_data = get_sample_processed('hihat', vel)
                if sample_data is None:
                    sample_data = synth_hihat(int(SR * 0.12), vel, False, brightness)

            # Hi-Hat Open
            elif note in [46]:
                sample_data = get_sample_processed('hihat', vel)
                if sample_data is None:
                    sample_data = synth_hihat(int(SR * 0.7), vel, True, brightness)

            # Toms
            elif note in [41, 43]:
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom(int(SR * 0.55), vel, 85)
            elif note in [45, 47]:
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom(int(SR * 0.45), vel, 130)
            elif note in [48, 50]:
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom(int(SR * 0.35), vel, 190)

            # Cymbals
            elif note in [49, 57, 51, 59]:
                sample_data = get_sample_processed('crash', vel)
                if sample_data is None:
                    sample_data = synth_hihat(int(SR * 2.5), vel * 0.8, True, brightness)

            if sample_data is not None:
                end_sample = start_sample + len(sample_data)
                if end_sample > total_samples:
                    sample_data = sample_data[:total_samples - start_sample]
                    end_sample = total_samples
                mix_buffer[start_sample:end_sample] += sample_data

    # ==========================================
    # 5. 鼓组总线处理 (Bus Processing)
    # ==========================================

    # --- 5a. 鼓组 EQ ---
    # 切掉 30Hz 以下的 rumble (保护低频头部空间)
    sos_hp = signal.butter(2, 30, 'hp', fs=SR, output='sos')
    mix_buffer = signal.sosfilt(sos_hp, mix_buffer)

    # 200-400Hz 轻微衰减 (减少底鼓和通鼓的"箱子味")
    b_mud, a_mud = signal.iirnotch(300, 6, SR)
    mud_cut = signal.lfilter(b_mud, a_mud, mix_buffer)
    mix_buffer = mix_buffer * 0.7 + mud_cut * 0.3

    # 3-5kHz 提升 (军鼓的"啪"和踩镲的"嘶")
    if brightness > 0.3:
        b_snap, a_snap = signal.iirpeak(4000, 8, SR)
        snap_boost = signal.lfilter(b_snap, a_snap, mix_buffer) * (brightness * 0.2)
        mix_buffer = mix_buffer + snap_boost

    # 8kHz+ 空气感 (让踩镲和镲片更通透)
    sos_air = signal.butter(1, 8000, 'hp', fs=SR, output='sos')
    air = signal.sosfilt(sos_air, mix_buffer) * 0.1
    mix_buffer = mix_buffer + air

    # --- 5b. 饱和 (用户控制) ---
    if body_mix > 0.0:
        drive = 1.0 + body_mix * 1.5
        mix_buffer = saturation(mix_buffer, drive)

    # --- 5c. 并行压缩 (Parallel Compression / New York Compression) ---
    # 核心原理：把压缩后的信号混回原始信号
    # 效果：保留瞬态(attack)的同时把小信号(sustain/ghost notes)推上来 → 鼓更"厚"更"在"
    compressed = mix_buffer.copy()
    comp_threshold = 0.3
    comp_ratio = 3.0   # 3:1 压缩比
    for i in range(len(compressed)):
        level = abs(compressed[i])
        if level > comp_threshold:
            excess = level - comp_threshold
            new_level = comp_threshold + excess / comp_ratio
            compressed[i] = compressed[i] * (new_level / level)

    # 混合：原始 70% + 压缩 30%
    mix_buffer = mix_buffer * 0.7 + compressed * 0.3

    # --- 5d. 混响 (偏短/偏干 — 鼓混响不宜过长) ---
    if reflection > 0.01:
        d1 = int(SR * 0.025)   # 25ms (早期反射)
        d2 = int(SR * 0.050)   # 50ms
        reverb = np.zeros_like(mix_buffer)
        if len(mix_buffer) > d2:
            reverb[d1:] += mix_buffer[:-d1] * 0.4
            reverb[d2:] += mix_buffer[:-d2] * 0.2
        # 混响过低频滤波 (不要让底鼓的低频也进混响)
        sos_rv_hp = signal.butter(2, 400, 'hp', fs=SR, output='sos')
        reverb = signal.sosfilt(sos_rv_hp, reverb)
        mix_buffer = mix_buffer * (1 - reflection * 0.3) + reverb * reflection

    # --- 5e. Limiter ---
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