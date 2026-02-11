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
    return np.tanh(x * drive) / np.tanh(drive)


def generate_metallic_noise(n_samples):
    """
    生成金属噪声 (TR-808 风格)
    通过多个非整数倍频率的方波叠加，制造镲片的金属质感
    """
    t = np.linspace(0, n_samples / SR, n_samples)
    # TR-808 的经典频率比率
    freqs = [200, 300, 450, 600, 750, 900]
    noise = np.zeros(n_samples)
    for f in freqs:
        # 使用方波产生丰富的高频谐波
        noise += np.sign(np.sin(2 * np.pi * f * t))
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
        # print("⚠️ 未找到 drum_samples，启用 TR-808 模拟合成引擎。")
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

                    # 简单的重采样（如果不是48k，为了性能暂不处理高精度重采样）
                    # 实际使用建议确保素材就是 48k

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
    """
    智能采样选择 + 实时 DSP 处理 (Humanize)
    """
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

    # --- Humanize: 动态音高抖动 (Pitch Jitter) ---
    # 真实鼓皮每次打击的张力有微小变化
    # 通过重采样实现微小的音高偏移 (+- 30 cents)
    pitch_shift = np.random.uniform(0.98, 1.02)
    if pitch_shift != 1.0 and len(audio) > 100:
        # 简单的线性插值重采样
        indices = np.arange(0, len(audio), pitch_shift)
        indices = indices[indices < len(audio) - 1]
        audio = np.interp(indices, np.arange(len(audio)), audio)

    # 动态增益
    gain = 0.2 + velocity * 0.8
    return audio * gain


# ==========================================
# 3. 高级模拟合成引擎 (Analog Modeling Synth)
# ==========================================

def synth_kick_advanced(duration_samples, velocity, brightness):
    """
    TR-909 风格底鼓合成
    包含：Pitch Envelope (Thump), Click (Attack), Saturation (Body)
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # 1. 音高包络 (Pitch Sweep) - 决定打击感 (Punch)
    # 从 250Hz 极速下降到 50Hz
    freq_env = 50 + 200 * np.exp(-40 * t)
    phase = np.cumsum(freq_env) / SR * 2 * np.pi
    sine_wave = np.sin(phase)

    # 2. 幅度包络 (Amp Envelope)
    amp_env = np.exp(-6 * t)
    body = sine_wave * amp_env

    # 3. 瞬态点击 (Click/Beater) - 决定清晰度
    # 短促的滤波噪声
    click_noise = np.random.uniform(-1, 1, duration_samples)
    click_env = np.exp(-80 * t)  # 极短

    # 低通滤波点击声，随 brightness 变化
    cutoff = 1000 + brightness * 4000
    sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
    click = signal.sosfilt(sos, click_noise) * click_env * 0.5

    # 4. 混合与饱和
    mix = body + click * brightness

    # 加入过载失真 (Saturation) 让底鼓更肥
    mix = saturation(mix, drive=1.5)

    return mix * velocity


def synth_snare_advanced(duration_samples, velocity, brightness):
    """
    TR-808 风格军鼓合成
    包含：Tonal (鼓皮) + Noise (响弦)
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # 1. 鼓皮音 (Tonal)
    # 频率在 180Hz - 240Hz 之间微降
    freq_tone = 180 * (1 + 0.1 * np.exp(-15 * t))
    tone_part = np.sin(np.cumsum(freq_tone) / SR * 2 * np.pi)
    tone_env = np.exp(-12 * t)  # 衰减快
    tone = tone_part * tone_env

    # 2. 响弦音 (Noise) - 军鼓的灵魂
    raw_noise = np.random.uniform(-1, 1, duration_samples)

    # 响弦不是全频带白噪，而是集中在中高频
    # 使用带通滤波模拟鼓腔
    sos = signal.butter(2, [800, 5000], 'bp', fs=SR, output='sos')
    snare_wires = signal.sosfilt(sos, raw_noise)

    # 响弦有两个衰减阶段：瞬态爆裂 + 尾音
    noise_env = 0.7 * np.exp(-25 * t) + 0.3 * np.exp(-5 * t)
    noise = snare_wires * noise_env

    # 3. 混合
    # Brightness 增加响弦比例
    mix = tone * 0.6 + noise * (0.6 + brightness * 0.4)

    return mix * velocity


def synth_hihat_metallic(duration_samples, velocity, open_hat=False, brightness=0.5):
    """
    金属质感 Hi-hat 合成
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # 1. 生成金属底噪
    metal_noise = generate_metallic_noise(duration_samples)

    # 2. 高通滤波 - 去除浑浊的低频
    # 闭镲切得更高
    cutoff = 5000 if not open_hat else 3000
    # 根据 brightness 调整
    cutoff += (brightness - 0.5) * 2000
    sos = signal.butter(4, cutoff, 'hp', fs=SR, output='sos')
    filtered = signal.sosfilt(sos, metal_noise)

    # 3. 包络
    decay = 40 if not open_hat else 6  # 开镲延音长
    env = np.exp(-decay * t)

    return filtered * env * velocity * 0.6


def synth_tom_advanced(duration_samples, velocity, freq):
    """
    通鼓合成：带明显 Pitch Bend 的正弦波
    """
    t = np.linspace(0, duration_samples / SR, duration_samples)

    # 音高下潜：从 1.5倍频率 降到 1.0倍
    f_sweep = freq * (1 + 0.5 * np.exp(-15 * t))
    wave = np.sin(np.cumsum(f_sweep) / SR * 2 * np.pi)

    env = np.exp(-4 * t)
    # 加一点饱和
    wave = saturation(wave, drive=1.2)

    return wave * env * velocity


# ==========================================
# 4. 主渲染逻辑
# ==========================================

def midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling):
    """
    params:
    - brightness: 鼓的明亮度 (EQ / Filter)
    - pluck_pos: 力度响应曲线 (Velocity Curve)
    - body_mix: 压缩/饱和度 (Compression/Saturation)
    - reflection: 房间混响 (Reverb)
    - coupling: 总体增益 (Master Gain)
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

    # Open Hi-hat 窒音机制 (Choke Group)
    # 当遇到闭镲或脚踩镲时，需要切断之前的开镲声音
    last_open_hat_end = 0

    for msg in mid:
        current_time += msg.time
        start_sample = int(current_time * SR)
        if start_sample >= total_samples: break

        if msg.type == 'note_on' and msg.velocity > 0:
            note = msg.note
            # 力度曲线调整
            vel = (msg.velocity / 127.0) ** pluck_pos

            sample_data = None

            # --- 路由逻辑 ---

            # 1. Kick (底鼓)
            if note in [35, 36]:
                sample_data = get_sample_processed('kick', vel)
                if sample_data is None:
                    sample_data = synth_kick_advanced(int(SR * 0.5), vel, brightness)

            # 2. Snare (军鼓)
            elif note in [38, 40, 37]:
                sample_data = get_sample_processed('snare', vel)
                if sample_data is None:
                    sample_data = synth_snare_advanced(int(SR * 0.3), vel, brightness)

            # 3. Hi-Hat (踩镲)
            elif note in [42, 44]:  # Closed
                sample_data = get_sample_processed('hihat', vel)
                if sample_data is None:
                    sample_data = synth_hihat_metallic(int(SR * 0.15), vel, open_hat=False, brightness=brightness)

                # 窒音逻辑：如果你正在播放开镲，这里应该切断它
                # (由于我们是线性叠加到 mix_buffer，很难撤销已经加进去的数值
                # 这里的简化处理是：不处理撤销，但确保当前 Closed 够响)

            elif note in [46]:  # Open
                sample_data = get_sample_processed('hihat', vel)  # 简单共用
                if sample_data is None:
                    sample_data = synth_hihat_metallic(int(SR * 0.8), vel, open_hat=True, brightness=brightness)

            # 4. Toms (通鼓)
            elif note in [41, 43]:  # Low
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.5), vel, 85)
            elif note in [45, 47]:  # Mid
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.45), vel, 130)
            elif note in [48, 50]:  # High
                sample_data = get_sample_processed('tom', vel)
                if sample_data is None: sample_data = synth_tom_advanced(int(SR * 0.4), vel, 190)

            # 5. Cymbals (镲片)
            elif note in [49, 57, 51, 59]:
                sample_data = get_sample_processed('crash', vel)
                if sample_data is None:
                    # 镲片合成
                    sample_data = synth_hihat_metallic(int(SR * 2.0), vel * 0.7, open_hat=True, brightness=brightness)

            # 叠加音频
            if sample_data is not None:
                end_sample = start_sample + len(sample_data)
                if end_sample > total_samples:
                    sample_data = sample_data[:total_samples - start_sample]
                    end_sample = total_samples

                mix_buffer[start_sample:end_sample] += sample_data

    # ==========================================
    # 5. 总线效果链 (Bus Effects) - 让鼓组粘合在一起
    # ==========================================

    # 1. 饱和压缩 (Saturation/Compression)
    # 这里的 body_mix 控制“胶水感” (Glue)
    if body_mix > 0.0:
        drive = 1.0 + body_mix * 2.0  # 1.0 ~ 3.0
        mix_buffer = saturation(mix_buffer, drive)

    # 2. EQ 塑形 (根据 Brightness)
    # Brightness > 0.5 提升高频，< 0.5 提升低频
    if brightness > 0.6:
        # High Shelf Boost
        sos = signal.butter(2, 5000, 'hp', fs=SR, output='sos')
        highs = signal.sosfilt(sos, mix_buffer) * (brightness - 0.6)
        mix_buffer += highs
    elif brightness < 0.4:
        # Low Shelf Boost
        sos = signal.butter(2, 200, 'lp', fs=SR, output='sos')
        lows = signal.sosfilt(sos, mix_buffer) * (0.4 - brightness)
        mix_buffer += lows

    # 3. 房间混响 (Reverb)
    if reflection > 0.01:
        # 鼓组需要短混响 (Plate/Room)
        delay_samps = int(SR * 0.04)  # 40ms 预延迟
        decay = 0.4

        # 简单的 FIR 混响模拟
        reverb = np.zeros_like(mix_buffer)
        # 制造两个回声点
        if len(mix_buffer) > delay_samps * 2:
            reverb[delay_samps:] += mix_buffer[:-delay_samps] * 0.6
            reverb[delay_samps * 2:] += mix_buffer[:-delay_samps * 2] * 0.3

        mix_buffer = mix_buffer * (1 - reflection * 0.5) + reverb * reflection

    # 4. 限制器 (Limiter)
    # 确保不爆音
    peak = np.max(np.abs(mix_buffer))
    target_peak = 0.95
    if peak > target_peak:
        mix_buffer = mix_buffer / peak * target_peak

    # 转 WAV
    samples_int = (mix_buffer * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    return buf.getvalue(), mix_buffer
