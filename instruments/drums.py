import numpy as np
import mido
import io
import wave
from scipy import signal

SR = 48000


def generate_kick(duration_samples, velocity):
    """底鼓（Kick）- 低频冲击"""
    t = np.arange(duration_samples) / SR

    # 频率从 150Hz 快速下扫到 50Hz
    freq_sweep = 150 * np.exp(-t * 15) + 50
    phase = np.cumsum(2 * np.pi * freq_sweep / SR)

    # 正弦波 + 轻微失真
    kick = np.sin(phase) * velocity
    kick = np.tanh(kick * 2.5)

    # 包络：快速衰减
    envelope = np.exp(-t * 18)

    # 添加点击声（拍击感）
    click = np.random.randn(duration_samples) * 0.15 * velocity
    click *= np.exp(-t * 80)

    return (kick * envelope + click) * 0.8


def generate_snare(duration_samples, velocity):
    """军鼓（Snare）- 白噪声 + 音调"""
    t = np.arange(duration_samples) / SR

    # 音调部分（200Hz）
    tone = np.sin(2 * np.pi * 200 * t) * velocity * 0.4

    # 噪声部分（响弦）
    noise = np.random.randn(duration_samples) * velocity * 0.6

    # 高通滤波噪声（去掉低频轰鸣）
    sos_hp = signal.butter(4, 300, 'hp', fs=SR, output='sos')
    noise = signal.sosfilt(sos_hp, noise)

    # 包络
    envelope = np.exp(-t * 25)

    return (tone + noise) * envelope


def generate_hihat(duration_samples, velocity, closed=True):
    """踩镲（Hi-Hat）"""
    t = np.arange(duration_samples) / SR

    # 高频噪声
    noise = np.random.randn(duration_samples) * velocity

    # 高通滤波（只保留高频）
    sos_hp = signal.butter(4, 6000, 'hp', fs=SR, output='sos')
    noise = signal.sosfilt(sos_hp, noise)

    # Closed hi-hat: 短促
    # Open hi-hat: 较长
    if closed:
        envelope = np.exp(-t * 60)
    else:
        envelope = np.exp(-t * 12)

    return noise * envelope * 0.5


def generate_tom(duration_samples, velocity, pitch='mid'):
    """通鼓（Tom）"""
    t = np.arange(duration_samples) / SR

    # 不同音高的通鼓
    if pitch == 'low':
        freq = 80
    elif pitch == 'mid':
        freq = 120
    else:  # high
        freq = 180

    # 音调下扫
    freq_sweep = freq * np.exp(-t * 8)
    phase = np.cumsum(2 * np.pi * freq_sweep / SR)

    tom = np.sin(phase) * velocity

    # 包络
    envelope = np.exp(-t * 12)

    return tom * envelope * 0.7


def generate_crash(duration_samples, velocity):
    """镲片（Crash Cymbal）"""
    t = np.arange(duration_samples) / SR

    # 复杂的高频噪声
    noise = np.random.randn(duration_samples) * velocity

    # 带通滤波（2-12kHz）
    sos_bp = signal.butter(2, [2000, 12000], 'bp', fs=SR, output='sos')
    noise = signal.sosfilt(sos_bp, noise)

    # 长延音
    envelope = np.exp(-t * 3)

    return noise * envelope * 0.6


def midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling):
    """
    架子鼓 MIDI 渲染

    参数说明（架子鼓不使用这些参数，保留接口统一）：
    - brightness: 未使用
    - pluck_pos: 未使用
    - body_mix: 未使用
    - reflection: 房间混响
    - coupling: 未使用
    """
    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        print(f"MIDI 解析失败: {e}")
        return None, None

    total_len = sum(msg.time for msg in mid) + 3.0
    total_samples = int(total_len * SR)
    if total_samples > SR * 300:
        total_samples = SR * 300

    mix_buffer = np.zeros(total_samples, dtype=np.float32)

    # MIDI 事件解析
    events = []
    cursor = 0

    for msg in mid:
        cursor += int(msg.time * SR)
        if msg.type == 'note_on' and msg.velocity > 0:
            events.append((cursor, msg.note, msg.velocity))

    print(f"🥁 架子鼓引擎：处理 {len(events)} 个打击事件")

    # 渲染打击乐器
    for start, note, velocity in events:
        if start >= total_samples:
            continue

        vel_norm = velocity / 127.0

        # GM Drum Map（通用MIDI鼓组映射）
        if note == 36:  # Bass Drum (Kick)
            sample = generate_kick(int(SR * 0.4), vel_norm)
        elif note in [38, 40]:  # Snare
            sample = generate_snare(int(SR * 0.25), vel_norm)
        elif note in [42, 44]:  # Closed Hi-Hat
            sample = generate_hihat(int(SR * 0.08), vel_norm, closed=True)
        elif note in [46]:  # Open Hi-Hat
            sample = generate_hihat(int(SR * 0.4), vel_norm, closed=False)
        elif note in [45, 47, 48, 50]:  # Toms
            if note == 45:
                sample = generate_tom(int(SR * 0.5), vel_norm, 'low')
            elif note in [47, 48]:
                sample = generate_tom(int(SR * 0.4), vel_norm, 'mid')
            else:
                sample = generate_tom(int(SR * 0.35), vel_norm, 'high')
        elif note in [49, 55, 57]:  # Crash Cymbal
            sample = generate_crash(int(SR * 2.0), vel_norm)
        elif note == 51:  # Ride Cymbal
            sample = generate_hihat(int(SR * 0.6), vel_norm * 0.8, closed=False)
        else:
            # 其他音符用简单的噪声
            sample = np.random.randn(int(SR * 0.1)) * vel_norm * 0.3

        # 叠加
        end_idx = min(start + len(sample), total_samples)
        sample_len = end_idx - start
        if sample_len > 0:
            mix_buffer[start:end_idx] += sample[:sample_len]

    # 后处理
    print("   应用后处理...")

    # 1. 压缩器（鼓组需要强压缩）
    threshold = 0.6
    ratio = 4.0
    for i in range(len(mix_buffer)):
        if abs(mix_buffer[i]) > threshold:
            sign = 1.0 if mix_buffer[i] > 0 else -1.0
            excess = abs(mix_buffer[i]) - threshold
            mix_buffer[i] = sign * (threshold + excess / ratio)

    # 2. 房间混响
    if reflection > 0.01:
        delay_time = int(SR * 0.05)
        if len(mix_buffer) > delay_time:
            reverb = np.zeros_like(mix_buffer)
            reverb[delay_time:] += mix_buffer[:-delay_time] * reflection * 0.3
            mix_buffer = mix_buffer * 0.9 + reverb * 0.1

    # 3. 归一化
    peak = np.max(np.abs(mix_buffer))
    if peak > 0.01:
        mix_buffer = mix_buffer / peak * 0.96

    # 转换为 WAV
    samples_int = (mix_buffer * 32767).astype(np.int16)

    buf = io.BytesIO()
    try:
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(samples_int.tobytes())
    except Exception as e:
        print(f"WAV 写入失败: {e}")
        return None, None

    print("✅ 架子鼓渲染完成")
    return buf.getvalue(), mix_buffer