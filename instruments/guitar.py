import numpy as np
import mido
import io
import wave
from numba import jit
from scipy import signal  # 引入科学计算库进行专业滤波

SR = 48000


@jit(nopython=True)
def karplus_strong_c(n_samples, delay_samples, velocity, brightness, decay_stretch, damping):
    """
    KS 核心算法 (High-Fidelity Ver.)
    """
    output = np.zeros(n_samples, dtype=np.float32)

    # 1. 激励信号 (Excitation)
    # 使用带通滤波的噪声，去除极低频和极高频的杂质
    burst_len = delay_samples
    if burst_len > n_samples: burst_len = n_samples

    # 动态调整激励的“冲击感”
    prev = 0.0
    for i in range(burst_len):
        white = np.random.uniform(-1.0, 1.0)
        # 混合：根据 brightness 决定是闷声还是亮声
        # 低通滤波平滑
        val = white * brightness + prev * (1.0 - brightness)
        output[i] = val * velocity
        prev = val

    # 2. 物理反馈循环
    # 动态计算 alpha，防止低频能量无限堆积
    # 高频(delay_samples小)需要更高的 alpha，低频(delay_samples大)需要稍低的 alpha
    freq_damp = 1.0 - (delay_samples / 10000.0)  # 简单的频率相关阻尼
    alpha = 0.995 * freq_damp + (decay_stretch * 0.005)

    # 强行限制反馈系数，杜绝自激啸叫
    if alpha > 0.996: alpha = 0.996

    # 环路滤波器系数
    # 模拟琴弦的刚性：brightness 越低，高频损失越快
    S = (1.0 - brightness) * 0.5
    w1 = 0.5 + S
    w2 = 0.5 - S

    # 预读
    idx = delay_samples
    while idx < n_samples:
        # 线性插值读取（比之前的整数读取更平滑，减少量化噪声）
        # 这里为了性能简化为两点平均
        s1 = output[idx - delay_samples]
        s2 = output[idx - delay_samples - 1]

        # 核心低通
        new_val = (s1 * w1 + s2 * w2) * alpha

        output[idx] = new_val
        idx += 1

    return output


def mastering_chain(audio_buffer, instrument_type='guitar'):
    # 1. 高通滤波 (High-Pass / Low Cut) - 解决“炸低音”的核心
    # 吉他切掉 75Hz 以下，钢琴切掉 40Hz 以下
    cutoff = 75 if instrument_type == 'guitar' else 40
    sos = signal.butter(4, cutoff, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos, audio_buffer)

    # 2. 临场感提升 (Presence Boost) - 在 3kHz 附近提升一点，增加清晰度
    # 使用 Peaking Filter
    b, a = signal.iirpeak(3000, 1.0, fs=SR)
    presence = signal.lfilter(b, a, audio_buffer) * 0.2
    audio_buffer += presence

    # 3. 动态压缩/饱和 (Saturation)
    # 使用 x / sqrt(1 + x^2) 曲线，比 tanh 更平滑，极大保留动态
    # 先应用增益 (Make-up Gain)
    gain = 1.5
    audio_buffer = audio_buffer * gain
    audio_buffer = audio_buffer / np.sqrt(1 + audio_buffer ** 2)

    # 4. 最终硬限制 (Safety Limiter)
    max_peak = np.max(np.abs(audio_buffer))
    target_peak = 0.95
    if max_peak > target_peak:
        audio_buffer = audio_buffer * (target_peak / max_peak)

    return audio_buffer


def midi_to_audio(midi_stream, brightness, pluck_position, body_mix, reflection, coupling):
    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        return None, None

    total_len = sum(msg.time for msg in mid) + 3.0
    total_samples = int(total_len * SR)
    if total_samples > SR * 300: total_samples = SR * 300

    mix_buffer = np.zeros(total_samples, dtype=np.float32)

    # MIDI 解析
    events = []
    cursor = 0
    active_notes = {}

    for msg in mid:
        cursor += int(msg.time * SR)
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = (cursor, msg.velocity)
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start, vel = active_notes.pop(msg.note)
                events.append((start, cursor, msg.note, vel))

    for note, (start, vel) in active_notes.items():
        events.append((start, total_samples - SR, note, vel))

    print(f"Guitar Pro: Processing {len(events)} notes...")

    for start, end, note, velocity in events:
        if start >= total_samples: continue

        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        if freq > SR / 2 or freq < 20: continue

        delay_samples = int(SR / freq)
        if delay_samples < 2: continue

        # === 核心修复：基于频率的增益补偿 (Frequency Key Tracking) ===
        # 低音(Freq小) -> 增益大幅降低
        # 高音(Freq大) -> 增益保持

        freq_gain = (freq / 400.0) ** 0.6  # 0.6次方让曲线平滑一些
        if freq_gain > 1.2: freq_gain = 1.2
        if freq_gain < 0.4: freq_gain = 0.4

        # 力度曲线优化 (平方律)
        real_velocity = (velocity / 127.0) ** 2.0

        # 最终输入振幅
        input_amp = real_velocity * freq_gain

        length = (end - start) + int(SR * 0.3)
        if start + length > total_samples: length = total_samples - start

        # 生成
        wave_snippet = karplus_strong_c(length, delay_samples, input_amp, brightness, coupling * 100, 0.5)

        # Release 包络
        release_len = int(SR * 0.1)
        if len(wave_snippet) > release_len:
            off_idx = end - start
            if 0 < off_idx < len(wave_snippet) - release_len:
                fade = np.linspace(1.0, 0.0, release_len)
                wave_snippet[off_idx:off_idx + release_len] *= fade
                wave_snippet[off_idx + release_len:] = 0.0

        end_idx = start + len(wave_snippet)
        if end_idx > total_samples:
            end_idx = total_samples
            wave_snippet = wave_snippet[:end_idx - start]

        mix_buffer[start:end_idx] += wave_snippet

    # === 后处理链 ===

    # 1. 简单的延迟混响
    if reflection > 0:
        d1 = int(SR * 0.08)
        # 使用衰减系数衰减 mix_buffer 本身，避免爆音
        mix_buffer *= 0.8
        if total_samples > d1:
            mix_buffer[d1:] += mix_buffer[:-d1] * reflection * 0.5

    # 2. 调用母带处理 (这里包含了 High-Pass Filter)
    mix_buffer = mastering_chain(mix_buffer, 'guitar')

    samples_int = (mix_buffer * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    return buf.getvalue(), mix_buffer