import numpy as np
import mido
import io
import wave
from numba import jit
from scipy import signal

SR = 48000


@jit(nopython=True, fastmath=True)
def karplus_strong_hifi(n_samples, delay_samples, velocity, brightness, decay_factor):
    """
    高保真 Karplus-Strong 算法

    关键改进：
    1. 平滑的激励信号（避免尖锐的点击声）
    2. 频率自适应的低通滤波
    3. 渐进式能量衰减（避免突然截止）
    """
    output = np.zeros(n_samples, dtype=np.float32)

    # === 1. 激励信号生成 ===
    # 使用带限噪声，避免高频啸叫
    burst_len = delay_samples
    if burst_len > n_samples:
        burst_len = n_samples

    # 三角滤波窗口，平滑激励
    for i in range(burst_len):
        # 白噪声
        noise = np.random.uniform(-0.5, 0.5)

        # 三角窗口（渐入渐出）
        if i < burst_len // 3:
            window = i / (burst_len // 3)
        elif i > 2 * burst_len // 3:
            window = (burst_len - i) / (burst_len // 3)
        else:
            window = 1.0

        # 混合亮度：高 brightness = 保留更多高频
        if i > 0:
            smoothed = noise * brightness + output[i - 1] * (1.0 - brightness) * 0.3
        else:
            smoothed = noise

        output[i] = smoothed * window * velocity

    # === 2. 物理反馈循环 ===
    # 频率自适应衰减：低音衰减慢，高音衰减快
    freq = SR / delay_samples

    # 基础衰减（高保真）
    base_decay = 0.9990

    # 频率相关的额外衰减
    freq_decay = min(freq / 1000.0, 1.0) * 0.001

    # 用户可控的衰减因子
    user_decay = decay_factor * 0.002

    final_decay = base_decay - freq_decay - user_decay
    final_decay = max(final_decay, 0.985)  # 防止过快衰减
    final_decay = min(final_decay, 0.999)  # 防止不衰减

    # 低通滤波器系数（Karplus-Strong 核心）
    # brightness 控制高频保留
    alpha = 0.5 + brightness * 0.3

    # 主循环
    for i in range(delay_samples, n_samples):
        # 延迟线读取
        delayed_1 = output[i - delay_samples]
        delayed_2 = output[i - delay_samples - 1] if i > delay_samples else 0.0

        # 低通滤波（平滑）
        filtered = delayed_1 * alpha + delayed_2 * (1.0 - alpha)

        # 应用衰减
        output[i] = filtered * final_decay

    return output


@jit(nopython=True, fastmath=True)
def soft_clipper(x, threshold=0.8):
    """
    平滑软削波器（比 tanh 更温和）

    使用分段函数：
    - |x| < threshold: 线性通过
    - |x| >= threshold: 三次函数平滑限制
    """
    if abs(x) < threshold:
        return x
    else:
        sign = 1.0 if x > 0 else -1.0
        excess = abs(x) - threshold
        # 三次曲线平滑过渡到 1.0
        clipped = threshold + excess / (1.0 + excess * excess)
        return sign * clipped


def adaptive_limiter(buffer, target_peak=0.95):
    """
    自适应限制器（Look-ahead）

    关键：提前检测峰值，平滑降低增益，避免硬削波
    """
    # 计算包络（RMS）
    window_size = 2048
    rms = np.sqrt(np.convolve(buffer ** 2, np.ones(window_size) / window_size, mode='same'))

    # 峰值检测
    peak = np.max(np.abs(buffer))

    if peak > target_peak:
        # 计算增益削减
        gain_reduction = target_peak / peak

        # 平滑应用增益（避免突变）
        buffer = buffer * gain_reduction

    # 软削波作为最后防线
    for i in range(len(buffer)):
        buffer[i] = soft_clipper(buffer[i], target_peak)

    return buffer


def spectral_balance_eq(audio_buffer):
    """
    频谱平衡均衡器

    解决问题：
    1. 低频轰鸣（< 80Hz）
    2. 泥泞的中低频（200-400Hz）
    3. 刺耳的高频（> 8kHz）
    """
    # 1. 高通滤波：切除 80Hz 以下
    sos_hp = signal.butter(4, 80, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_hp, audio_buffer)

    # 2. 中低频轻微衰减（减少"箱体感"）
    # 使用陷波滤波器在 300Hz
    b_notch, a_notch = signal.iirnotch(300, 30, SR)
    notch_signal = signal.lfilter(b_notch, a_notch, audio_buffer)
    audio_buffer = audio_buffer * 0.85 + notch_signal * 0.15

    # 3. 临场感提升：3-5kHz 轻微提升
    b_peak, a_peak = signal.iirpeak(4000, 30, SR)
    presence = signal.lfilter(b_peak, a_peak, audio_buffer) * 0.15
    audio_buffer = audio_buffer + presence

    # 4. 高频柔化：8kHz 以上轻微滚降
    sos_lp = signal.butter(2, 10000, 'lp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_lp, audio_buffer)

    return audio_buffer


def midi_to_audio(midi_stream, brightness, pluck_position, body_mix, reflection, coupling):
    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        print(f"MIDI 解析失败: {e}")
        return None, None

    # 预计算总时长
    total_len = sum(msg.time for msg in mid) + 3.0
    total_samples = int(total_len * SR)
    if total_samples > SR * 300:
        total_samples = SR * 300

    mix_buffer = np.zeros(total_samples, dtype=np.float32)

    # === MIDI 事件解析 ===
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

    # 未关闭的音符
    for note, (start, vel) in active_notes.items():
        events.append((start, total_samples - SR, note, vel))

    print(f"🎸 吉他引擎：处理 {len(events)} 个音符事件")

    # === 关键：动态范围压缩预算 ===
    # 统计同时发声的最大音符数，用于自动增益控制
    max_polyphony = 1
    time_grid = np.zeros(total_samples, dtype=np.int16)
    for start, end, note, vel in events:
        if start < total_samples and end > start:
            end = min(end, total_samples)
            time_grid[start:end] += 1
            max_polyphony = max(max_polyphony, np.max(time_grid[start:end]))

    # 自动增益控制因子
    agc_factor = 1.0 / np.sqrt(max_polyphony)
    print(f"   最大复音数: {max_polyphony}, 自动增益: {agc_factor:.3f}")

    # === 音符渲染 ===
    for start, end, note, velocity in events:
        if start >= total_samples:
            continue

        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        if freq > SR / 2 or freq < 30:
            continue

        delay_samples = int(SR / freq)
        if delay_samples < 2:
            continue

        # === 改进的音量曲线 ===
        # 1. 力度响应（接近真实吉他）
        vel_curve = (velocity / 127.0) ** 1.8  # 1.8 次方更自然

        # 2. 频率平衡（大幅削减低音，消除金属刺耳声）
        if freq < 150:
            freq_gain = 0.00  # 极低音（消除刺耳）
        elif freq < 250:
            freq_gain = 0.25  # 低音大幅衰减
        elif freq < 500:
            freq_gain = 0.65  # 中低音适度衰减
        else:
            freq_gain = 1.0  # 高音保持

        # 3. 自动增益补偿
        final_velocity = vel_curve * freq_gain * agc_factor * 0.8

        # 生成音符
        duration = (end - start) + int(SR * 0.5)  # 留 0.5 秒余音
        duration = min(duration, total_samples - start)

        wave_snippet = karplus_strong_hifi(
            duration,
            delay_samples,
            final_velocity,
            brightness,
            coupling
        )

        # === 释放包络（ADSR 的 R） ===
        release_time = int(SR * 0.15)
        note_off = end - start

        if note_off > 0 and note_off < len(wave_snippet):
            if note_off + release_time < len(wave_snippet):
                # 平滑释放
                fade = np.linspace(1.0, 0.0, release_time)
                wave_snippet[note_off:note_off + release_time] *= fade
                wave_snippet[note_off + release_time:] = 0.0

        # 叠加到混音缓冲
        end_idx = min(start + len(wave_snippet), total_samples)
        mix_buffer[start:end_idx] += wave_snippet[:end_idx - start]

    # === 后处理链 ===
    print("   应用后处理...")

    # 1. 频谱平衡
    mix_buffer = spectral_balance_eq(mix_buffer)

    # 2. 空间混响
    if reflection > 0.01:
        delay_samples = int(SR * 0.08)
        if len(mix_buffer) > delay_samples:
            # 多重延迟线（更丰富的混响）
            reverb = np.zeros_like(mix_buffer)
            reverb[delay_samples:] += mix_buffer[:-delay_samples] * reflection * 0.5

            delay2 = int(SR * 0.12)
            if len(mix_buffer) > delay2:
                reverb[delay2:] += mix_buffer[:-delay2] * reflection * 0.3

            mix_buffer = mix_buffer * 0.8 + reverb * 0.2

    # 3. 自适应限制器
    mix_buffer = adaptive_limiter(mix_buffer, target_peak=0.93)

    # 4. 最终归一化
    peak = np.max(np.abs(mix_buffer))
    if peak > 0.01:
        mix_buffer = mix_buffer / peak * 0.95

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

    print("✅ 吉他渲染完成")
    return buf.getvalue(), mix_buffer

