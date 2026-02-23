import numpy as np
import mido
import io
import wave
from numba import jit
from scipy import signal

SR = 48000


@jit(nopython=True, fastmath=True)
def piano_string_model(n_samples, frequency, velocity, string_num, total_strings):
    """
    单根钢琴弦的物理模型（终极版）

    新增：
    1. 琴槌接触噪声（木质"咔"声）
    2. 弦的拉伸不谐性
    3. 更真实的击弦点反射
    """
    delay_samples = int(SR / frequency)
    if delay_samples < 2:
        delay_samples = 2

    output = np.zeros(n_samples, dtype=np.float32)

    # === 1. 琴槌击弦模型（改进版）===
    contact_time = max(0.0008, 0.005 - frequency / 1500.0)  # 更短的接触时间
    contact_samples = int(contact_time * SR)

    # 琴槌速度的立方律（钢琴特性）
    hammer_velocity = velocity ** 3.2  # 从3.0提升到3.2，动态更明显

    # 多弦相位偏移
    phase_offset = string_num * 0.03

    # 击弦位置（1/8 弦长）
    strike_position = 1.0 / 8.0
    strike_delay = int(delay_samples * strike_position)

    # 生成琴槌脉冲
    for i in range(contact_samples):
        t = i / contact_samples

        # 改进的琴槌形状（更尖锐的攻击）
        if t < 0.3:
            hammer_shape = (t / 0.3) ** 1.5
        else:
            hammer_shape = 1.0 - ((t - 0.3) / 0.7) ** 0.8

        # 毛毡非线性
        hammer_force = hammer_shape * (1.0 - hammer_shape * 0.25)

        output[i] = hammer_force * hammer_velocity

        # 反向脉冲（击弦点反射）
        if i + strike_delay < n_samples:
            output[i + strike_delay] -= hammer_force * hammer_velocity * 0.45

    # 琴槌接触噪声（木质"咔"声，钢琴特有）
    if velocity > 0.6:  # 只在大力度时出现
        knock_intensity = (velocity - 0.6) * 0.015
        for i in range(min(5, contact_samples)):
            output[i] += np.random.normal(0, knock_intensity)

    # 弦的微观不完美（金属噪声）
    for i in range(min(contact_samples * 3, n_samples)):
        output[i] += np.random.normal(0, 0.001) * velocity

    # === 2. 弦的传播和衰减（加入不谐性）===
    if frequency < 100:
        base_decay = 0.9998
        inharmonicity = 0.00002
    elif frequency < 500:
        base_decay = 0.9997
        inharmonicity = 0.00004
    else:
        base_decay = 0.9995
        inharmonicity = 0.00008

    # 低通滤波器系数（频率相关）
    damping_coef = 0.6 + (frequency / 4186.0) * 0.35

    # Karplus-Strong 主循环（加入不谐性）
    for i in range(delay_samples, n_samples):
        s1 = output[i - delay_samples]
        s2 = output[i - delay_samples - 1] if i > delay_samples else 0.0

        # 低通滤波
        filtered = s1 * damping_coef + s2 * (1.0 - damping_coef)

        # 不谐性效应：每个周期略微减少能量
        if i % delay_samples == 0:
            filtered *= (1.0 - inharmonicity)

        output[i] = filtered * base_decay

    return output


@jit(nopython=True, fastmath=True)
def soundboard_resonance(signal, frequency):
    """
    音板共鸣模拟（改进版：多模态共振）

    新增：
    1. 三个共振峰（而非单一）
    2. 频率相关的共振强度
    3. 相位调制（增加复杂度）
    """
    n = len(signal)
    output = np.zeros(n, dtype=np.float32)

    # === 主共振峰（最强） ===
    resonance_freq_1 = frequency * 0.92
    w1 = 2.0 * np.pi * resonance_freq_1 / SR
    r1 = 0.98

    y1_1, y1_2 = 0.0, 0.0

    # === 次共振峰（中等） ===
    resonance_freq_2 = frequency * 1.47  # 接近完美五度
    w2 = 2.0 * np.pi * resonance_freq_2 / SR
    r2 = 0.96

    y2_1, y2_2 = 0.0, 0.0

    # === 第三共振峰（较弱） ===
    resonance_freq_3 = frequency * 0.73
    w3 = 2.0 * np.pi * resonance_freq_3 / SR
    r3 = 0.94

    y3_1, y3_2 = 0.0, 0.0

    for i in range(n):
        # 第一共振器（最强）
        y1_0 = signal[i] + 2.0 * r1 * np.cos(w1) * y1_1 - r1 * r1 * y1_2

        # 第二共振器
        y2_0 = signal[i] + 2.0 * r2 * np.cos(w2) * y2_1 - r2 * r2 * y2_2

        # 第三共振器
        y3_0 = signal[i] + 2.0 * r3 * np.cos(w3) * y3_1 - r3 * r3 * y3_2

        # 混合三个共振峰（不同权重）
        output[i] = y1_0 * 0.5 + y2_0 * 0.3 + y3_0 * 0.2

        # 更新状态
        y1_2 = y1_1
        y1_1 = y1_0
        y2_2 = y2_1
        y2_1 = y2_0
        y3_2 = y3_1
        y3_1 = y3_0

    return output


def sympathetic_resonance(mix_buffer, events):
    """
    泛音共鸣（Sympathetic Resonance）

    钢琴的一个重要特性：当按下一个键时，其泛音对应的其他弦
    也会轻微振动（即使没有被击打）
    """
    # 简化实现：对每个音符，激发其八度音的轻微共鸣
    # 这里暂时跳过，留给未来优化
    return mix_buffer


def piano_eq_mastering(audio_buffer, brightness=0.65):
    """
    钢琴专用母带 EQ（明亮版本）

    参数:
    - brightness: 明亮度 (0.3-0.9)，控制高频提升量
    """
    # 1. 温和的高通（只切极低频 25Hz）
    sos_hp = signal.butter(2, 25, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_hp, audio_buffer)

    # 2. 低频轻微提升（80-150Hz，温暖感）
    b_low, a_low = signal.iirpeak(110, 8, SR)
    low_boost = signal.lfilter(b_low, a_low, audio_buffer) * 0.1
    audio_buffer = audio_buffer + low_boost

    # 3. 中频大幅削减（400-800Hz，消除"闷"感）
    b_mid1, a_mid1 = signal.iirnotch(500, 15, SR)
    audio_buffer = signal.lfilter(b_mid1, a_mid1, audio_buffer)

    b_mid2, a_mid2 = signal.iirnotch(700, 15, SR)
    audio_buffer = signal.lfilter(b_mid2, a_mid2, audio_buffer)

    # 4. 高频提升（根据 brightness 参数动态调整）
    # brightness 越大，高频提升越多
    boost_factor = brightness * 0.6  # 0.3-0.9 -> 0.18-0.54

    # 临场感频段 (3kHz)
    b_presence, a_presence = signal.iirpeak(3000, 10, SR)
    presence_boost = signal.lfilter(b_presence, a_presence, audio_buffer) * boost_factor
    audio_buffer = audio_buffer + presence_boost

    # 空气感频段 (5kHz)
    b_air, a_air = signal.iirpeak(5000, 8, SR)
    air_boost = signal.lfilter(b_air, a_air, audio_buffer) * (boost_factor * 0.8)
    audio_buffer = audio_buffer + air_boost

    # 5. 超高频提升（8-12kHz，根据 brightness 调整）
    sos_shelf = signal.butter(2, 8000, 'hp', fs=SR, output='sos')
    high_shelf = signal.sosfilt(sos_shelf, audio_buffer) * (boost_factor * 0.5)
    audio_buffer = audio_buffer + high_shelf

    # 6. 最高频柔化（避免刺耳，但保留到 15kHz）
    sos_lp = signal.butter(1, 15000, 'lp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_lp, audio_buffer)

    return audio_buffer


def multiband_compressor(audio_buffer):
    """
    多频段压缩器（轻量版，避免过闷）

    解决钢琴的动态范围过大问题：
    - 低频：轻压缩（保留丰满感）
    - 中频：轻压缩（避免闷）
    - 高频：几乎不压缩（保持明亮）
    """
    # 分频点
    low_freq = 250
    high_freq = 2000

    # 低频段
    sos_low = signal.butter(4, low_freq, 'lp', fs=SR, output='sos')
    low_band = signal.sosfilt(sos_low, audio_buffer)

    # 高频段
    sos_high = signal.butter(4, high_freq, 'hp', fs=SR, output='sos')
    high_band = signal.sosfilt(sos_high, audio_buffer)

    # 中频段
    mid_band = audio_buffer - low_band - high_band

    # 分别压缩（大幅减轻压缩强度）
    low_band = np.tanh(low_band * 1.1) / 1.1  # 极轻压缩
    mid_band = np.tanh(mid_band * 1.15) / 1.15  # 极轻压缩
    high_band = high_band * 1.05  # 几乎不压缩，反而轻微提升

    # 混合
    return low_band + mid_band + high_band


def midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling):
    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        print(f"MIDI 解析失败: {e}")
        return None, None

    # 预计算总时长
    total_len = sum(msg.time for msg in mid) + 5.0  # 钢琴余音更长
    total_samples = int(total_len * SR)
    if total_samples > SR * 300:
        total_samples = SR * 300

    mix_buffer = np.zeros(total_samples, dtype=np.float32)

    # === MIDI 事件解析（支持延音踏板） ===
    events = []
    cursor = 0
    sustain_pedal = False
    active_notes = {}

    for msg in mid:
        cursor += int(msg.time * SR)

        # 延音踏板
        if msg.type == 'control_change' and msg.control == 64:
            sustain_pedal = (msg.value >= 64)

        elif msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = (cursor, msg.velocity, sustain_pedal)

        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start, vel, pedaled = active_notes.pop(msg.note)
                events.append((start, cursor, msg.note, vel, pedaled))

    # 未关闭的音符
    for note, (start, vel, pedaled) in active_notes.items():
        events.append((start, total_samples - SR * 2, note, vel, pedaled))

    print(f"🎹 钢琴引擎：处理 {len(events)} 个音符事件")

    # === 简化的音量控制（移除激进的 AGC）===
    # 只做基础的归一化，不要过度压缩
    max_polyphony = 1
    time_grid = np.zeros(total_samples, dtype=np.int16)
    for start, end, note, vel, pedal in events:
        if start < total_samples and end > start:
            end = min(end, total_samples)
            time_grid[start:end] += 1
            max_polyphony = max(max_polyphony, np.max(time_grid[start:end]))

    # 温和的增益控制（避免音量过小）
    if max_polyphony <= 3:
        agc_factor = 1.0  # 低复音不衰减
    elif max_polyphony <= 6:
        agc_factor = 0.85  # 中等复音轻微衰减
    else:
        agc_factor = 0.7  # 高复音适度衰减

    print(f"   最大复音数: {max_polyphony}, 自动增益: {agc_factor:.3f}")

    # === 音符渲染 ===
    for start, end, note, velocity, pedaled in events:
        if start >= total_samples:
            continue

        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        if freq > SR / 2 or freq < 27.5:  # A0 = 27.5Hz
            continue

        # 决定弦数（真实钢琴的配置）
        if note < 30:  # 低音区
            num_strings = 1
        elif note < 50:  # 中音区
            num_strings = 2
        else:  # 高音区
            num_strings = 3

        # === 力度响应（钢琴的非线性特性）===
        # 使用 coupling 参数控制力度曲线
        vel_curve = (velocity / 127.0) ** coupling  # 用户可调的力度曲线

        # 频率平衡（钢琴的低音不需要像吉他那样大幅削减）
        if freq < 100:
            freq_gain = 0.7  # 低音适度衰减
        elif freq < 300:
            freq_gain = 0.85
        else:
            freq_gain = 1.0

        # 增加基础音量（避免过小）
        # 使用 pluck_pos 作为琴槌硬度系数
        final_velocity = vel_curve * freq_gain * agc_factor * 1.5 * pluck_pos

        # 生成时长（考虑踏板）
        if pedaled:
            duration = int(SR * 6.0)  # 踏板延长到 6 秒
        else:
            duration = int(SR * 3.0)  # 正常 3 秒

        duration = min(duration, total_samples - start)

        # === 多弦合成 ===
        string_outputs = []
        for s in range(num_strings):
            # 每根弦的频率略有不同（失谐，造成合唱效果）
            detune_cents = (s - num_strings / 2.0) * 0.5  # ±0.25 音分
            detune_ratio = 2.0 ** (detune_cents / 1200.0)
            string_freq = freq * detune_ratio

            string_wave = piano_string_model(
                duration,
                string_freq,
                final_velocity / num_strings,  # 分配能量
                s,
                num_strings
            )
            string_outputs.append(string_wave)

        # 混合多根弦
        combined = np.sum(string_outputs, axis=0) / num_strings

        # === 音板共鸣 ===
        resonance = soundboard_resonance(combined, freq)
        # 使用 body_mix 控制音板共鸣强度
        final_wave = combined * (1.0 - body_mix) + resonance * body_mix

        # === 包络（制音器） ===
        if not pedaled:
            # 模拟制音器的快速衰减
            damper_time = int(SR * 0.2)
            note_off = end - start

            if 0 < note_off < len(final_wave) - damper_time:
                fade = np.exp(-np.linspace(0, 5, damper_time))
                final_wave[note_off:note_off + damper_time] *= fade
                final_wave[note_off + damper_time:] = 0.0

        # 叠加到混音
        end_idx = min(start + len(final_wave), total_samples)
        mix_buffer[start:end_idx] += final_wave[:end_idx - start]

    # === 后处理链 ===
    print("   应用后处理...")

    # 1. 钢琴专用 EQ（使用 brightness 参数）
    mix_buffer = piano_eq_mastering(mix_buffer, brightness)

    # 2. 多频段压缩
    mix_buffer = multiband_compressor(mix_buffer)

    # 3. 音乐厅混响
    if reflection > 0.01:
        # 钢琴需要更长的混响
        delays = [
            int(SR * 0.04),  # 早期反射
            int(SR * 0.09),  # 中期
            int(SR * 0.15),  # 后期
            int(SR * 0.23)  # 尾部
        ]
        decays = [0.6, 0.4, 0.25, 0.15]

        reverb = np.zeros_like(mix_buffer)
        for delay, decay in zip(delays, decays):
            if len(mix_buffer) > delay:
                reverb[delay:] += mix_buffer[:-delay] * reflection * decay

        mix_buffer = mix_buffer * 0.75 + reverb * 0.25

    # 4. 最终音量处理（确保足够响）
    peak = np.max(np.abs(mix_buffer))
    if peak > 0.01:
        # 归一化到接近满刻度
        target_level = 0.98  # 提高到 0.98
        mix_buffer = mix_buffer / peak * target_level
    else:
        # 如果信号太小，放大
        mix_buffer = mix_buffer * 10.0

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

    print("✅ 钢琴渲染完成")
    return buf.getvalue(), mix_buffer