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
    单根钢琴弦的物理模型
    
    参数:
    - string_num: 当前是第几根弦（0, 1, 2）
    - total_strings: 总共几根弦（1, 2, 3）
    """
    delay_samples = int(SR / frequency)
    if delay_samples < 2:
        delay_samples = 2
    
    output = np.zeros(n_samples, dtype=np.float32)
    
    # === 1. 琴槌击弦模型（真实物理） ===
    # 琴槌接触时间：约 1-4ms（频率越高，接触时间越短）
    contact_time = max(0.001, 0.004 - frequency / 2000.0)
    contact_samples = int(contact_time * SR)
    
    # 琴槌速度到位移的转换（非线性）
    # 钢琴的响应曲线接近 v^3（比吉他的 v^2 更陡）
    hammer_velocity = velocity ** 3.0
    
    # 多弦系统：每根弦的相位略有不同
    phase_offset = string_num * 0.05
    
    # 击弦位置（钢琴通常在弦长的 1/7 到 1/9 处）
    strike_position = 1.0 / 8.0
    strike_delay = int(delay_samples * strike_position)
    
    # 生成琴槌脉冲
    for i in range(contact_samples):
        t = i / contact_samples
        
        # 琴槌形状：快速上升 + 慢速回落
        # 使用 raised cosine 函数
        hammer_shape = (1.0 - np.cos(np.pi * t)) / 2.0
        
        # 应用非线性（毛毡的弹性）
        hammer_force = hammer_shape * (1.0 - hammer_shape * 0.3)
        
        output[i] = hammer_force * hammer_velocity
        
        # 反向脉冲（在击弦点产生）
        if i + strike_delay < n_samples:
            output[i + strike_delay] -= hammer_force * hammer_velocity * 0.4
    
    # 添加微小噪声（琴弦的微观不完美）
    for i in range(min(contact_samples * 2, n_samples)):
        output[i] += np.random.normal(0, 0.002) * velocity
    
    # === 2. 弦的传播和衰减 ===
    # 钢琴弦的衰减非常复杂，分为三个阶段
    
    # 基础衰减（与频率强相关）- 调整为更明亮
    if frequency < 100:
        # 低音弦：长、粗、缠绕，衰减极慢
        base_decay = 0.9998
    elif frequency < 500:
        # 中音弦：中等衰减（增加亮度）
        base_decay = 0.9997
    else:
        # 高音弦：短、细，但不要衰减太快（保持明亮）
        base_decay = 0.9995
    
    # 高频成分衰减更快（色散效应）- 减少这个效应，保持明亮
    inharmonicity = 0.00005 * (frequency / 1000.0)  # 降低一半
    
    # 低通滤波器系数（模拟弦的阻尼）- 提高系数，保留更多高频
    damping_coef = 0.6 + (frequency / 4186.0) * 0.35  # 提高基础值
    
    # Karplus-Strong 主循环
    for i in range(delay_samples, n_samples):
        # 读取延迟线
        s1 = output[i - delay_samples]
        s2 = output[i - delay_samples - 1] if i > delay_samples else 0.0
        
        # 低通滤波（能量守恒）
        filtered = s1 * damping_coef + s2 * (1.0 - damping_coef)
        
        # 非谐波成分（钢琴的金属质感）
        # 添加轻微的频率调制
        if i % (delay_samples * 2) == 0:
            filtered *= (1.0 - inharmonicity)
        
        # 应用衰减
        output[i] = filtered * base_decay
    
    return output


@jit(nopython=True, fastmath=True)
def soundboard_resonance(signal, frequency):
    """
    音板共鸣模拟（简化的模态合成）
    
    钢琴音板的特点：
    1. 有多个共振峰（模态）
    2. 低频共振峰在 100-200Hz
    3. 中频共振峰在 400-600Hz
    """
    n = len(signal)
    output = np.zeros(n, dtype=np.float32)
    
    # 主共振峰（根据音符频率调整）
    resonance_freq = frequency * 0.93
    
    # 二阶共振滤波器参数
    w = 2.0 * np.pi * resonance_freq / SR
    r = 0.98  # Q 值
    
    # 状态变量
    y1, y2 = 0.0, 0.0
    
    for i in range(n):
        # IIR 二阶共振器
        y0 = signal[i] + 2.0 * r * np.cos(w) * y1 - r * r * y2
        output[i] = y0
        y2 = y1
        y1 = y0
    
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


def piano_eq_mastering(audio_buffer):
    """
    钢琴专用母带 EQ（明亮版本）
    
    目标：
    1. 保留低频丰满感
    2. 大幅提升高频明亮度（解决闷音）
    3. 削减中频"木头味"
    """
    # 1. 温和的高通（只切极低频 25Hz）
    sos_hp = signal.butter(2, 25, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_hp, audio_buffer)
    
    # 2. 低频轻微提升（80-150Hz，温暖感）
    b_low, a_low = signal.iirpeak(110, 8, SR)
    low_boost = signal.lfilter(b_low, a_low, audio_buffer) * 0.1
    audio_buffer = audio_buffer + low_boost
    
    # 3. 中频大幅削减（400-800Hz，消除"闷"感）
    # 使用宽带陷波
    b_mid1, a_mid1 = signal.iirnotch(500, 15, SR)
    audio_buffer = signal.lfilter(b_mid1, a_mid1, audio_buffer)
    
    b_mid2, a_mid2 = signal.iirnotch(700, 15, SR)
    audio_buffer = signal.lfilter(b_mid2, a_mid2, audio_buffer)
    
    # 4. 高频大幅提升（2-6kHz，明亮感）
    # 临场感频段
    b_presence, a_presence = signal.iirpeak(3000, 10, SR)
    presence_boost = signal.lfilter(b_presence, a_presence, audio_buffer) * 0.4
    audio_buffer = audio_buffer + presence_boost
    
    # 空气感频段
    b_air, a_air = signal.iirpeak(5000, 8, SR)
    air_boost = signal.lfilter(b_air, a_air, audio_buffer) * 0.3
    audio_buffer = audio_buffer + air_boost
    
    # 5. 超高频提升（8-12kHz，"空气感"）
    sos_shelf = signal.butter(2, 8000, 'hp', fs=SR, output='sos')
    high_shelf = signal.sosfilt(sos_shelf, audio_buffer) * 0.2
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
    low_band = np.tanh(low_band * 1.1) / 1.1   # 极轻压缩
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
        agc_factor = 0.7   # 高复音适度衰减
    
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
        # 钢琴力度曲线比吉他更陡峭
        vel_curve = (velocity / 127.0) ** 2.5
        
        # 频率平衡（钢琴的低音不需要像吉他那样大幅削减）
        if freq < 100:
            freq_gain = 0.7  # 低音适度衰减
        elif freq < 300:
            freq_gain = 0.85
        else:
            freq_gain = 1.0
        
        # 增加基础音量（避免过小）
        final_velocity = vel_curve * freq_gain * agc_factor * 1.5  # 提高到 1.5 倍
        
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
        final_wave = combined * 0.7 + resonance * 0.3
        
        # === 包络（制音器） ===
        if not pedaled:
            # 模拟制音器的快速衰减
            damper_time = int(SR * 0.2)
            note_off = end - start
            
            if 0 < note_off < len(final_wave) - damper_time:
                fade = np.exp(-np.linspace(0, 5, damper_time))
                final_wave[note_off:note_off+damper_time] *= fade
                final_wave[note_off+damper_time:] = 0.0
        
        # 叠加到混音
        end_idx = min(start + len(final_wave), total_samples)
        mix_buffer[start:end_idx] += final_wave[:end_idx-start]
    
    # === 后处理链 ===
    print("   应用后处理...")
    
    # 1. 钢琴专用 EQ
    mix_buffer = piano_eq_mastering(mix_buffer)
    
    # 2. 多频段压缩
    mix_buffer = multiband_compressor(mix_buffer)
    
    # 3. 音乐厅混响
    if reflection > 0.01:
        # 钢琴需要更长的混响
        delays = [
            int(SR * 0.04),  # 早期反射
            int(SR * 0.09),  # 中期
            int(SR * 0.15),  # 后期
            int(SR * 0.23)   # 尾部
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
