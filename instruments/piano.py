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
    """
    delay_samples = int(SR / frequency)
    if delay_samples < 2:
        delay_samples = 2
    
    output = np.zeros(n_samples, dtype=np.float32)
    
    # 琴槌模型
    contact_time = max(0.001, 0.004 - frequency / 2000.0)
    contact_samples = int(contact_time * SR)
    
    # [修复1] 降低琴槌对力度的敏感度，防止小力度没声音
    hammer_velocity = velocity ** 1.5 
    
    phase_offset = string_num * 0.05
    strike_position = 1.0 / 8.0
    strike_delay = int(delay_samples * strike_position)
    
    for i in range(contact_samples):
        t = i / contact_samples
        hammer_shape = (1.0 - np.cos(np.pi * t)) / 2.0
        hammer_force = hammer_shape * (1.0 - hammer_shape * 0.3)
        output[i] = hammer_force * hammer_velocity
        
        if i + strike_delay < n_samples:
            output[i + strike_delay] -= hammer_force * hammer_velocity * 0.4
    
    # 增加一点随机底噪，保证物理模型始终有能量输入
    noise_burst = np.random.normal(0, 0.005, min(contact_samples * 2, n_samples)) * velocity
    for i in range(len(noise_burst)):
        output[i] += noise_burst[i]
    
    # 弦衰减参数
    if frequency < 100:
        base_decay = 0.9998
    elif frequency < 500:
        base_decay = 0.9997
    else:
        base_decay = 0.9995
    
    inharmonicity = 0.00005 * (frequency / 1000.0)
    damping_coef = 0.6 + (frequency / 4186.0) * 0.35
    
    for i in range(delay_samples, n_samples):
        s1 = output[i - delay_samples]
        s2 = output[i - delay_samples - 1] if i > delay_samples else 0.0
        
        filtered = s1 * damping_coef + s2 * (1.0 - damping_coef)
        
        if i % (delay_samples * 2) == 0:
            filtered *= (1.0 - inharmonicity)
        
        output[i] = filtered * base_decay
    
    return output


@jit(nopython=True, fastmath=True)
def soundboard_resonance(signal, frequency):
    """音板共鸣"""
    n = len(signal)
    output = np.zeros(n, dtype=np.float32)
    resonance_freq = frequency * 0.93
    w = 2.0 * np.pi * resonance_freq / SR
    r = 0.98
    y1, y2 = 0.0, 0.0
    
    for i in range(n):
        y0 = signal[i] + 2.0 * r * np.cos(w) * y1 - r * r * y2
        output[i] = y0
        y2 = y1
        y1 = y0
    
    return output


def piano_eq_mastering(audio_buffer):
    """钢琴专用 EQ"""
    sos_hp = signal.butter(2, 25, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_hp, audio_buffer)
    
    b_low, a_low = signal.iirpeak(110, 8, SR)
    low_boost = signal.lfilter(b_low, a_low, audio_buffer) * 0.1
    audio_buffer = audio_buffer + low_boost
    
    b_mid1, a_mid1 = signal.iirnotch(500, 15, SR)
    audio_buffer = signal.lfilter(b_mid1, a_mid1, audio_buffer)
    
    b_mid2, a_mid2 = signal.iirnotch(700, 15, SR)
    audio_buffer = signal.lfilter(b_mid2, a_mid2, audio_buffer)
    
    b_presence, a_presence = signal.iirpeak(3000, 10, SR)
    presence_boost = signal.lfilter(b_presence, a_presence, audio_buffer) * 0.4
    audio_buffer = audio_buffer + presence_boost
    
    sos_shelf = signal.butter(2, 8000, 'hp', fs=SR, output='sos')
    high_shelf = signal.sosfilt(sos_shelf, audio_buffer) * 0.2
    audio_buffer = audio_buffer + high_shelf
    
    return audio_buffer


def multiband_compressor(audio_buffer):
    """简单压缩"""
    # 移除过于复杂的逻辑，只做简单的压限防止过载
    return np.tanh(audio_buffer * 1.5)


def midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling):
    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        print(f"MIDI 解析失败: {e}")
        return None, None
    
    total_len = sum(msg.time for msg in mid) + 5.0
    total_samples = int(total_len * SR)
    if total_samples > SR * 300:
        total_samples = SR * 300
    
    mix_buffer = np.zeros(total_samples, dtype=np.float32)
    
    events = []
    cursor = 0
    sustain_pedal = False
    active_notes = {}
    
    for msg in mid:
        cursor += int(msg.time * SR)
        
        if msg.type == 'control_change' and msg.control == 64:
            sustain_pedal = (msg.value >= 64)
        
        elif msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = (cursor, msg.velocity, sustain_pedal)
        
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start, vel, pedaled = active_notes.pop(msg.note)
                events.append((start, cursor, msg.note, vel, pedaled))
    
    for note, (start, vel, pedaled) in active_notes.items():
        events.append((start, total_samples - SR * 2, note, vel, pedaled))
    
    print(f"🎹 钢琴引擎：处理 {len(events)} 个音符事件")
    
    # [修复2] 移除复杂的自动增益计算 (AGC)
    # 不再根据最大复音数来惩罚音量，而是使用固定增益
    # 这样音符少的曲子和音符多的曲子基础音量差距不会太大
    fixed_gain = 0.4 

    for start, end, note, velocity, pedaled in events:
        if start >= total_samples:
            continue
        
        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        if freq > SR / 2 or freq < 27.5:
            continue
        
        if note < 30: num_strings = 1
        elif note < 50: num_strings = 2
        else: num_strings = 3
        
        # [修复3] 拉平力度曲线
        # 之前是 velocity ** 2.5，导致小力度(velocity<60)几乎无声
        # 现在改为 1.2，接近线性，保证所有力度的音符都能被听到
        vel_curve = (velocity / 127.0) ** 1.2
        
        final_velocity = vel_curve * fixed_gain
        
        if pedaled:
            duration = int(SR * 6.0)
        else:
            duration = int(SR * 3.0)
        
        duration = min(duration, total_samples - start)
        
        string_outputs = []
        for s in range(num_strings):
            detune_cents = (s - num_strings / 2.0) * 0.5
            detune_ratio = 2.0 ** (detune_cents / 1200.0)
            string_freq = freq * detune_ratio
            
            string_wave = piano_string_model(
                duration,
                string_freq,
                final_velocity / num_strings,
                s,
                num_strings
            )
            string_outputs.append(string_wave)
        
        combined = np.sum(string_outputs, axis=0) / num_strings
        
        resonance = soundboard_resonance(combined, freq)
        final_wave = combined * 0.7 + resonance * 0.3
        
        if not pedaled:
            damper_time = int(SR * 0.2)
            note_off = end - start
            
            if 0 < note_off < len(final_wave) - damper_time:
                fade = np.exp(-np.linspace(0, 5, damper_time))
                final_wave[note_off:note_off+damper_time] *= fade
                final_wave[note_off+damper_time:] = 0.0
        
        end_idx = min(start + len(final_wave), total_samples)
        mix_buffer[start:end_idx] += final_wave[:end_idx-start]
    
    # === 后处理链 ===
    mix_buffer = piano_eq_mastering(mix_buffer)
    mix_buffer = multiband_compressor(mix_buffer)
    
    if reflection > 0.01:
        delays = [int(SR * 0.04), int(SR * 0.09)]
        decays = [0.6, 0.4]
        reverb = np.zeros_like(mix_buffer)
        for delay, decay in zip(delays, decays):
            if len(mix_buffer) > delay:
                reverb[delay:] += mix_buffer[:-delay] * reflection * decay
        mix_buffer = mix_buffer * 0.8 + reverb * 0.2
    
    # [修复4] 强制音量归一化 (Peak Normalization)
    # 无论之前的计算结果是 0.01 还是 10.0
    # 这里统一把最大峰值拉到 0.95，确保输出音量始终饱满
    peak = np.max(np.abs(mix_buffer))
    if peak > 0:
        # 使用 tanh 进行软限制，然后归一化
        # 先放大一点让 tanh 产生饱和感 (Volume Boost)
        mix_buffer = np.tanh(mix_buffer * 2.0) 
        new_peak = np.max(np.abs(mix_buffer))
        if new_peak > 0:
            mix_buffer = mix_buffer / new_peak * 0.95
    
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
