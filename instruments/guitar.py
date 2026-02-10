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
    高保真 Karplus-Strong 算法（终极版）
    
    新增：
    1. 弦张力非线性（大振幅时频率上扬）
    2. 更真实的激励信号（三角形而非噪声）
    3. 动态阻尼（振幅大时阻尼大）
    """
    output = np.zeros(n_samples, dtype=np.float32)
    
    # === 1. 激励信号生成（改进的三角波 + 噪声混合）===
    burst_len = delay_samples
    if burst_len > n_samples:
        burst_len = n_samples
    
    # 使用三角波而非纯噪声（更接近真实拨弦）
    for i in range(burst_len):
        # 三角波形状
        if i < burst_len // 2:
            triangle = (i / (burst_len // 2)) * 2.0 - 1.0
        else:
            triangle = 1.0 - ((i - burst_len // 2) / (burst_len // 2)) * 2.0
        
        # 混合少量噪声
        noise = np.random.uniform(-0.2, 0.2)
        
        # 窗口函数
        if i < burst_len // 4:
            window = i / (burst_len // 4)
        elif i > 3 * burst_len // 4:
            window = (burst_len - i) / (burst_len // 4)
        else:
            window = 1.0
        
        # 亮度控制（高 brightness = 保留更多高频）
        if i > 0:
            smoothed = triangle * brightness + output[i-1] * (1.0 - brightness) * 0.2
        else:
            smoothed = triangle
        
        output[i] = (smoothed * 0.8 + noise * 0.2) * window * velocity
    
    # === 2. 物理反馈循环（加入非线性）===
    freq = SR / delay_samples
    
    # 基础衰减
    base_decay = 0.9992
    freq_decay = min(freq / 1200.0, 1.0) * 0.0008
    user_decay = decay_factor * 0.002
    final_decay = base_decay - freq_decay - user_decay
    final_decay = max(final_decay, 0.988)
    final_decay = min(final_decay, 0.9995)
    
    # 低通滤波器系数
    alpha = 0.5 + brightness * 0.35
    
    # 主循环（加入非线性效果）
    for i in range(delay_samples, n_samples):
        delayed_1 = output[i - delay_samples]
        delayed_2 = output[i - delay_samples - 1] if i > delay_samples else 0.0
        
        # 低通滤波
        filtered = delayed_1 * alpha + delayed_2 * (1.0 - alpha)
        
        # 弦张力非线性：大振幅时产生轻微的频率上扬（类似真实吉他）
        amplitude = abs(filtered)
        if amplitude > 0.3:
            tension_factor = 1.0 + (amplitude - 0.3) * 0.02
            filtered *= tension_factor
        
        # 动态阻尼：振幅越大，阻尼越大（能量守恒）
        dynamic_decay = final_decay * (1.0 - amplitude * 0.01)
        
        output[i] = filtered * dynamic_decay
    
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
    rms = np.sqrt(np.convolve(buffer**2, np.ones(window_size)/window_size, mode='same'))
    
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
    频谱平衡均衡器（终极版）
    
    新增：
    1. 拾音器共振峰模拟（2-3kHz）
    2. 更平滑的高频滚降
    3. 动态低频控制
    """
    # 1. 高通滤波：切除 80Hz 以下（更陡峭）
    sos_hp = signal.butter(6, 80, 'hp', fs=SR, output='sos')  # 从4阶提升到6阶
    audio_buffer = signal.sosfilt(sos_hp, audio_buffer)
    
    # 2. 中低频控制（200-400Hz）- 减少"箱体轰鸣"
    b_notch, a_notch = signal.iirnotch(280, 25, SR)
    notch_signal = signal.lfilter(b_notch, a_notch, audio_buffer)
    audio_buffer = audio_buffer * 0.8 + notch_signal * 0.2
    
    # 3. 拾音器共振峰（2-3kHz）- 吉他特有的"金属质感"
    b_pickup, a_pickup = signal.iirpeak(2500, 12, SR)
    pickup_resonance = signal.lfilter(b_pickup, a_pickup, audio_buffer) * 0.25
    audio_buffer = audio_buffer + pickup_resonance
    
    # 4. 临场感提升（4-5kHz）
    b_presence, a_presence = signal.iirpeak(4500, 20, SR)
    presence = signal.lfilter(b_presence, a_presence, audio_buffer) * 0.18
    audio_buffer = audio_buffer + presence
    
    # 5. 空气感（8kHz 架子提升）
    sos_air = signal.butter(1, 8000, 'hp', fs=SR, output='sos')
    air = signal.sosfilt(sos_air, audio_buffer) * 0.12
    audio_buffer = audio_buffer + air
    
    # 6. 高频柔化（12kHz 平滑滚降）
    sos_lp = signal.butter(3, 12000, 'lp', fs=SR, output='sos')  # 从2阶提升到3阶
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
            freq_gain = 0.25  # 极低音大幅削减（消除刺耳）
        elif freq < 250:
            freq_gain = 0.4   # 低音大幅衰减
        elif freq < 500:
            freq_gain = 0.65  # 中低音适度衰减
        else:
            freq_gain = 1.0   # 高音保持
        
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
                wave_snippet[note_off:note_off+release_time] *= fade
                wave_snippet[note_off+release_time:] = 0.0
        
        # 叠加到混音缓冲
        end_idx = min(start + len(wave_snippet), total_samples)
        mix_buffer[start:end_idx] += wave_snippet[:end_idx-start]
    
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
