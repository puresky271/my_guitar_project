import numpy as np
import mido
import io
import wave
from numba import jit
from scipy import signal

SR = 48000


# ==========================================
#  效果器模块 (Punk FX Chain)
# ==========================================

def tube_screamer(buf, drive=0.6, tone=0.6, level=0.8):
    """
    Tube Screamer 风格过载 (Overdrive)
    非对称软削波模拟真空管饱和，产生温暖的偶次谐波。
    - drive: 失真深度 0~1
    - tone:  音色明暗 0~1
    - level: 输出电平 0~1
    """
    if drive < 0.01:
        return buf * level

    # 前置增益
    gained = buf * (1.0 + drive * 15.0)

    # 非对称软削波 (模拟二极管整流)
    # 正半周温和 tanh，负半周更猛 → 产生偶次谐波
    out = np.where(
        gained >= 0,
        np.tanh(gained * (1.0 + drive)),
        np.tanh(gained * (1.0 + drive * 1.6)) * 0.85
    )

    # Tone 控制 (一阶低通)
    cutoff = 800 + tone * 4000
    sos = signal.butter(1, cutoff, 'lp', fs=SR, output='sos')
    out = signal.sosfilt(sos, out)

    return out * level


def punk_distortion(buf, gain=0.7):
    """
    朋克硬失真 (Hard Clipping + Crossover Distortion)
    比 Tube Screamer 更脏更暴力，适合 full_band 合奏。
    """
    if gain < 0.01:
        return buf

    # 大增益推入削波
    gained = buf * (1.0 + gain * 25.0)

    # 不对称硬削波
    clipped = np.clip(gained, -0.7, 0.65)

    # 轻微 crossover distortion (推挽放大器的死区)
    dead_zone = 0.02 * gain
    clipped = np.where(np.abs(clipped) < dead_zone, 0.0, clipped)

    # 去掉最刺耳的高频毛刺
    sos = signal.butter(2, 6000, 'lp', fs=SR, output='sos')
    clipped = signal.sosfilt(sos, clipped)

    return clipped


def cabinet_sim(buf, cab_type='punk'):
    """
    吉他音箱箱体模拟 (Cabinet IR 近似)
    'punk': Marshall 1960 — 中频突出、高频有 bite
    'clean': Fender Twin — 平衡温暖
    """
    if cab_type == 'punk':
        # 切掉 80Hz 以下的轰鸣
        sos_hp = signal.butter(4, 80, 'hp', fs=SR, output='sos')
        buf = signal.sosfilt(sos_hp, buf)

        # Marshall 中频甜区共振 (800-2kHz)
        b1, a1 = signal.iirpeak(1200, 3, SR)
        buf = signal.lfilter(b1, a1, buf) * 0.6 + buf * 0.5

        # Presence bite (3-5kHz 的攻击感)
        b2, a2 = signal.iirpeak(3500, 6, SR)
        buf = buf + signal.lfilter(b2, a2, buf) * 0.25

        # 纸盆喇叭高频滚降
        sos_lp = signal.butter(3, 7000, 'lp', fs=SR, output='sos')
        buf = signal.sosfilt(sos_lp, buf)

        # 箱体低频共振 (150Hz 的木质感)
        b3, a3 = signal.iirpeak(150, 5, SR)
        buf = buf + signal.lfilter(b3, a3, buf) * 0.15

    else:  # clean — Fender Twin
        sos_hp = signal.butter(4, 80, 'hp', fs=SR, output='sos')
        buf = signal.sosfilt(sos_hp, buf)
        # Fender 式中频微凹
        b_n, a_n = signal.iirnotch(400, 8, SR)
        buf = signal.lfilter(b_n, a_n, buf)
        sos_lp = signal.butter(3, 10000, 'lp', fs=SR, output='sos')
        buf = signal.sosfilt(sos_lp, buf)

    return buf


def spring_reverb(buf, amount=0.3):
    """
    弹簧混响模拟 (Punk/Surf 经典)
    多段延迟 + 金属质感带通 → 弹簧的 "drip" 质感
    """
    if amount < 0.01:
        return buf

    wet = np.zeros_like(buf)
    # 三段延迟线模拟弹簧的多次反射
    delays = [int(SR * 0.033), int(SR * 0.057), int(SR * 0.079)]
    gains = [0.45, 0.30, 0.18]

    for d, g in zip(delays, gains):
        if len(buf) > d:
            wet[d:] += buf[:-d] * g

    # 弹簧的金属共振: 带通 600-4000Hz
    sos = signal.butter(2, [600, 4000], 'bp', fs=SR, output='sos')
    wet = signal.sosfilt(sos, wet)

    return buf * (1.0 - amount * 0.3) + wet * amount


# ==========================================
#  核心合成引擎 (Karplus-Strong v3)
# ==========================================

@jit(nopython=True, fastmath=True)
def karplus_strong_hifi(n_samples, delay_samples, velocity, brightness, decay_factor):
    """
    高保真 Karplus-Strong v3
    改进：三角波+谐波激励、全音域衰减平衡、弦张力非线性
    """
    output = np.zeros(n_samples, dtype=np.float32)

    # === 1. 激励信号 (三角波 + 谐波 + 噪声) ===
    burst_len = delay_samples
    if burst_len > n_samples:
        burst_len = n_samples

    half = max(burst_len // 2, 1)
    quarter = max(burst_len // 4, 1)

    for i in range(burst_len):
        # 三角波
        if i < half:
            triangle = (i / half) * 2.0 - 1.0
        else:
            triangle = 1.0 - ((i - half) / half) * 2.0

        # 二次/三次谐波 (让音色更丰满)
        phase = (i / max(delay_samples, 1)) * 2.0 * np.pi
        harmonic2 = np.sin(phase * 2.0) * 0.15
        harmonic3 = np.sin(phase * 3.0) * 0.08

        noise = np.random.uniform(-0.2, 0.2)

        # 窗口函数
        if i < quarter:
            window = i / quarter
        elif i > burst_len - quarter:
            window = (burst_len - i) / quarter
        else:
            window = 1.0

        # 亮度控制
        if i > 0:
            smoothed = (triangle + harmonic2 + harmonic3) * brightness + output[i - 1] * (1.0 - brightness) * 0.15
        else:
            smoothed = triangle + harmonic2 + harmonic3

        output[i] = (smoothed * 0.75 + noise * 0.25) * window * velocity

    # === 2. 物理反馈循环 ===
    freq = SR / delay_samples

    # 频率自适应衰减 (低音衰减慢，高音衰减快)
    if freq < 120:
        base_decay = 0.9988
    elif freq < 300:
        base_decay = 0.9992
    elif freq < 800:
        base_decay = 0.9994
    else:
        base_decay = 0.9996

    freq_decay = min(freq / 1500.0, 1.0) * 0.0006
    user_decay = decay_factor * 0.002
    final_decay = base_decay - freq_decay - user_decay
    final_decay = max(final_decay, 0.990)
    final_decay = min(final_decay, 0.9996)

    # 低通系数
    alpha = 0.45 + brightness * 0.40

    for i in range(delay_samples, n_samples):
        delayed_1 = output[i - delay_samples]
        delayed_2 = output[i - delay_samples - 1] if i > delay_samples else 0.0

        # 低通滤波
        filtered = delayed_1 * alpha + delayed_2 * (1.0 - alpha)

        # 弦张力非线性 (大振幅时频率微升)
        amplitude = abs(filtered)
        if amplitude > 0.25:
            filtered *= 1.0 + (amplitude - 0.25) * 0.015

        # 动态阻尼
        dynamic_decay = final_decay * (1.0 - amplitude * 0.008)
        output[i] = filtered * dynamic_decay

    return output


@jit(nopython=True, fastmath=True)
def soft_clipper(x, threshold=0.8):
    if abs(x) < threshold:
        return x
    sign = 1.0 if x > 0 else -1.0
    excess = abs(x) - threshold
    return sign * (threshold + excess / (1.0 + excess * excess))


def adaptive_limiter(buffer, target_peak=0.95):
    peak = np.max(np.abs(buffer))
    if peak > target_peak:
        buffer = buffer * (target_peak / peak)
    for i in range(len(buffer)):
        buffer[i] = soft_clipper(buffer[i], target_peak)
    return buffer


# ==========================================
#  频谱平衡 (Clean 模式专用)
# ==========================================

def spectral_balance_eq(audio_buffer):
    """Clean 吉他的频谱平衡 EQ"""
    # 高通 80Hz
    sos_hp = signal.butter(6, 80, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_hp, audio_buffer)

    # 中低频 de-mud (280Hz)
    b_n, a_n = signal.iirnotch(280, 25, SR)
    audio_buffer = audio_buffer * 0.8 + signal.lfilter(b_n, a_n, audio_buffer) * 0.2

    # 拾音器共振峰 (2.5kHz)
    b_p, a_p = signal.iirpeak(2500, 12, SR)
    audio_buffer += signal.lfilter(b_p, a_p, audio_buffer) * 0.25

    # Presence (4.5kHz)
    b_pr, a_pr = signal.iirpeak(4500, 20, SR)
    audio_buffer += signal.lfilter(b_pr, a_pr, audio_buffer) * 0.18

    # Air (8kHz+)
    sos_air = signal.butter(1, 8000, 'hp', fs=SR, output='sos')
    audio_buffer += signal.sosfilt(sos_air, audio_buffer) * 0.12

    # 高频柔化 (12kHz)
    sos_lp = signal.butter(3, 12000, 'lp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_lp, audio_buffer)

    return audio_buffer


# ==========================================
#  主渲染入口
# ==========================================

def midi_to_audio(midi_stream, brightness, pluck_position, body_mix, reflection, coupling,
                  distortion_mode='clean'):
    """
    distortion_mode:
      'clean'     — 原始清音 (独奏默认，向后兼容)
      'overdrive' — Tube Screamer 过载 (guitar_bass 二重奏)
      'punk'      — 硬失真 + Marshall 箱体 (full_band 朋克)
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

    for note, (start, vel) in active_notes.items():
        events.append((start, total_samples - SR, note, vel))

    print(f"🎸 吉他引擎 [{distortion_mode}]：{len(events)} 个音符")

    # === 复音数统计 & AGC ===
    max_polyphony = 1
    time_grid = np.zeros(total_samples, dtype=np.int16)
    for start, end, note, vel in events:
        if start < total_samples and end > start:
            e = min(end, total_samples)
            time_grid[start:e] += 1
            max_polyphony = max(max_polyphony, np.max(time_grid[start:e]))

    agc_factor = 1.0 / np.sqrt(max_polyphony)
    print(f"   最大复音数: {max_polyphony}, AGC: {agc_factor:.3f}")

    # 失真模式前置增益 (把信号推高，否则失真效果不够)
    dist_pre_gain = {'clean': 1.0, 'overdrive': 1.3, 'punk': 1.5}.get(distortion_mode, 1.0)

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

        vel_curve = (velocity / 127.0) ** 1.8

        # 频率增益 (失真模式下低频不需要大幅削减，失真本身压缩动态)
        if distortion_mode in ('overdrive', 'punk'):
            if freq < 100:
                freq_gain = 0.40
            elif freq < 200:
                freq_gain = 0.60
            else:
                freq_gain = 1.0
        else:
            if freq < 150:
                freq_gain = 0.25
            elif freq < 250:
                freq_gain = 0.4
            elif freq < 500:
                freq_gain = 0.65
            else:
                freq_gain = 1.0

        final_velocity = vel_curve * freq_gain * agc_factor * 0.8 * dist_pre_gain

        duration = (end - start) + int(SR * 0.5)
        duration = min(duration, total_samples - start)

        wave_snippet = karplus_strong_hifi(duration, delay_samples, final_velocity, brightness, coupling)

        # 释放包络
        release_time = int(SR * 0.15)
        note_off = end - start
        if 0 < note_off < len(wave_snippet):
            if note_off + release_time < len(wave_snippet):
                wave_snippet[note_off:note_off + release_time] *= np.linspace(1.0, 0.0, release_time)
                wave_snippet[note_off + release_time:] = 0.0

        end_idx = min(start + len(wave_snippet), total_samples)
        mix_buffer[start:end_idx] += wave_snippet[:end_idx - start]

    # === 后处理效果链 (根据 distortion_mode 切换) ===
    print(f"   效果链: {distortion_mode}")

    if distortion_mode == 'punk':
        # 朋克链: Distortion → Marshall Cabinet → Spring Reverb
        mix_buffer = punk_distortion(mix_buffer, gain=body_mix * 0.8 + 0.3)
        mix_buffer = cabinet_sim(mix_buffer, cab_type='punk')
        mix_buffer = spring_reverb(mix_buffer, amount=reflection * 0.5)

    elif distortion_mode == 'overdrive':
        # 过载链: Tube Screamer → Cabinet → Reverb
        mix_buffer = tube_screamer(mix_buffer, drive=body_mix * 0.7 + 0.2, tone=brightness, level=0.85)
        mix_buffer = cabinet_sim(mix_buffer, cab_type='punk')
        if reflection > 0.01:
            mix_buffer = spring_reverb(mix_buffer, amount=reflection * 0.4)

    else:  # clean
        # 清音链: EQ → Reverb (原始逻辑)
        mix_buffer = spectral_balance_eq(mix_buffer)
        if reflection > 0.01:
            d1 = int(SR * 0.08)
            if len(mix_buffer) > d1:
                reverb = np.zeros_like(mix_buffer)
                reverb[d1:] += mix_buffer[:-d1] * reflection * 0.5
                d2 = int(SR * 0.12)
                if len(mix_buffer) > d2:
                    reverb[d2:] += mix_buffer[:-d2] * reflection * 0.3
                mix_buffer = mix_buffer * 0.8 + reverb * 0.2

    # === 最终限制 ===
    mix_buffer = adaptive_limiter(mix_buffer, target_peak=0.93)

    peak = np.max(np.abs(mix_buffer))
    if peak > 0.01:
        mix_buffer = mix_buffer / peak * 0.95

    samples_int = (mix_buffer * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    print("✅ 吉他渲染完成")
    return buf.getvalue(), mix_buffer