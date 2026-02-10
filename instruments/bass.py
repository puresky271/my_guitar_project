import numpy as np
import mido
import io
import wave
from numba import jit
from scipy import signal

SR = 48000


@jit(nopython=True, fastmath=True)
def bass_string_model(n_samples, delay_samples, velocity, brightness):
    """
    贝斯弦物理模型

    贝斯特点：
    1. 弦更粗更重 → 衰减极慢
    2. 张力更低 → 更多非线性
    3. 低频丰富 → 需要特殊处理
    """
    output = np.zeros(n_samples, dtype=np.float32)

    # === 1. 激励信号（贝斯的拨弦更"肉"） ===
    burst_len = int(delay_samples * 1.2)  # 贝斯激励更长
    if burst_len > n_samples:
        burst_len = n_samples
    
    # [BUGFIX] 防止 burst_len 过小时除以 0
    rise_len = burst_len // 4
    if rise_len < 1:
        rise_len = 1

    # 使用更厚重的激励波形
    for i in range(burst_len):
        # 梯形波（而非三角波），更"厚实"
        if i < rise_len:
            shape = i / rise_len
        elif i < 3 * rise_len:
            shape = 1.0
        else:
            # 这里的逻辑也需要适配 rise_len 以防越界，但在 Numba 中通常 ok
            # 简单修改为基于 rise_len 的下降
            fall_phase = i - 3 * rise_len
            shape = 1.0 - (fall_phase / rise_len)
            if shape < 0: shape = 0.0

        # 少量噪声
        noise = np.random.uniform(-0.15, 0.15)

        # 低通平滑（贝斯高频少）
        if i > 0:
            smoothed = shape * 0.7 + output[i - 1] * 0.3
        else:
            smoothed = shape

        output[i] = (smoothed * 0.85 + noise * 0.15) * velocity

    # === 2. 物理反馈循环（贝斯衰减极慢） ===
    freq = SR / delay_samples

    # 贝斯的衰减比吉他慢得多
    base_decay = 0.9996  # 吉他是 0.9992

    # 低频额外保护（贝斯最重要的是低频持续）
    if freq < 100:
        base_decay = 0.9998
    elif freq < 200:
        base_decay = 0.9997

    # 贝斯的低通滤波更激进（天然高频少）
    alpha = 0.4 + brightness * 0.25  # 比吉他更低

    # 主循环（加入贝斯特有的"松弛"非线性）
    for i in range(delay_samples, n_samples):
        delayed_1 = output[i - delay_samples]
        delayed_2 = output[i - delay_samples - 1] if i > delay_samples else 0.0

        # 低通滤波
        filtered = delayed_1 * alpha + delayed_2 * (1.0 - alpha)

        # 贝斯弦的"松弛"非线性：低张力导致的频率下探
        amplitude = abs(filtered)
        if amplitude > 0.2:
            # 大振幅时频率略微下降（与吉他相反）
            tension_sag = 1.0 - (amplitude - 0.2) * 0.015
            filtered *= tension_sag

        output[i] = filtered * base_decay

    return output


def bass_body_filter(samples, mix):
    """
    贝斯箱体共鸣（与吉他不同）

    贝斯特点：
    - 主共振在 80-120Hz（更低）
    - Q 值更高（更窄的峰）
    """
    if mix <= 0:
        return samples

    # 主共振峰在 100Hz
    b_body, a_body = signal.iirpeak(100, 8, SR)
    body_resonance = signal.lfilter(b_body, a_body, samples)

    # 次共振峰在 180Hz
    b_body2, a_body2 = signal.iirpeak(180, 12, SR)
    body_resonance2 = signal.lfilter(b_body2, a_body2, samples)

    # 混合
    result = samples * (1 - mix) + (body_resonance * 0.6 + body_resonance2 * 0.4) * mix

    return result


def bass_eq_mastering(audio_buffer):
    """
    贝斯专用 EQ

    目标：
    1. 保留 40-150Hz 的核心低频
    2. 削减 200-500Hz 的"泥泞"
    3. 提升 2-4kHz 的"颗粒感"（拨弦声）
    """
    # 1. 高通 30Hz（只切最低的隆隆声）
    sos_hp = signal.butter(4, 30, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_hp, audio_buffer)

    # 2. 低频核心提升（80Hz）
    b_low, a_low = signal.iirpeak(80, 6, SR)
    low_boost = signal.lfilter(b_low, a_low, audio_buffer) * 0.2
    audio_buffer = audio_buffer + low_boost

    # 3. 中低频削减（250-400Hz，消除"泥泞"）
    b_mud, a_mud = signal.iirnotch(320, 10, SR)
    audio_buffer = signal.lfilter(b_mud, a_mud, audio_buffer)

    # 4. 高中频提升（2.5kHz，拨弦"颗粒感"）
    b_attack, a_attack = signal.iirpeak(2500, 15, SR)
    attack_boost = signal.lfilter(b_attack, a_attack, audio_buffer) * 0.25
    audio_buffer = audio_buffer + attack_boost

    # 5. 高频适度滚降（贝斯不需要太多高频）
    sos_lp = signal.butter(2, 8000, 'lp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_lp, audio_buffer)

    return audio_buffer


def adaptive_limiter(buffer, target_peak=0.95):
    """贝斯专用限制器（低频友好）"""
    # 对低频更温和的限制
    for i in range(len(buffer)):
        if abs(buffer[i]) > target_peak:
            # 软削波
            sign = 1.0 if buffer[i] > 0 else -1.0
            excess = abs(buffer[i]) - target_peak
            buffer[i] = sign * (target_peak + excess / (1.0 + excess * 2))

    return buffer


def midi_to_audio(midi_stream, brightness, pluck_position, body_mix, reflection, coupling):
    """
    贝斯 MIDI 渲染

    参数映射：
    - brightness: 音色明亮度（控制高频）
    - pluck_position: 拨弦力度曲线
    - body_mix: 箱体共鸣强度
    - reflection: 房间混响
    - coupling: 未使用（贝斯单弦）
    """
    # 是否启用贝斯自动改编（只影响 Bass 独奏）
    AUTO_BASS_ARRANGE = True

    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        print(f"MIDI 解析失败: {e}")
        return None, None

    total_len = sum(msg.time for msg in mid) + 4.0
    total_samples = int(total_len * SR)
    if total_samples > SR * 300:
        total_samples = SR * 300

    mix_buffer = np.zeros(total_samples, dtype=np.float32)

    # MIDI 事件解析
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

    print(f"🎸 贝斯引擎：处理 {len(events)} 个音符事件")

    # 自动增益控制
    max_polyphony = 1
    time_grid = np.zeros(total_samples, dtype=np.int16)
    # [BUGFIX] events 这里是 4 个元素，修正解包变量
    for start, end, note, vel in events:
        if start < total_samples and end > start:
            end = min(end, total_samples)
            time_grid[start:end] += 1
            max_polyphony = max(max_polyphony, np.max(time_grid[start:end]))

    agc_factor = 1.0 / np.sqrt(max_polyphony)
    print(f"   最大复音数: {max_polyphony}, 自动增益: {agc_factor:.3f}")

    # 音符渲染
    # [BUGFIX] 去掉了 ped，events 只有 4 个元素
    for start, end, note, velocity in events:

        # ================== Bass 自动改编核心 ==================
        if AUTO_BASS_ARRANGE:
            # 压到贝斯音域 E1 ~ G3
            while note > 55:
                note -= 12
            while note < 28:
                note += 12

            # 贝斯不演和弦，只取低音（已经是最低音域了）
            # 并且延长时值，让旋律连贯
            end += int(0.15 * SR)
        # ======================================================

        if start >= total_samples:
            continue

        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))

        # 贝斯有效音域：E1 (41.2Hz) 到 C4 (261Hz)
        if freq > 300 or freq < 35:
            continue

        delay_samples = int(SR / freq)
        if delay_samples < 2:
            continue

        # 力度曲线（使用 pluck_position 参数）
        vel_curve = (velocity / 127.0) ** pluck_position

        # 贝斯不需要频率平衡（低频就是优势）
        freq_gain = 1.0

        final_velocity = vel_curve * freq_gain * agc_factor * 1.2

        # 生成音符（贝斯余音更长）
        duration = (end - start) + int(SR * 0.8)
        duration = min(duration, total_samples - start)

        wave_snippet = bass_string_model(
            duration,
            delay_samples,
            final_velocity,
            brightness
        )

        # 释放包络
        release_time = int(SR * 0.2)
        note_off = end - start

        if note_off > 0 and note_off < len(wave_snippet):
            if note_off + release_time < len(wave_snippet):
                fade = np.linspace(1.0, 0.0, release_time)
                wave_snippet[note_off:note_off + release_time] *= fade
                wave_snippet[note_off + release_time:] = 0.0

        # 叠加
        end_idx = min(start + len(wave_snippet), total_samples)
        # 确保切片长度一致
        snippet_len = end_idx - start
        if snippet_len > 0:
            mix_buffer[start:end_idx] += wave_snippet[:snippet_len]

    # 后处理链
    print("   应用后处理...")

    # 1. 贝斯箱体共鸣
    mix_buffer = bass_body_filter(mix_buffer, body_mix)

    # 2. 贝斯 EQ
    mix_buffer = bass_eq_mastering(mix_buffer)

    # 3. 房间混响
    if reflection > 0.01:
        delay_samples = int(SR * 0.06)
        if len(mix_buffer) > delay_samples:
            reverb = np.zeros_like(mix_buffer)
            reverb[delay_samples:] += mix_buffer[:-delay_samples] * reflection * 0.4
            mix_buffer = mix_buffer * 0.85 + reverb * 0.15

    # 4. 自适应限制器
    mix_buffer = adaptive_limiter(mix_buffer, target_peak=0.95)

    # 5. 最终归一化
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

    print("✅ 贝斯渲染完成")

    return buf.getvalue(), mix_buffer
