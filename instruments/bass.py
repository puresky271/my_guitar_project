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
    改进的贝斯弦物理模型 v2.2
    修复了低频能量无限堆积导致的背景轰鸣 (DC Offset / Rumble)
    """
    output = np.zeros(n_samples, dtype=np.float32)

    # === 1. 改进的拨弦激励 ===
    burst_len = int(delay_samples * 0.8)
    if burst_len > n_samples: burst_len = n_samples
    if burst_len < 1: burst_len = 1
    rise_len = max(1, burst_len // 6)

    for i in range(burst_len):
        if i < rise_len:
            shape = (i / rise_len) ** 0.6
        elif i < burst_len - rise_len:
            shape = 1.0
        else:
            fall_idx = i - (burst_len - rise_len)
            shape = 1.0 - (fall_idx / rise_len) ** 1.5
            if shape < 0: shape = 0.0

        phase = (i / delay_samples) * 2.0 * np.pi
        harmonic = np.sin(phase * 2.0) * 0.20 + np.sin(phase * 3.0) * 0.10
        noise = np.random.uniform(-0.15, 0.15)

        if i > 0:
            smoothed = (shape + harmonic) * 0.7 + output[i - 1] * 0.3
        else:
            smoothed = shape + harmonic

        output[i] = (smoothed * 0.75 + noise * 0.25) * velocity

    # === 2. 物理反馈循环 ===
    freq = SR / delay_samples

    if freq < 50:
        base_decay = 0.992
    elif freq < 100:
        base_decay = 0.996
    else:
        base_decay = 0.997

    alpha = 0.35 + brightness * 0.25

    for i in range(delay_samples, n_samples):
        delayed_1 = output[i - delay_samples]
        delayed_2 = output[i - delay_samples - 1] if i > delay_samples else 0.0

        filtered = delayed_1 * alpha + delayed_2 * (1.0 - alpha)

        amplitude = abs(filtered)
        if amplitude > 0.15:
            tension_sag = 1.0 - (amplitude - 0.15) * 0.008
            filtered *= tension_sag

        if amplitude > 0.6:
            filtered = np.tanh(filtered)

        output[i] = filtered * base_decay

        if output[i] > 4.0: output[i] = 4.0
        if output[i] < -4.0: output[i] = -4.0

    return output


def bass_body_filter(buffer, body_mix):
    if body_mix <= 0.01:
        return buffer
    b, a = signal.iirpeak(100, 2.5, SR)
    body_resonance = signal.lfilter(b, a, buffer)
    return buffer * (1.0 - body_mix * 0.6) + body_resonance * body_mix


def bass_eq_mastering(audio_buffer, brightness=0.5):
    # DC Blocker
    sos_dc = signal.butter(2, 25, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_dc, audio_buffer)

    # Sub Boost
    b_sub, a_sub = signal.iirpeak(70, 3, SR)
    sub_boost = signal.lfilter(b_sub, a_sub, audio_buffer) * 0.4
    audio_buffer = audio_buffer + sub_boost

    # De-mud
    b_mud, a_mud = signal.iirnotch(280, 5, SR)
    audio_buffer = signal.lfilter(b_mud, a_mud, audio_buffer)

    # Attack & Presence
    boost_factor = brightness * 0.6
    b_att, a_att = signal.iirpeak(2000, 8, SR)
    attack = signal.lfilter(b_att, a_att, audio_buffer) * boost_factor
    audio_buffer = audio_buffer + attack

    # LP
    sos_lp = signal.butter(2, 5000, 'lp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos_lp, audio_buffer)

    return audio_buffer


def adaptive_limiter(buffer, target_peak=0.96):
    peak = np.max(np.abs(buffer))
    if peak > target_peak:
        buffer = buffer * (target_peak / peak)
    return buffer


def midi_to_audio(midi_stream, brightness, pluck_position, body_mix, reflection, coupling, solo_mode=False):
    """
    solo_mode=True: 独奏模式，保留所有音符，不做节奏删减
    solo_mode=False: 伴奏模式，启用智能编曲，删减密集音符
    """
    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        print(f"MIDI 解析失败: {e}")
        return None, None

    total_len = sum(msg.time for msg in mid) + 4.0
    total_samples = int(total_len * SR)
    if total_samples > SR * 600: total_samples = SR * 600

    mix_buffer = np.zeros(total_samples, dtype=np.float32)

    raw_events = []
    cursor = 0
    active_notes = {}

    for msg in mid:
        cursor += int(msg.time * SR)
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = (cursor, msg.velocity)
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start, vel = active_notes.pop(msg.note)
                raw_events.append({'start': start, 'end': cursor, 'note': msg.note, 'vel': vel})

    raw_events.sort(key=lambda x: x['start'])

    # === 核心分歧：独奏 vs 伴奏 ===
    filtered_events = []

    if solo_mode:
        print("🎸 贝斯独奏模式：全音符保留")
        # 独奏模式：简单处理，仅做音域映射
        for evt in raw_events:
            note = evt['note']
            # 即使是独奏，太高的音用贝斯弹也不好听，适当降八度
            while note > 60:  # Middle C
                note -= 12
            while note < 28:  # E1
                note += 12

            filtered_events.append({
                'start': evt['start'],
                'end': evt['end'],
                'note': note,
                'vel': evt['vel']
            })
    else:
        print("🎸 贝斯伴奏模式：启用智能编曲")
        # 伴奏模式：使用 Smart Arranger (保留原有逻辑)
        last_start_time = -999999
        min_interval = int(SR * 0.120)
        time_window = int(SR * 0.04)

        i = 0
        while i < len(raw_events):
            current_cluster = [raw_events[i]]
            j = i + 1
            while j < len(raw_events) and (raw_events[j]['start'] - raw_events[i]['start'] < time_window):
                current_cluster.append(raw_events[j])
                j += 1

            best_note_event = min(current_cluster, key=lambda x: x['note'])

            time_diff = best_note_event['start'] - last_start_time
            is_strong_beat = best_note_event['vel'] > 90

            should_play = False
            if time_diff > min_interval:
                should_play = True
            elif is_strong_beat and time_diff > min_interval * 0.5:
                should_play = True

            if best_note_event['note'] > 67: should_play = False

            if should_play:
                target_note = best_note_event['note']
                while target_note > 48: target_note -= 12
                while target_note < 28: target_note += 12

                bass_vel = int(best_note_event['vel'] * 0.9 + 10)
                if bass_vel > 127: bass_vel = 127

                filtered_events.append({
                    'start': best_note_event['start'],
                    'end': best_note_event['end'],
                    'note': target_note,
                    'vel': bass_vel
                })
                last_start_time = best_note_event['start']
            i = j

    # === 音频渲染 ===
    for evt in filtered_events:
        start, end, note, velocity = evt['start'], evt['end'], evt['note'], evt['vel']

        midi_duration = end - start
        min_len = int(SR * 0.15)
        duration = max(midi_duration, min_len)
        duration = min(duration, total_samples - start)

        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        if freq < 20: continue

        delay_samples = int(SR / freq)
        if delay_samples < 2: continue

        # 动态控制
        # pluck_position 在这里做动态压缩
        # solo 模式下 pluck_position 可能还是默认的，确保它不是0
        p_pos = pluck_position if pluck_position > 0.1 else 1.0
        vel_curve = (velocity / 127.0) ** (1.0 / p_pos)
        final_velocity = vel_curve * 0.7

        wave_snippet = bass_string_model(duration, delay_samples, final_velocity, brightness)

        # 简单淡入淡出
        end_idx = min(start + len(wave_snippet), total_samples)
        snippet_len = end_idx - start
        fade_len = min(200, snippet_len // 4)
        if fade_len > 0:
            wave_snippet[:fade_len] *= np.linspace(0, 1, fade_len)
            wave_snippet[snippet_len - fade_len:snippet_len] *= np.linspace(1, 0, fade_len)

        if snippet_len > 0:
            mix_buffer[start:end_idx] += wave_snippet[:snippet_len]

    mix_buffer = bass_body_filter(mix_buffer, body_mix)
    mix_buffer = bass_eq_mastering(mix_buffer, brightness)

    if reflection > 0.01:
        delay_samples = int(SR * 0.03)
        if len(mix_buffer) > delay_samples:
            reverb_wet = np.zeros_like(mix_buffer)
            reverb_wet[delay_samples:] = mix_buffer[:-delay_samples] * reflection * 0.4
            mix_buffer += reverb_wet

    mix_buffer = adaptive_limiter(mix_buffer, target_peak=0.95)

    samples_int = (mix_buffer * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    return buf.getvalue(), mix_buffer
