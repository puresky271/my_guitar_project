import numpy as np
import mido
import io
import wave
from numba import jit
from scipy import signal

SR = 48000


@jit(nopython=True)
def piano_hammer(n_samples, delay_samples, velocity):
    output = np.zeros(n_samples, dtype=np.float32)

    # 1. 击弦激励 (Improved Hammer)
    # 模拟像毛毡一样的软击打
    burst_len = int(delay_samples * 0.5)
    if burst_len < 4: burst_len = 4
    if burst_len > n_samples: burst_len = n_samples

    for i in range(burst_len):
        # 多项式脉冲，比正弦波更像真实的物理撞击
        x = i / burst_len
        hammer = (x ** 2 * (1 - x) ** 2) * 16.0 * velocity

        # 加入极少量的高频噪声模拟琴弦接触瞬间的“呲”声
        noise = np.random.uniform(-0.1, 0.1) * velocity

        output[i] = hammer + noise

    # 2. 衰减系数
    # 钢琴：低音(长delay) decay 极慢 ~0.999
    # 高音(短delay) decay 快 ~0.990
    freq = SR / delay_samples
    rho = 0.997 - (freq * 0.00001)
    if rho < 0.98: rho = 0.98
    if rho > 0.999: rho = 0.999

    # 3. 循环
    for i in range(delay_samples, n_samples):
        # 简单的低通滤波器 loop
        idx = i - delay_samples

        # 稍微改变一点相位的平均，增加金属质感
        val = (output[idx] + output[idx - 1]) * 0.5

        output[i] = val * rho

    return output


def mastering_chain(audio_buffer, instrument_type='piano'):
    """
    通用母带处理
    """
    # 1. High Pass (切除 40Hz 以下的隆隆声)
    sos = signal.butter(4, 40, 'hp', fs=SR, output='sos')
    audio_buffer = signal.sosfilt(sos, audio_buffer)

    # 2. 钢琴专用的中频挖空 (Scoop) - 让声音更清晰
    b, a = signal.iirpeak(400, 1.0, fs=SR)
    # 3. 饱和与软限制
    # Soft Saturation
    gain = 1.3
    audio_buffer = audio_buffer * gain
    audio_buffer = audio_buffer / np.sqrt(1 + audio_buffer ** 2)

    # 4. Normalize
    max_val = np.max(np.abs(audio_buffer))
    if max_val > 0.96:
        audio_buffer = audio_buffer / max_val * 0.96

    return audio_buffer


def midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling):
    try:
        mid = mido.MidiFile(file=midi_stream)
    except Exception as e:
        return None, None

    total_len = sum(msg.time for msg in mid) + 4.0
    total_samples = int(total_len * SR)
    if total_samples > SR * 300: total_samples = SR * 300

    mix_buffer = np.zeros(total_samples, dtype=np.float32)
    current_sample = 0
    sustain_pedal = False
    active_notes = {}
    events = []

    for msg in mid:
        dt = int(msg.time * SR)
        current_sample += dt
        if msg.type == 'control_change' and msg.control == 64:
            sustain_pedal = (msg.value >= 64)
        elif msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = (current_sample, msg.velocity, sustain_pedal)
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start, vel, pedaled = active_notes.pop(msg.note)
                events.append((start, current_sample, msg.note, vel, sustain_pedal))

    for note, (start, vel, pedaled) in active_notes.items():
        events.append((start, total_samples - SR, note, vel, pedaled))

    print(f"Piano Pro: Processing {len(events)} events...")

    for start, end, note, velocity, pedaled in events:
        if start >= total_samples: continue

        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        if freq > SR / 2 or freq < 20: continue

        delay = int(SR / freq)
        if delay < 2: continue

        # === 钢琴音量平衡 ===
        # 同样应用低频衰减，防止左手低音掩盖右手旋律
        freq_gain = (freq / 200.0) ** 0.5
        if freq_gain < 0.6: freq_gain = 0.6
        if freq_gain > 1.1: freq_gain = 1.1

        # 力度平方律
        real_vel = (velocity / 127.0) ** 2.2

        # 生成
        full_decay = int(SR * 5.0)
        wave_snippet = piano_hammer(full_decay, delay, real_vel * freq_gain)

        # Envelope
        duration = end - start
        if not pedaled:
            # 模拟制音器
            rel_t = int(0.15 * SR)
            if duration < full_decay:
                damp_idx = duration
                if damp_idx + rel_t < len(wave_snippet):
                    fade = np.linspace(1.0, 0.0, rel_t)
                    wave_snippet[damp_idx:damp_idx + rel_t] *= fade
                    wave_snippet[damp_idx + rel_t:] = 0.0
                    wave_snippet = wave_snippet[:damp_idx + rel_t]

        # 叠加
        end_idx = start + len(wave_snippet)
        if end_idx > total_samples:
            end_idx = total_samples
            wave_snippet = wave_snippet[:end_idx - start]

        mix_buffer[start:end_idx] += wave_snippet * 0.7

    # Hall Reverb (Simple Convolution-like)
    if reflection > 0:
        # Pre-delay
        d1 = int(SR * 0.04)
        hall_lvl = reflection * 0.4
        if total_samples > d1:
            mix_buffer[d1:] += mix_buffer[:-d1] * hall_lvl

    # Mastering
    mix_buffer = mastering_chain(mix_buffer, 'piano')

    samples_int = (mix_buffer * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    return buf.getvalue(), mix_buffer