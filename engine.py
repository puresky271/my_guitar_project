import mido
import numpy as np
from collections import deque
import io
import wave

SR = 48000  # 专业音频标准


# ---------------- 吉他琴箱共鸣 ----------------
def guitar_body_filter(samples, mix, sr=SR):
    f = 140      # 更接近吉他箱体主共鸣
    r = 0.96     # 大幅降低Q值（关键）

    y = np.zeros_like(samples)

    for n in range(2, len(samples)):
        y[n] = (
            samples[n]
            + 2 * r * np.cos(2*np.pi*f/sr) * y[n-1]
            - (r**2) * y[n-2]
        )

    # ⭐ 核心：与原声混合，而不是替代
    return samples * (1 - mix) + y * mix



# ---------------- 空间早期反射 ----------------
def early_reflection(samples, reflection, delay=120):
    out = np.copy(samples)
    for i in range(delay, len(samples)):
        out[i] += samples[i - delay] * reflection
    return out



# ---------------- 吉他弦模型 ----------------
class GuitarString:
    def __init__(self, frequency, brightness, pluck_position, coupling):
        self.capacity = int(round(SR / frequency))
        self.buffer = deque([0.0] * self.capacity)

        self.decay = 0.9985 - (frequency / 500000)
        self.brightness = brightness
        self.pluck_position = pluck_position
        self.coupling = coupling

        self.comb_delay = int(self.capacity * self.pluck_position)
        self.history = deque([0.0] * self.comb_delay, maxlen=self.comb_delay)

    def pluck(self, velocity=1.0):
        raw = np.random.uniform(-0.5, 0.5, self.capacity)
        noise = np.random.normal(0, 0.003, self.capacity)

        self.buffer.clear()
        for i in range(self.capacity):
            prev = raw[(i - 1) % self.capacity]
            curr = raw[i]
            nxt = raw[(i + 1) % self.capacity]
            smoothed = (prev * 0.25 + curr * 0.5 + nxt * 0.25)
            self.buffer.append((smoothed * velocity) + noise[i])

    def tic(self, neighbor_sample=0.0):
        front = self.buffer.popleft()
        nextv = self.buffer[0]

        # ⭐ 核心：加入高频成分（差分项）
        high_freq = front - nextv

        new_sample = (
                (front + nextv) * 0.5  # 原始KS低通
                + high_freq * 0.35  # 加入高频
        )

        new_sample = (new_sample + neighbor_sample * self.coupling) * self.decay
        self.buffer.append(new_sample)

    def sample(self):
        raw = self.buffer[0]

        delayed = self.history[0]
        out = raw - delayed * 0.85

        self.history.append(raw)
        return out


# ---------------- MIDI 渲染核心  ----------------
def midi_to_audio(midi_stream, brightness, pluck_position, body_mix, reflection, coupling):
    mid = mido.MidiFile(file=midi_stream)

    # 初始化 128 根弦
    strings = [
        GuitarString(
            440.0 * (2.0 ** ((i - 69) / 12.0)),
            brightness, pluck_position, coupling
        ) for i in range(128)
    ]

    active = set()
    samples = []

    # 简单的进度回调机制（如果需要在UI显示进度条，可以在这里扩展，目前保持简单）
    for msg in mid:
        ticks = int(msg.time * SR)

        # 为了性能，限制最大静默时长处理
        if ticks > SR * 2:
            ticks = SR * 2

        for _ in range(ticks):
            total = 0.0
            current_samples = {idx: strings[idx].buffer[0] for idx in active}

            for idx in list(active):
                neighbor = 0.0
                if idx - 1 in current_samples: neighbor += current_samples[idx - 1]
                if idx + 1 in current_samples: neighbor += current_samples[idx + 1]

                total += strings[idx].sample()
                strings[idx].tic(neighbor)

            # 软削波限制
            samples.append(np.tanh(total))  # 使用 tanh 获得更好的电子管过载感

        if msg.type == 'note_on' and msg.velocity > 0:
            if 20 <= msg.note <= 108:  # 限制有效音域防止溢出
                active.add(msg.note)
                strings[msg.note].pluck(msg.velocity / 127.0)
        elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
            active.discard(msg.note)

    if not samples:
        return None, None

    samples_np = np.array(samples)

    # 后处理链
    samples_np = guitar_body_filter(samples_np, body_mix)
    samples_np = early_reflection(samples_np, reflection)

    # 归一化
    max_val = np.max(np.abs(samples_np))
    if max_val > 0:
        samples_np /= (max_val + 1e-6)

    # 转为 Int16
    samples_int = (samples_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples_int.tobytes())

    # --- 修改点：同时返回 bytes 和 numpy 数组 ---
    return buf.getvalue(), samples_np