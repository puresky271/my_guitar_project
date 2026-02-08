import streamlit as st
import io
import time
import base64
import streamlit.components.v1 as components
from engine import midi_to_audio

# --- 缓存装饰器  ---
@st.cache_data(show_spinner=False)
def midi_to_audio_cached(file_content, brightness, pluck_pos, body_mix, reflection, coupling):
  
    midi_stream = io.BytesIO(file_content)
   
    audio_bytes, _ = midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling)
    return audio_bytes

# --- 页面配置 ---
st.set_page_config(
    page_title="Karplus-Strong Studio",
    page_icon="🎸",
    layout="wide"
)

# --- CSS 样式优化  ---
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
        border: 1px solid #ff4b4b;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255,75,75,0.2);
    }
    .metric-container {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .stMarkdown p { font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State 初始化 ---
DEFAULTS = {
    "brightness": 0.5,
    "pluck_position": 0.18,
    "body_mix": 0.28,
    "reflection": 0.12,
    "coupling": 0.002,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.get("reset_tone"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.reset_tone = False

# --- 核心组件：同步波形播放器  ---
def render_sync_player(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; overflow: hidden; }}
            .audio-container {{ width: 100%; margin-bottom: 10px; }}
            audio {{ width: 100%; outline: none; filter: invert(0.9); }}
            #waveform {{ width: 100%; height: 80px; border-radius: 4px; background: rgba(255,255,255,0.05); }}
        </style>
    </head>
    <body>
        <div class="audio-container">
            <audio id="track" controls src="data:audio/wav;base64,{b64}"></audio>
        </div>
        <div id="waveform"></div>
        <script>
            const audioEl = document.querySelector('#track');
            const wavesurfer = WaveSurfer.create({{
                container: '#waveform',
                media: audioEl,
                waveColor: '#ff4b4b',
                progressColor: '#2C5364',
                barWidth: 2, barGap: 2, barRadius: 2, height: 80, normalize: true, interact: false,
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=140)
# --- 侧边栏 ---
with st.sidebar:
    st.title("音色实验室")
    st.caption("在调参后请手动重新生成")
    st.markdown("---")
    st.subheader("物理建模参数")

    pluck_position = st.slider(
        "拨弦位置（靠琴桥 ⇄ 靠指板）", 0.08, 0.35, step=0.01, key="pluck_position",
        help="决定音色是清脆还是温暖。数值越小越清脆（靠琴桥），数值越大越圆润（靠指板）。"
    )

    body_mix = st.slider(
        "琴箱共鸣强度", 0.0, 0.6, step=0.02, key="body_mix",
        help="越大越有木头味（Boxy），但过大会导致声音变闷。"
    )

    reflection = st.slider(
        "空间反射感", 0.0, 0.3, step=0.01, key="reflection",
        help="模拟琴体内部的回响，增加空气感。"
    )

    brightness = st.slider(
        "弦的亮度", 0.2, 0.8, step=0.02, key="brightness",
        help="控制弦振动的高频保留时间，值越大声音越明亮。"
    )

    coupling = st.slider(
        "弦间共振（串扰）", 0.0, 0.01, step=0.0005, key="coupling",
        help="一根弦震动带动其它弦震动，增加真实感和浑厚度。"
    )

    if st.button("🔄 恢复默认音色", use_container_width=True):
        st.session_state.reset_tone = True
        st.rerun()
        
# --- 标题  ---
st.markdown("""
<div style="background: linear-gradient(90deg,#0f2027,#203a43,#2c5364); padding: 18px 28px; border-radius: 12px; color: white; margin-bottom: 20px;">
    <h2 style="margin:0;">🎸 Karplus-Strong Studio</h2>
    <p style="margin:0; opacity:0.85;">物理建模 · MIDI → 原声吉他 · 高保真合成</p>
</div>
""", unsafe_allow_html=True)

# --- 主布局 ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 1. 选择 MIDI 来源")
    mode = st.radio("MIDI 来源", ["😡为什么要演奏春日影", "📂上传自己的 MIDI"], horizontal=True)

    file_content = None
    file_name = ""

    if mode == "📂上传自己的 MIDI":
        f = st.file_uploader("上传 MIDI 序列", type=["mid", "midi"], label_visibility="collapsed")
        if f:
            file_content = f.read()
            file_name = f.name
    else:
        with open("assets/春日影-mygo.mid", "rb") as f:
            file_content = f.read()
            file_name = "春日影-mygo.mid"

    if file_content:
        st.markdown(f"""
        <div class="metric-container">
            <div><strong>📄 文件:</strong> <span style="font-family: monospace;">{file_name}</span></div>
            <div><strong>🎚️ 采样率:</strong> <span style="font-family: monospace;">48000 Hz</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 2. 执行渲染")
        if st.button("🎧 GuitarHero，启动！", type="primary", use_container_width=True):
            with st.status("正在进行流式物理计算...", expanded=True) as status:
                st.write("初始化 128 根虚拟琴弦...")

                st.write("解析 MIDI 事件并进行活跃弦追踪...")

 
                audio_bytes = midi_to_audio_cached(
                    file_content, brightness, pluck_position, body_mix, reflection, coupling
                )

                if audio_bytes:
                    st.session_state.audio_out = audio_bytes
                    status.update(label="✅ 渲染成功!", state="complete", expanded=False)
                else:
                    st.error("渲染失败，请检查文件。")

with col_right:
    st.markdown("### 3. 输出与试听")
    if 'audio_out' in st.session_state and st.session_state.audio_out:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        render_sync_player(st.session_state.audio_out)
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="💾 点我下载 WAV 文件",
            data=st.session_state.audio_out,
            file_name=f"render_{time.strftime('%Y%m%d_%H%M')}.wav",
            mime="audio/wav",
            use_container_width=True
        )
    else:
        st.markdown("""
            <div style="margin-top: 20px; border: 2px dashed #333; border-radius: 10px; padding: 60px; text-align: center; color: #666;">
                等待渲染任务完成...
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2026 青空 Karplus-Strong Studio | 基于CS61B Java 原版逻辑复刻</p>", unsafe_allow_html=True)




