import numpy as np
import streamlit as st
import io
import time
import base64
import os
import glob
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

# 设置非交互式后端
matplotlib.use('Agg')

# --- 页面配置 ---
st.set_page_config(
    page_title="Karplus-Strong Studio",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 深度优化 (保持 DAW 风格布局) ---
st.markdown("""
    <style>
    /* 全局背景 */
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }

    /* 标题美化 */
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
    }

    /* 按钮样式复原+增强 */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(255, 75, 75, 0.5);
        background-color: rgba(255, 75, 75, 0.1);
        color: #ff4b4b;
        transition: all 0.2s ease-in-out;
        height: 45px;
    }
    .stButton>button:hover {
        background-color: #ff4b4b;
        color: white;
        border-color: #ff4b4b;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }

    /* 信息卡片容器 */
    .metric-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.95rem;
        color: #aaa;
    }
    .metric-val {
        font-family: 'SF Mono', 'Consolas', monospace;
        color: #fff;
        font-weight: 500;
    }

    /* 侧边栏优化 */
    [data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #303030;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------- 资源加载 (GIF 逻辑修正) ----------
@st.cache_data(show_spinner=False)
def get_gif_button_html():
    """
    [修正] 只生成一个隐藏的全屏容器和一个文字按钮。
    页面上不显示任何图片预览，点击按钮才弹出全屏。
    """
    paths = [
        r"D:\python\my_guitar_project\assets\mygo.gif",
        "assets/mygo.gif",
        "./assets/mygo.gif",
        "../assets/mygo.gif"
    ]
    gif_b64 = None
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    gif_b64 = base64.b64encode(f.read()).decode()
                break
            except Exception:
                continue

    if not gif_b64:
        return ""

    # 注意：img 标签只在 display:none 的容器里
    return f"""
    <div id="fs-container" onclick="closeFS()" 
         style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:black; z-index:99999; align-items:center; justify-content:center; cursor: pointer;">
        <img src="data:image/gif;base64,{gif_b64}" style="max-width:100%; max-height:100%; object-fit: contain;">
    </div>

    <div style="margin-top: 15px; text-align: center;">
        <a href="javascript:void(0)" onclick="openFS()" 
           style="color: #ff4b4b; text-decoration: none; font-weight: bold; font-size: 14px; padding: 8px 16px; border: 1px dashed #ff4b4b; border-radius: 4px; transition: all 0.3s;"
           onmouseover="this.style.background='rgba(255,75,75,0.1)'" 
           onmouseout="this.style.background='transparent'">
           [ 🎬 好康的 ]
        </a>
    </div>

    <script>
        function openFS() {{
            var elem = document.getElementById("fs-container");
            elem.style.display = "flex";
            if (elem.requestFullscreen) elem.requestFullscreen();
            else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
        }}

        function closeFS() {{
            if (document.exitFullscreen) document.exitFullscreen();
            else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            document.getElementById("fs-container").style.display = "none";
        }}

        // 监听全屏退出事件，确保容器隐藏
        document.addEventListener('fullscreenchange', () => {{
            if (!document.fullscreenElement) {{
                document.getElementById("fs-container").style.display = "none";
            }}
        }});
    </script>
    """


# ---------- 状态初始化 (已更新为推荐参数) ----------
DEFAULTS = {
    "brightness": 0.75,  # 提升亮度，让声音更像新弦
    "pluck_position": 0.20,  # 微调拨弦位置，平衡清脆度
    "body_mix": 0.15,  # 降低共鸣，减少浑浊感 (关键优化)
    "reflection": 0.15,  # 适度增加空气感
    "coupling": 0.004,  # 增加一点延音
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

if st.session_state.get("reset_tone"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.reset_tone = False


# ---------- 辅助函数：扫描本地 MIDI ----------
def get_local_midi_files():
    """扫描 assets 文件夹下的所有 mid/midi 文件"""
    search_paths = [
        "assets/*.mid", "assets/*.midi",
        "../assets/*.mid", "../assets/*.midi",
        "./*.mid", "./*.midi"  # 容错
    ]
    files = []
    for pattern in search_paths:
        files.extend(glob.glob(pattern))

    # 去重并排序
    files = sorted(list(set(files)))
    return files


# ---------- 核心音频引擎 (带缓存) ----------
@st.cache_data(show_spinner=False)
def midi_to_audio_cached(file_bytes, instrument, brightness, pluck_pos, body_mix, reflection, coupling):
    try:
        if instrument == "guitar":
            from instruments import guitar as engine_module
        else:
            from instruments import piano as engine_module

        midi_stream = io.BytesIO(file_bytes)
        result = engine_module.midi_to_audio(
            midi_stream, brightness, pluck_pos, body_mix, reflection, coupling
        )

        if result is None or not isinstance(result, tuple) or result[0] is None:
            return None
        return result[0]
    except Exception as e:
        st.error(f"渲染引擎错误: {str(e)}")
        return None


# --- 可视化生成 ---
def generate_minimal_spectrogram(audio_bytes):
    try:
        with io.BytesIO(audio_bytes) as f:
            import wave
            with wave.open(f, 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
                audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

        fig = plt.figure(figsize=(12, 2.5), dpi=72, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        Pxx, freqs, bins, im = ax.specgram(audio_data, NFFT=1024, Fs=sr, noverlap=512,
                                           cmap='gray', mode='magnitude', scale='dB')
        im.set_alpha(0.25)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        return base64.b64encode(img_buf.getvalue()).decode()
    except Exception:
        return None


# --- 播放器 V3 ---
def render_sync_player(audio_bytes):
    try:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        spec_img_b64 = generate_minimal_spectrogram(audio_bytes)
        bg_style = ""
        if spec_img_b64:
            bg_style = f"background-image: url('data:image/png;base64,{spec_img_b64}'); background-size: cover; opacity: 0.8;"
    except Exception as e:
        st.error(f"播放器错误: {e}")
        return

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
        <style>
            body {{
                margin: 0; padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background: transparent;
                overflow: hidden;
                user-select: none;
                color: #e0e0e0;
            }}

            .player-card {{
                background: #0e1117;
                border: 1px solid #303030;
                border-radius: 12px;
                padding: 16px;
                display: flex;
                gap: 16px;
                height: 90px;
                box-sizing: border-box;
                align-items: center;
            }}

            .play-section {{ flex-shrink: 0; }}
            .play-btn {{
                width: 48px; height: 48px;
                border-radius: 50%;
                background: #ff4b4b;
                border: none;
                cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                transition: all 0.2s;
            }}
            .play-btn:hover {{ background: #ff6b6b; transform: scale(1.05); }}
            .play-btn svg {{ fill: white; width: 20px; height: 20px; margin-left: 3px; }}
            .play-btn.playing svg {{ margin-left: 0; }}

            .wave-section {{
                flex-grow: 1; height: 100%;
                display: flex; flex-direction: column; justify-content: center;
                position: relative;
                background: rgba(255,255,255,0.02);
                border-radius: 8px; overflow: hidden;
            }}
            .spectrogram-bg {{
                position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                {bg_style}
                filter: grayscale(100%) contrast(1.1); z-index: 0;
            }}
            #waveform {{
                position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                z-index: 1; cursor: text;
            }}
            .loader {{
                position: absolute; z-index: 2; top: 50%; left: 50%;
                transform: translate(-50%,-50%);
                font-size: 11px; color: #666; letter-spacing: 1px;
            }}

            .controls-section {{
                width: 140px; flex-shrink: 0;
                display: flex; flex-direction: column; justify-content: space-between;
                height: 100%; padding-left: 10px;
                border-left: 1px solid #222;
            }}

            .time-display {{
                font-family: 'SF Mono', 'Consolas', monospace;
                font-size: 13px; color: #ff4b4b;
                text-align: right; font-weight: 500;
            }}

            .ctrl-row {{
                display: flex; align-items: center; justify-content: space-between; gap: 8px;
            }}

            .vol-wrap {{ display: flex; align-items: center; gap: 4px; flex: 1; }}
            input[type=range] {{ -webkit-appearance: none; width: 100%; background: transparent; }}
            input[type=range]::-webkit-slider-runnable-track {{
                width: 100%; height: 4px; background: #333; border-radius: 2px;
            }}
            input[type=range]::-webkit-slider-thumb {{
                -webkit-appearance: none; height: 10px; width: 10px;
                border-radius: 50%; background: #ccc; margin-top: -3px;
                cursor: pointer;
            }}
            input[type=range]:hover::-webkit-slider-thumb {{ background: #fff; }}

            .speed-select {{
                background: transparent; border: 1px solid #333;
                color: #888; font-size: 10px; border-radius: 4px;
                padding: 2px 4px; cursor: pointer; outline: none;
            }}
            .speed-select:hover {{ border-color: #555; color: #ccc; }}
        </style>
    </head>
    <body>
        <div class="player-card">
            <div class="play-section">
                <button class="play-btn" id="playBtn" onclick="togglePlay()">
                    <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                </button>
            </div>
            <div class="wave-section">
                <div class="loader" id="loader">LOADING...</div>
                <div class="spectrogram-bg"></div>
                <div id="waveform"></div>
            </div>
            <div class="controls-section">
                <div class="time-display" id="timeDisplay">00:00</div>
                <div class="ctrl-row">
                    <div class="vol-wrap" title="音量">
                        <svg viewBox="0 0 24 24" width="12" height="12" fill="#666"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg>
                        <input type="range" id="volSlider" min="0" max="1" step="0.05" value="0.8">
                    </div>
                    <select class="speed-select" id="speedSelect" title="倍速">
                        <option value="0.5">0.5x</option>
                        <option value="0.8">0.8x</option>
                        <option value="1.0" selected>1.0x</option>
                        <option value="1.2">1.2x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2.0">2.0x</option>
                    </select>
                </div>
            </div>
        </div>
        <script>
            const audioData = "data:audio/wav;base64,{audio_b64}";
            let isPlaying = false;
            let wavesurfer;

            function fmt(t) {{
                const m = Math.floor(t / 60).toString().padStart(2, '0');
                const s = Math.floor(t % 60).toString().padStart(2, '0');
                return `${{m}}:${{s}}`;
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                wavesurfer = WaveSurfer.create({{
                    container: '#waveform',
                    waveColor: '#555', progressColor: '#ff4b4b',
                    cursorColor: 'rgba(255,255,255,0.8)', cursorWidth: 1,
                    barWidth: 2, barGap: 2, barRadius: 2,
                    height: 58, normalize: true, interact: true,
                }});
                wavesurfer.load(audioData);
                wavesurfer.on('ready', () => {{
                    document.getElementById('loader').style.display = 'none';
                    wavesurfer.setVolume(0.8);
                    updateTime();
                }});
                wavesurfer.on('audioprocess', updateTime);
                wavesurfer.on('seek', updateTime);
                wavesurfer.on('finish', () => {{ isPlaying = false; updateBtn(); }});

                document.getElementById('volSlider').addEventListener('input', (e) => wavesurfer.setVolume(e.target.value));
                document.getElementById('speedSelect').addEventListener('change', (e) => wavesurfer.setPlaybackRate(parseFloat(e.target.value)));

                window.togglePlay = function() {{
                    wavesurfer.playPause(); isPlaying = !isPlaying; updateBtn();
                }};

                function updateBtn() {{
                    const btn = document.getElementById('playBtn');
                    if (isPlaying) {{
                        btn.classList.add('playing');
                        btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
                    }} else {{
                        btn.classList.remove('playing');
                        btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
                    }}
                }}
                function updateTime() {{
                    document.getElementById('timeDisplay').innerText = fmt(wavesurfer.getCurrentTime());
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=125)


# --- 侧边栏 ---
with st.sidebar:
    st.title("音色实验室")
    st.caption("在调参后请手动重新生成，虽然我也不建议你改就是了")
    st.caption("如果你觉得麦很炸，就换成钢琴模式")
    st.markdown("---")

    # 乐器状态管理
    instrument = st.session_state.get('instrument', 'guitar')

    if instrument == "guitar":
        st.subheader("🎸 吉他物理参数")
        pluck_position = st.slider(
            "拨弦位置（近琴桥 ⇄ 近指板）", 0.08, 0.35, step=0.01, key="pluck_position",
            help="决定音色是清脆还是温暖。数值越小越清脆（近琴桥），数值越大越圆润（近指板）。"
        )
        body_mix = st.slider("琴箱共鸣强度", 0.0, 0.6, step=0.02, key="body_mix",
                             help="越大越有木头味（Boxy），但过大会导致声音变闷。")
        reflection = st.slider("空间反射感", 0.0, 0.3, step=0.01, key="reflection",
                               help="模拟琴体内部的回响，增加空气感。")
        brightness = st.slider("弦的亮度", 0.2, 0.8, step=0.02, key="brightness",
                               help="控制弦振动的高频保留时间，值越大声音越明亮。")
        coupling = st.slider("弦间共振（串扰）", 0.0, 0.01, step=0.0005, key="coupling",
                             help="一根弦震动带动其它弦震动，增加真实感和浑厚度。")
    else:
        st.subheader("🎹 钢琴物理参数")
        st.info("我不知道为什么有时候钢琴反而听起来更像吉他")
        reflection = st.slider("音乐厅混响", 0.0, 0.4, step=0.02, key="reflection",
                               help="模拟音乐厅的混响效果，值越大空间感越强。")
        st.markdown("---")
        st.markdown("""
        **钢琴物理特性：**
        - 低音区：单弦
        - 中音区：双弦耦合
        - 高音区：三弦合唱
        - 自动音板共鸣
        """)

    st.markdown("---")
    if st.button("🔄 恢复默认音色", use_container_width=True):
        st.session_state.reset_tone = True
        st.rerun()

# --- 主界面标题区 ---
if instrument == 'guitar':
    icon = "🎸"
    title = "Karplus-Strong 吉他工作室"
    subtitle = "物理建模 · MIDI → 原声吉他 · 高保真合成"
    gradient = "linear-gradient(90deg,#0f2027,#203a43,#2c5364)"
else:
    icon = "🎹"
    title = "Karplus-Strong 钢琴工作室"
    subtitle = "多弦耦合 · MIDI → 三角钢琴 · 音乐厅混响"
    gradient = "linear-gradient(90deg,#1a1a2e,#16213e,#0f3460)"

st.markdown(f"""
<div style="background: {gradient}; padding: 18px 28px; border-radius: 12px; color: white; margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin:0;">{icon} {title}</h2>
            <p style="margin:0; opacity:0.85;">{subtitle}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 乐器切换按钮
col_inst1, col_inst2, col_inst3 = st.columns([1, 1, 3])
with col_inst1:
    if st.button("🎸 吉他模式", type="primary" if instrument == "guitar" else "secondary", use_container_width=True):
        st.session_state.instrument = "guitar"
        st.rerun()
with col_inst2:
    if st.button("🎹 钢琴模式", type="primary" if instrument == "piano" else "secondary", use_container_width=True):
        st.session_state.instrument = "piano"
        st.rerun()

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# --- 主工作区 ---
col_main, col_output = st.columns([1, 1], gap="large")

with col_main:
    st.markdown("### 1. 选择 MIDI 来源")

    # --- 内置库逻辑 ---
    source_options = ["😡为什么要演奏春日影", "💿 内置 MIDI 库", "📂上传自己的 MIDI"]
    mode = st.radio("MIDI 来源", source_options, horizontal=True, label_visibility="collapsed")

    uploaded_file = None

    if mode == "📂上传自己的 MIDI":
        f = st.file_uploader("上传 MIDI 序列", type=["mid", "midi"], label_visibility="collapsed")
        if f:
            uploaded_file = io.BytesIO(f.read())
            uploaded_file.name = f.name

    elif mode == "💿 内置 MIDI 库":
        # 扫描文件
        local_files = get_local_midi_files()
        if not local_files:
            st.warning("⚠️ assets 文件夹下没有找到 MIDI 文件。")
        else:
            # 创建文件名列表供显示
            file_options = {os.path.basename(p): p for p in local_files}
            selected_name = st.selectbox("请选择一首歌曲:", list(file_options.keys()))

            # 读取选中文件
            if selected_name:
                selected_path = file_options[selected_name]
                try:
                    with open(selected_path, "rb") as f:
                        uploaded_file = io.BytesIO(f.read())
                        uploaded_file.name = selected_name
                except Exception as e:
                    st.error(f"无法读取文件: {e}")

    else:  # 😡为什么要演奏春日影 (Legacy)
        try:
            # 尝试多个路径寻找
            paths = ["assets/春日影-mygo.mid", "../assets/春日影-mygo.mid", "春日影-mygo.mid"]
            found = False
            for p in paths:
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        uploaded_file = io.BytesIO(f.read())
                        uploaded_file.name = "春日影-mygo.mid"
                        found = True
                    break
            if not found:
                st.warning("⚠️ 默认 MIDI 文件未找到，请检查 assets 文件夹。")
        except Exception:
            st.warning("⚠️ 读取默认文件失败")

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        # 信息卡片
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-row"><span>📄 文件:</span> <span class="metric-val">{uploaded_file.name}</span></div>
            <div class="metric-row"><span>🎚️ 采样率:</span> <span class="metric-val">48000 Hz</span></div>
            <div class="metric-row"><span>🎼 乐器:</span> <span class="metric-val">{instrument.upper()}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 2. 执行渲染")

        # 根据乐器模式改变按钮文字
        if instrument == "guitar":
            button_text = "🎸 GuitarHero，启动！"
            status_text = "正在进行吉他弦振动模拟..."
            init_text = "初始化 128 根虚拟吉他弦..."
            parse_text = "解析 MIDI 事件并进行活跃弦追踪..."
        else:
            button_text = "🎹 PianoMaster，启动！"
            status_text = "正在进行钢琴物理建模..."
            init_text = "初始化 88 键三角钢琴（多弦耦合）..."
            parse_text = "解析 MIDI 事件并模拟琴槌敲击..."

        if st.button(button_text, type="primary", use_container_width=True):
            with st.status(status_text, expanded=True) as status:
                st.write(init_text)
                time.sleep(0.3)
                st.write(parse_text)

                audio_bytes = midi_to_audio_cached(
                    file_bytes, instrument,
                    st.session_state.brightness,
                    st.session_state.pluck_position,
                    st.session_state.body_mix,
                    st.session_state.reflection,
                    st.session_state.coupling
                )

                if audio_bytes:
                    st.session_state.audio_out = audio_bytes
                    st.session_state.render_done = True
                    status.update(label="✅ 音频加载成功，请稍等渲染结果", state="complete", expanded=False)
                else:
                    st.session_state.render_done = False
                    status.update(label="❌ 渲染失败", state="error", expanded=False)

        # 彩蛋按钮 (仅在选择"为什么要演奏春日影"时显示)
        if mode == "😡为什么要演奏春日影" and st.session_state.get("render_done"):
            st.components.v1.html(get_gif_button_html(), height=60)

with col_output:
    st.markdown("### 3. 输出与试听")

    if 'audio_out' in st.session_state and st.session_state.audio_out:
        # 播放器组件
        render_sync_player(st.session_state.audio_out)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # 下载 + 清空缓存 按钮组
        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            st.download_button(
                label="💾 点我下载 WAV 文件",
                data=st.session_state.audio_out,
                file_name=f"render_{instrument}_{time.strftime('%Y%m%d_%H%M')}.wav",
                mime="audio/wav",
                use_container_width=True
            )
        with d_col2:
            st.button("🗑️", help="清除缓存", on_click=lambda: st.session_state.pop('audio_out', None),
                      use_container_width=True)

    else:
        st.markdown("""
        <div style="
            border: 2px dashed #333; 
            border-radius: 12px; 
            height: 150px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            color: #666; 
            background: rgba(255,255,255,0.01);">
            等待渲染任务完成...
        </div>
        """, unsafe_allow_html=True)

# --- 页脚 ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>© 2026 青空 Karplus-Strong Studio | 基于CS61B Java 原版逻辑复刻</p>",
    unsafe_allow_html=True
)

