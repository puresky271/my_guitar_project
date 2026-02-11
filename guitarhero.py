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

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="Karplus-Strong Studio",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS 样式定义  ---
st.markdown("""
    <style>
    /* =================================
       背景透明化处理 (关键)
       ================================= */
    /* 1. 让主区域背景透明，否则会挡住背景图 */
    .main {
        background-color: transparent !important;
        color: #f0f2f6;
    }

    /* 2. 确保 Streamlit 的滚动容器也是透明的 */
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }

    /* 侧边栏保持深色，形成层次感 */
    [data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #303030;
    }

    /* =================================
       UI 组件美化
       ================================= */
    /* 标题字体 */
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
    }

    /* 按钮样式增强 */
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
        background: rgba(255, 255, 255, 0.03); /* 保持微弱背景以确保文字可读 */
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
    </style>
    """, unsafe_allow_html=True)


# --- 3. 背景图加载逻辑  ---
def set_background():
    """
    智能背景加载器：
    1. 自动搜索 assets 文件夹下的 jpg/png/jpeg 图片
    2. 如果找到，应用透明度背景
    3. 如果没找到，显示警告
    """
    # 搜索所有可能的图片
    valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif"]
    image_files = []

    # 检查当前目录和 assets 目录
    search_dirs = ["assets", ".", "./assets"]

    for directory in search_dirs:
        for ext in valid_extensions:
            # 拼接路径进行搜索
            pattern = os.path.join(directory, ext)
            image_files.extend(glob.glob(pattern))

    # 去重
    image_files = sorted(list(set(image_files)))

    # 如果没找到图片，发出警告并退出
    if not image_files:
        st.warning("⚠️ 背景图未生效：请在 assets 文件夹放入一张图片 (jpg/png)")
        return

    # 默认取第一张找到的图
    bg_path = image_files[0]

    # 尝试读取
    try:
        with open(bg_path, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()

        style = f"""
            <style>
            /* 强制清除 Streamlit 默认背景 */
            .stApp {{
                background: transparent !important;
            }}
            [data-testid="stAppViewContainer"] {{
                background: transparent !important;
            }}
            .main {{
                background: transparent !important;
            }}

            /* 添加背景图伪元素 */
            [data-testid="stAppViewContainer"]::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;

                /* 图片设置 */
                background-image: url(data:image/png;base64,{b64_encoded});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;

                /* --- 透明度调节 --- */
                opacity: 0.45;  /* 0.1(极淡) ~ 1.0(原图) */

                /* 确保在最底层 */
                z-index: -1;
                pointer-events: none; /* 确保不影响点击 */
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
        # 调试用：如果成功，下面这行可以注释掉
        # st.toast(f"已加载背景: {os.path.basename(bg_path)}")

    except Exception as e:
        st.error(f"背景图加载失败: {e}")


# 执行加载
set_background()


# --- 4. 资源加载 (GIF) ---
@st.cache_data(show_spinner=False)
def get_gif_button_html():
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
        document.addEventListener('fullscreenchange', () => {{
            if (!document.fullscreenElement) {{
                document.getElementById("fs-container").style.display = "none";
            }}
        }});
    </script>
    """


# --- 5. [彻底修复] 状态初始化与参数管理 ---

# 默认参数配置字典
DEFAULT_PARAMS = {
    "guitar": {
        "brightness": 0.60,
        "pluck_position": 0.25,
        "body_mix": 0.15,
        "reflection": 0.15,
        "coupling": 0.005
    },
    "bass": {
        "brightness": 0.65,
        "pluck_position": 1.8,
        "body_mix": 0.3,
        "reflection": 0.1,
        "coupling": 0.0
    },
    "piano": {
        "brightness": 0.65,
        "pluck_position": 1.0,
        "body_mix": 0.3,
        "reflection": 0.15,
        "coupling": 2.5
    },
    "guitar_bass": {
        "brightness": 0.5,
        "pluck_position": 1.8,
        "body_mix": 0.28,
        "reflection": 0.12,
        "coupling": 52
    },
    "drums": {
        "brightness": 0.7,
        "pluck_position": 1.2,
        "body_mix": 0.4,
        "reflection": 0.2,
        "coupling": 2.0
    },
    "full_band": {
        "brightness": 0.7,
        "pluck_position": 1.5,
        "body_mix": 0.35,
        "reflection": 0.18,
        "coupling": 52
    }
}

# 参数范围配置字典
PARAM_RANGES = {
    "guitar": {
        "brightness": (0.2, 0.8, 0.02),
        "pluck_position": (0.08, 0.40, 0.01),
        "body_mix": (0.0, 0.6, 0.02),
        "reflection": (0.0, 0.3, 0.01),
        "coupling": (0.0, 0.01, 0.0005)
    },
    "bass": {
        "brightness": (0.2, 0.7, 0.05),
        "pluck_position": (1.2, 2.5, 0.1),
        "body_mix": (0.0, 0.6, 0.05),
        "reflection": (0.0, 0.3, 0.02),
        "coupling": (0.0, 1.0, 0.1)
    },
    "piano": {
        "brightness": (0.3, 0.9, 0.05),
        "pluck_position": (0.5, 2.0, 0.1),
        "body_mix": (0.0, 0.5, 0.05),
        "reflection": (0.0, 0.4, 0.02),
        "coupling": (1.5, 3.5, 0.1)
    },
    "guitar_bass": {
        "brightness": (0.3, 0.8, 0.05),
        "pluck_position": (0.3, 3.0, 0.1),
        "body_mix": (0.0, 0.5, 0.02),
        "reflection": (0.0, 0.3, 0.01),
        "coupling": (45, 60, 1)
    },
    "drums": {
        "brightness": (0.3, 0.9, 0.05),
        "pluck_position": (0.5, 2.0, 0.1),
        "body_mix": (0.0, 0.8, 0.05),
        "reflection": (0.0, 0.5, 0.02),
        "coupling": (1.0, 3.0, 0.1)
    },
    "full_band": {
        "brightness": (0.4, 0.9, 0.05),
        "pluck_position": (0.8, 2.5, 0.1),
        "body_mix": (0.0, 0.6, 0.05),
        "reflection": (0.0, 0.4, 0.02),
        "coupling": (40, 65, 1)
    }
}

# 参数显示名称配置
PARAM_LABELS = {
    "guitar": {
        "brightness": "亮度",
        "pluck_position": "拨弦位置",
        "body_mix": "琴箱共鸣",
        "reflection": "空间反射",
        "coupling": "弦间共振"
    },
    "bass": {
        "brightness": "明亮度",
        "pluck_position": "拨弦力度",
        "body_mix": "箱体共鸣",
        "reflection": "房间混响",
        "coupling": None  # 不显示
    },
    "piano": {
        "brightness": "明亮度",
        "pluck_position": "琴槌硬度",
        "body_mix": "音板共鸣",
        "reflection": "混响",
        "coupling": "力度响应"
    },
    "guitar_bass": {
        "brightness": "整体亮度",
        "pluck_position": "音量平衡(左吉右贝)",
        "body_mix": "箱体共鸣",
        "reflection": "空间感",
        "coupling": "分频点(MIDI音符)"
    },
    "drums": {
        "brightness": "鼓皮硬度",
        "pluck_position": "打击响应",
        "body_mix": "腔体共鸣",
        "reflection": "混响",
        "coupling": "压缩感"
    },
    "full_band": {
        "brightness": "整体明亮",
        "pluck_position": "动态平衡",
        "body_mix": "乐器共鸣",
        "reflection": "混响",
        "coupling": "贝斯分频点"
    }
}

# 获取当前乐器
current_instrument = st.session_state.get('instrument', 'guitar')

# 检测乐器切换：如果乐器变化，重置所有参数为新乐器的默认值
if 'last_instrument' not in st.session_state:
    st.session_state.last_instrument = current_instrument
    # 首次加载，初始化参数
    for param, value in DEFAULT_PARAMS[current_instrument].items():
        if param not in st.session_state:
            st.session_state[param] = value

elif st.session_state.last_instrument != current_instrument:
    # 乐器切换了，重置所有参数
    for param, value in DEFAULT_PARAMS[current_instrument].items():
        st.session_state[param] = value
    st.session_state.last_instrument = current_instrument

# 恢复默认值功能
if st.session_state.get("reset_tone"):
    for param, value in DEFAULT_PARAMS[current_instrument].items():
        st.session_state[param] = value
    st.session_state.reset_tone = False


# --- 6. 辅助函数 ---
def get_local_midi_files():
    search_paths = [
        "assets/*.mid", "assets/*.midi",
        "../assets/*.mid", "../assets/*.midi",
        "./*.mid", "./*.midi"
    ]
    files = []
    for pattern in search_paths:
        files.extend(glob.glob(pattern))
    return sorted(list(set(files)))


@st.cache_data(show_spinner=False)
def midi_to_audio_cached(file_bytes, instrument, brightness, pluck_pos, body_mix, reflection, coupling):
    try:
        if instrument == "guitar":
            from instruments import guitar as engine_module
            midi_stream = io.BytesIO(file_bytes)

            result = engine_module.midi_to_audio(
                midi_stream,
                brightness,
                pluck_pos,
                body_mix,
                reflection,
                coupling
            )

            if result is None or not isinstance(result, tuple) or result[0] is None:
                return None

            return result[0]

        elif instrument == "bass":
            from instruments import bass as engine_module
            midi_stream = io.BytesIO(file_bytes)
            # 贝斯独奏模式：开启 solo_mode=True
            result = engine_module.midi_to_audio(
                midi_stream, brightness, pluck_pos, body_mix, reflection, coupling, solo_mode=True
            )
            if result is None or not isinstance(result, tuple) or result[0] is None:
                return None
            return result[0]

        elif instrument == "guitar_bass":
            from instruments import guitar, bass
            import numpy as np
            from scipy import signal

            midi_stream_guitar = io.BytesIO(file_bytes)
            midi_stream_bass = io.BytesIO(file_bytes)

            # ========== 改进的合奏策略 ==========

            # 1. 吉他：保持标准音色，不受 pluck_pos 影响
            GUITAR_PLUCK = 0.25
            GUITAR_COUPLING = 0.005

            result_guitar = guitar.midi_to_audio(
                midi_stream_guitar,
                brightness,  # 使用用户设置的明亮度
                GUITAR_PLUCK,
                body_mix,
                reflection,
                GUITAR_COUPLING
            )

            # 2. 贝斯：伴奏模式，使用标准参数
            BASS_PLUCK = 1.8

            result_bass = bass.midi_to_audio(
                midi_stream_bass,
                brightness * 0.85,  # 贝斯稍暗一点
                BASS_PLUCK,
                body_mix * 1.1,  # 贝斯箱体共鸣稍强
                reflection * 0.9,  # 贝斯混响稍弱
                0.0,
                solo_mode=False  # 伴奏模式
            )

            if not (result_guitar and result_bass and result_guitar[1] is not None and result_bass[1] is not None):
                return None

            guitar_samples = result_guitar[1]
            bass_samples = result_bass[1]

            # 3. 统一长度
            max_len = max(len(guitar_samples), len(bass_samples))
            if len(guitar_samples) < max_len:
                guitar_samples = np.pad(guitar_samples, (0, max_len - len(guitar_samples)))
            if len(bass_samples) < max_len:
                bass_samples = np.pad(bass_samples, (0, max_len - len(bass_samples)))

            # ========== 智能混音算法 ==========

            # 4. 动态能量检测（分析吉他的演奏密度）
            window_size = 48000  # 1秒窗口
            guitar_energy = np.convolve(
                guitar_samples ** 2,
                np.ones(window_size) / window_size,
                mode='same'
            )
            guitar_energy_norm = guitar_energy / (np.max(guitar_energy) + 1e-8)

            # 5. 贝斯呼吸感调制
            # 当吉他演奏密集时，贝斯音量降低30%；吉他稀疏时，贝斯填补空间
            bass_breathing = 1.0 - (guitar_energy_norm * 0.3)

            # 高斯平滑（避免突变）
            from scipy.ndimage import gaussian_filter1d
            bass_breathing = gaussian_filter1d(bass_breathing, sigma=4800)  # 0.1秒平滑

            # 应用呼吸感
            bass_samples_modulated = bass_samples * bass_breathing

            # 6. 频段分离混音（避免频段冲突）
            # 贝斯：强调 40-250Hz
            sos_bass_lp = signal.butter(4, 250, 'lp', fs=48000, output='sos')
            bass_low = signal.sosfilt(sos_bass_lp, bass_samples_modulated)

            # 吉他：强调 200Hz 以上
            sos_guitar_hp = signal.butter(4, 200, 'hp', fs=48000, output='sos')
            guitar_high = signal.sosfilt(sos_guitar_hp, guitar_samples)

            # 7. 音量平衡控制（使用 pluck_position 参数）
            # pluck_position: 0.3-3.0
            # < 1.0: 偏向吉他
            # = 1.0: 平衡
            # > 1.0: 偏向贝斯

            if pluck_pos < 1.0:
                # 偏向吉他
                guitar_vol = 0.65 + (1.0 - pluck_pos) * 0.2  # 0.65-0.85
                bass_vol = 0.35 - (1.0 - pluck_pos) * 0.15  # 0.20-0.35
            elif pluck_pos > 1.0:
                # 偏向贝斯
                guitar_vol = 0.65 - (pluck_pos - 1.0) * 0.15  # 0.35-0.65
                bass_vol = 0.35 + (pluck_pos - 1.0) * 0.20  # 0.35-0.75
            else:
                # 平衡 (pluck_pos == 1.0)
                guitar_vol = 0.60
                bass_vol = 0.40

            # 归一化
            total_vol = guitar_vol + bass_vol
            guitar_vol /= total_vol
            bass_vol /= total_vol

            # 8. 混合
            mixed = guitar_high * guitar_vol + bass_low * bass_vol

            # 9. 最终处理
            peak = np.max(np.abs(mixed))
            if peak > 0.01:
                mixed = mixed / peak * 0.96

            # 10. 输出
            samples_int = (mixed * 32767).astype(np.int16)
            buf = io.BytesIO()
            import wave
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(samples_int.tobytes())
            return buf.getvalue()

        elif instrument == "drums":
            from instruments import drums as engine_module
            midi_stream = io.BytesIO(file_bytes)
            result = engine_module.midi_to_audio(
                midi_stream, brightness, pluck_pos, body_mix, reflection, coupling
            )
            if result is None or not isinstance(result, tuple) or result[0] is None:
                return None
            return result[0]

        elif instrument == "full_band":
            from instruments import guitar, bass, drums
            import numpy as np
            from scipy import signal

            original_data = file_bytes

            # ========== 三轨独立渲染 ==========

            # 1. 吉他轨：主旋律，使用标准参数
            midi_stream_guitar = io.BytesIO(original_data)
            GUITAR_PLUCK = 0.25
            GUITAR_COUPLING = 0.005

            result_guitar = guitar.midi_to_audio(
                midi_stream_guitar,
                brightness * 1.05,  # 稍亮
                GUITAR_PLUCK,
                body_mix * 0.85,  # 共鸣稍弱
                reflection * 0.9,  # 混响稍弱
                GUITAR_COUPLING
            )

            # 2. 贝斯轨：低音基础，伴奏模式
            midi_stream_bass = io.BytesIO(original_data)
            BASS_PLUCK = 1.8

            result_bass = bass.midi_to_audio(
                midi_stream_bass,
                brightness * 0.85,  # 贝斯偏暗
                BASS_PLUCK,
                body_mix * 1.15,  # 贝斯共鸣强
                reflection * 0.85,  # 混响适中
                0.0,
                solo_mode=False  # 伴奏模式
            )

            # 3. 鼓组轨：节奏骨架
            midi_stream_drums = io.BytesIO(original_data)
            DRUMS_PLUCK = 1.2

            result_drums = drums.midi_to_audio(
                midi_stream_drums,
                brightness * 0.9,  # 鼓皮硬度适中
                DRUMS_PLUCK,
                body_mix * 0.6,  # 腔体共鸣适中
                reflection * 1.1,  # 混响稍强
                coupling  # 压缩感
            )

            if not all([result_guitar, result_bass, result_drums]):
                return None

            if not all([result_guitar[1] is not None, result_bass[1] is not None, result_drums[1] is not None]):
                return None

            guitar_samples = result_guitar[1]
            bass_samples = result_bass[1]
            drums_samples = result_drums[1]

            # 4. 统一长度
            max_len = max(len(guitar_samples), len(bass_samples), len(drums_samples))
            if len(guitar_samples) < max_len:
                guitar_samples = np.pad(guitar_samples, (0, max_len - len(guitar_samples)))
            if len(bass_samples) < max_len:
                bass_samples = np.pad(bass_samples, (0, max_len - len(bass_samples)))
            if len(drums_samples) < max_len:
                drums_samples = np.pad(drums_samples, (0, max_len - len(drums_samples)))

            # ========== 智能三轨混音 ==========

            # 5. 能量分析（分析各轨道的演奏密度）
            window_size = 48000  # 1秒窗口

            guitar_energy = np.convolve(guitar_samples ** 2, np.ones(window_size) / window_size, mode='same')
            bass_energy = np.convolve(bass_samples ** 2, np.ones(window_size) / window_size, mode='same')
            drums_energy = np.convolve(drums_samples ** 2, np.ones(window_size) / window_size, mode='same')

            # 归一化能量
            guitar_energy_norm = guitar_energy / (np.max(guitar_energy) + 1e-8)
            drums_energy_norm = drums_energy / (np.max(drums_energy) + 1e-8)

            # 6. 动态音量调制
            # 当吉他或鼓密集时，贝斯适当退后；稀疏时，贝斯填补空间
            combined_energy = (guitar_energy_norm + drums_energy_norm) / 2
            bass_ducking = 1.0 - (combined_energy * 0.25)  # 最多降低25%

            # 平滑处理
            from scipy.ndimage import gaussian_filter1d
            bass_ducking = gaussian_filter1d(bass_ducking, sigma=4800)  # 0.1秒平滑

            # 应用到贝斯
            bass_samples_ducked = bass_samples * bass_ducking

            # 7. 频段分离混音
            # 贝斯：40-250Hz
            sos_bass_lp = signal.butter(4, 250, 'lp', fs=48000, output='sos')
            bass_low = signal.sosfilt(sos_bass_lp, bass_samples_ducked)

            # 吉他：200Hz-8kHz
            sos_guitar_bp = signal.butter(2, [200, 8000], 'bp', fs=48000, output='sos')
            guitar_mid = signal.sosfilt(sos_guitar_bp, guitar_samples)

            # 鼓：全频段（但低频与贝斯共享，高频独占）
            sos_drums_hp = signal.butter(2, 100, 'hp', fs=48000, output='sos')
            drums_full = signal.sosfilt(sos_drums_hp, drums_samples)

            # 8. 音量平衡（使用 pluck_position 参数控制整体平衡）
            # pluck_position: 0.8-2.5
            # < 1.5: 偏向吉他主导
            # = 1.5: 平衡
            # > 1.5: 偏向节奏组（贝斯+鼓）

            base_guitar = 0.40
            base_bass = 0.32
            base_drums = 0.28

            if pluck_pos < 1.5:
                # 偏向吉他
                factor = (1.5 - pluck_pos) / 0.7  # 0-1
                guitar_vol = base_guitar * (1.0 + factor * 0.3)
                bass_vol = base_bass * (1.0 - factor * 0.2)
                drums_vol = base_drums * (1.0 - factor * 0.15)
            elif pluck_pos > 1.5:
                # 偏向节奏组
                factor = (pluck_pos - 1.5) / 1.0  # 0-1
                guitar_vol = base_guitar * (1.0 - factor * 0.25)
                bass_vol = base_bass * (1.0 + factor * 0.3)
                drums_vol = base_drums * (1.0 + factor * 0.25)
            else:
                # 平衡
                guitar_vol = base_guitar
                bass_vol = base_bass
                drums_vol = base_drums

            # 归一化
            total_vol = guitar_vol + bass_vol + drums_vol
            guitar_vol /= total_vol
            bass_vol /= total_vol
            drums_vol /= total_vol

            # 9. 混合三轨
            mixed = (
                    guitar_mid * guitar_vol +
                    bass_low * bass_vol +
                    drums_full * drums_vol
            )

            # 10. 母带压缩（轻微，保留动态）
            # Soft knee compressor
            threshold = 0.7
            ratio = 3.0
            for i in range(len(mixed)):
                if abs(mixed[i]) > threshold:
                    sign = 1.0 if mixed[i] > 0 else -1.0
                    excess = abs(mixed[i]) - threshold
                    mixed[i] = sign * (threshold + excess / ratio)

            # 11. 最终归一化
            peak = np.max(np.abs(mixed))
            if peak > 0.01:
                mixed = mixed / peak * 0.96

            # 12. 输出
            samples_int = (mixed * 32767).astype(np.int16)
            buf = io.BytesIO()
            import wave
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(samples_int.tobytes())
            return buf.getvalue()

        else:  # piano
            # Piano logic remains unchanged, assume it doesn't need solo_mode arg or it handles **kwargs
            # Ideally check piano module, but for now just pass standard args
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
        import traceback
        traceback.print_exc()
        return None


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
                display: flex; gap: 16px; height: 90px;
                box-sizing: border-box; align-items: center;
            }}
            .play-section {{ flex-shrink: 0; }}
            .play-btn {{
                width: 48px; height: 48px; border-radius: 50%;
                background: #ff4b4b; border: none; cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                transition: all 0.2s;
            }}
            .play-btn:hover {{ background: #ff6b6b; transform: scale(1.05); }}
            .play-btn svg {{ fill: white; width: 20px; height: 20px; margin-left: 3px; }}
            .play-btn.playing svg {{ margin-left: 0; }}
            .wave-section {{
                flex-grow: 1; height: 100%; position: relative;
                display: flex; flex-direction: column; justify-content: center;
                background: rgba(255,255,255,0.02); border-radius: 8px; overflow: hidden;
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
                transform: translate(-50%,-50%); font-size: 11px; color: #666; letter-spacing: 1px;
            }}
            .controls-section {{
                width: 140px; flex-shrink: 0;
                display: flex; flex-direction: column; justify-content: space-between;
                height: 100%; padding-left: 10px; border-left: 1px solid #222;
            }}
            .time-display {{
                font-family: 'SF Mono', 'Consolas', monospace;
                font-size: 13px; color: #ff4b4b; text-align: right; font-weight: 500;
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
                border-radius: 50%; background: #ccc; margin-top: -3px; cursor: pointer;
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
                window.togglePlay = function() {{ wavesurfer.playPause(); isPlaying = !isPlaying; updateBtn(); }};
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


# --- 7. 侧边栏 (彻底修复版) ---
with st.sidebar:
    st.title("音色实验室")
    st.caption("在调参后请手动重新生成")
    st.markdown("---")

    instrument = st.session_state.get('instrument', 'guitar')

    # 渲染参数标题
    title_map = {
        "guitar": "🎸 吉他参数",
        "bass": "🎸 贝斯参数",
        "piano": "🎹 钢琴参数",
        "guitar_bass": "🎸+🎸 混合参数",
        "drums": "🥁 鼓组参数",
        "full_band": "🎸🥁 乐队参数"
    }
    st.subheader(title_map.get(instrument, "参数"))

    # 获取当前乐器的参数配置
    ranges = PARAM_RANGES[instrument]
    labels = PARAM_LABELS[instrument]

    # 渲染每个参数的滑块
    for param in ["brightness", "pluck_position", "body_mix", "reflection", "coupling"]:
        # 检查是否需要显示此参数（贝斯的coupling不显示）
        if labels[param] is None:
            continue

        min_val, max_val, step = ranges[param]
        current_val = st.session_state.get(param, DEFAULT_PARAMS[instrument][param])

        # 确保当前值在范围内
        if current_val < min_val or current_val > max_val:
            current_val = DEFAULT_PARAMS[instrument][param]
            st.session_state[param] = current_val

        # 渲染滑块
        st.slider(
            labels[param],
            min_val,
            max_val,
            value=current_val,
            step=step,
            key=param
        )

    st.markdown("---")
    if st.button("🔄 恢复默认音色", use_container_width=True):
        st.session_state.reset_tone = True
        st.rerun()

# --- 8. 主界面 (带沉浸模式开关) ---

# 1. 定义布局：左边占位，右边放开关
# 我们利用 columns 把开关挤到最右边，模拟"内嵌"在标题栏上方的效果
col_header_spacer, col_header_toggle = st.columns([6, 1.2])

with col_header_toggle:
    # 这里的 key 保证了状态会被记住
    is_transparent = st.toggle("👁️ 沉浸模式", value=False, help="让soyo和猫猫的脸露出来")

# 2. 定义默认（有颜色）的样式
text_color = "white"
text_shadow = "none"
border_style = "none"  # 默认无边框

if instrument == 'guitar':
    icon = "🎸"
    title = "Karplus-Strong 吉他工作室"
    subtitle = "物理建模 · MIDI → 原声吉他 · 高保真合成"
    gradient = "linear-gradient(90deg, #FF9EAA, #FFFFFF)"
    text_color = "#333333"

elif instrument == 'bass':
    icon = "🎸"
    title = "Karplus-Strong 贝斯工作室"
    subtitle = "低频物理建模 · MIDI → 电贝斯 · 厚重低音"
    gradient = "linear-gradient(90deg, #8B5E4F, #D97757, #EAD1C3)"
    text_shadow = "1px 1px 2px rgba(0,0,0,0.3)"

elif instrument == 'guitar_bass':
    icon = "🎸🎸"
    title = "Karplus-Strong 混合工作室"
    subtitle = "吉他+贝斯 · 自动音域分配 · 全频段覆盖"
    gradient = "linear-gradient(135deg, #FB8DA0 0%, #FFC0CB 50%, #D97757 50%, #8B5E4F 100%)"
    text_shadow = "0px 2px 4px rgba(0,0,0,0.6)"

elif instrument == 'drums':
    icon = "🥁"
    title = "Karplus-Strong 鼓组工作室"
    subtitle = "节奏建模 · MIDI → 原声鼓组 · 动态打击"
    # 深紫渐变（和钢琴一个气质，但更有力量）
    gradient = "linear-gradient(90deg, #1b1028, #2e1a47, #3d2466)"
    text_shadow = "0 2px 6px rgba(0,0,0,0.8)"

elif instrument == 'full_band':
    icon = "🎤🎸🎸🎸🥁"
    title = "Karplus-Strong 组一辈子乐队"
    subtitle = "全乐器自动编配 · 吉他+贝斯+鼓 · 全频段覆盖"
    # 深蓝 → 浅蓝，不要绿色
    gradient = "linear-gradient(90deg, #0b2239, #123a5a, #1e5f8a, #4da3d9)"
    text_shadow = "0 2px 6px rgba(0,0,0,0.7)"


else:  # piano
    icon = "🎹"
    title = "Karplus-Strong 钢琴工作室"
    subtitle = "多弦耦合 · MIDI → 三角钢琴 · 音乐厅混响"
    gradient = "linear-gradient(90deg,#1a1a2e,#16213e,#0f3460)"

# 3. 核心逻辑：如果开启了透明模式，覆盖上面的样式
# 极简高透明方案
if is_transparent:
    # --- 这里是核心修改点 ---
    gradient = "rgba(255, 255, 255, 0.03)"  # 和信息卡一样的透明度
    border_style = "1px solid rgba(255, 255, 255, 0.08)"  # 和信息卡一致
    text_color = "#ffffff"
    text_shadow = "0 2px 8px rgba(0,0,0,0.8)"
# 4. 渲染标题卡片
style_block = f"""
background: {gradient};
padding: 18px 28px;
border-radius: 12px;
color: {text_color};
text-shadow: {text_shadow};
border: {border_style};
margin-bottom: 20px;
transition: all 0.3s ease;
"""

# 移除毛玻璃效果

st.markdown(f"""
<div style='{style_block}'>
    <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
            <h2 style='margin:0;color:{text_color};text-shadow:{text_shadow};'>
                {icon} {title}
            </h2>
            <p style='margin:0;opacity:0.9;'>{subtitle}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 乐器切换按钮
cols = st.columns(6)

buttons = [
    ("🎤🎸🎸🎸🥁 乐队", "full_band"),
    ("🎸 吉他", "guitar"),
    ("🎸 贝斯", "bass"),
    ("🎸+🎸 混合", "guitar_bass"),
    ("🥁 鼓组", "drums"),
    ("🎹 钢琴", "piano"),
]

for col, (label, key) in zip(cols, buttons):
    with col:
        if st.button(
                label,
                type="primary" if instrument == key else "secondary",
                use_container_width=True,
        ):
            st.session_state.instrument = key
            st.rerun()

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

col_main, col_output = st.columns([1, 1], gap="large")

with col_main:
    st.markdown("### 1. 选择 MIDI 来源")

    source_options = ["😡为什么要演奏春日影", "💿 内置 MIDI 库", "📂上传自己的 MIDI"]
    mode = st.radio("MIDI 来源", source_options, horizontal=True, label_visibility="collapsed")

    uploaded_file = None
    if mode == "📂上传自己的 MIDI":
        f = st.file_uploader("上传 MIDI 序列", type=["mid", "midi"], label_visibility="collapsed")
        if f:
            uploaded_file = io.BytesIO(f.read())
            uploaded_file.name = f.name
    elif mode == "💿 内置 MIDI 库":
        local_files = get_local_midi_files()
        if not local_files:
            st.warning("⚠️ assets 文件夹下没有找到 MIDI 文件。")
        else:
            file_options = {os.path.basename(p): p for p in local_files}
            selected_name = st.selectbox("请选择一首歌曲:", list(file_options.keys()))
            if selected_name:
                selected_path = file_options[selected_name]
                try:
                    with open(selected_path, "rb") as f:
                        uploaded_file = io.BytesIO(f.read())
                        uploaded_file.name = selected_name
                except Exception as e:
                    st.error(f"无法读取文件: {e}")
    else:
        try:
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
                st.warning("⚠️ 默认 MIDI 文件未找到。")
        except Exception:
            st.warning("⚠️ 读取默认文件失败")

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-row"><span>📄 文件:</span> <span class="metric-val">{uploaded_file.name}</span></div>
            <div class="metric-row"><span>🎚️ 采样率:</span> <span class="metric-val">48000 Hz</span></div>
            <div class="metric-row"><span>🎼 乐器:</span> <span class="metric-val">{instrument.upper()}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 2. 执行渲染")

        if instrument == "guitar":
            button_text = "🎸**GuitarHero，启动！**"
            status_text = "正在进行吉他弦振动模拟..."
            init_text = "初始化 128 根虚拟吉他弦..."
            parse_text = "解析 MIDI 事件并进行活跃弦追踪..."
        elif instrument == "bass":
            button_text = "🎸**BassMaster，启动！**"
            status_text = "正在进行贝斯低频建模..."
            init_text = "初始化贝斯低音弦（E1-C4）..."
            parse_text = "解析 MIDI 事件并渲染厚重低音..."
        elif instrument == "guitar_bass":
            button_text = "🎸+🎸**我们联合！**"
            status_text = "正在进行双轨渲染..."
            init_text = "初始化吉他+贝斯混合引擎..."
            parse_text = "自动分配音域并混合渲染..."
        elif instrument == "drums":
            button_text = "🥁**DrumMaster，启动！**"
            status_text = "正在进行架子鼓模拟..."
            init_text = "初始化架子鼓引擎..."
            parse_text = "解析 MIDI 事件并生成打击乐..."
        elif instrument == "full_band":
            button_text = "🎤+🎸+🎸+🎸🥁**组一被子乐队！**"
            status_text = "正在进行全轨渲染..."
            init_text = "初始化吉他+贝斯+架子鼓混合引擎..."
            parse_text = "自动分配音域并混合渲染..."
        else:  # piano
            button_text = "🎹**PianoMaster，启动！**"
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
                    status.update(label="✅ 音频加载成功", state="complete", expanded=False)
                else:
                    st.session_state.render_done = False
                    status.update(label="❌ 渲染失败", state="error", expanded=False)

        if mode == "😡为什么要演奏春日影" and st.session_state.get("render_done"):
            st.components.v1.html(get_gif_button_html(), height=60)

with col_output:
    st.markdown("### 3. 输出与试听")

    if 'audio_out' in st.session_state and st.session_state.audio_out:
        render_sync_player(st.session_state.audio_out)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
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
            display: flex; align-items: center; justify-content: center; 
            color: #666; background: rgba(255,255,255,0.01);">
            等待渲染任务完成...
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>© 2026 青空 Karplus-Strong Studio | 基于CS61B Java 原版逻辑复刻</p>",
    unsafe_allow_html=True
)
