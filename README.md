---

# 🎸 Karplus-Strong Studio | MyGO!!!!! Fan Creation Platform

**Based on Physical Modeling Synthesis & AI-Driven Narrative**  
基于物理建模合成与 AI 驱动叙事的 MyGO!!!!! 同人二创平台

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT_%28Code_Only%29-green.svg?style=flat-square)](LICENSE)

**Karplus-Strong Studio** 是一个集成了**物理音频合成**、**大语言模型 (LLM) 剧本生成**以及**4K 下落式音游**的综合性 Web 应用。本项目以《BanG Dream! It's MyGO!!!!!》为主题，旨在探索技术与二创结合的可能性。

---

## ⚠️ 版权声明与免责条款 (Copyright & Disclaimer)

**本项目为非营利性技术交流与同人二创作品，严禁用于任何商业用途。**

在使用本项目前，请务必仔细阅读以下版权说明。

### 1. 核心 IP 版权
*   本项目中出现的所有《BanG Dream! It's MyGO!!!!!》相关的**角色名称、立绘图像、背景设定、原作剧情**等知识产权（IP），均归属于 **Bushiroad Inc.**、**Craft Egg Inc.** 及相关权利人所有。
*   本项目仅作为粉丝技术交流使用，不代表官方立场。

### 2. MIDI 音乐版权 (MIDI Usage Policy)
本项目包含 MIDI 文件解析与播放功能，关于 MIDI 文件的使用请遵循以下原则：
*   **内置/演示 MIDI**：项目中若包含任何内置 MIDI 文件（如 `春日影-mygo.mid`），仅作为算法演示用途。其原曲的**词曲著作权**归原作者及版权方所有。
*   **用户上传 MIDI**：
    *   用户通过“上传模式”导入的 MIDI 文件，其版权责任由用户自行承担。
    *   请勿上传、传播或演示未获得授权的商业级 MIDI 文件。
    *   本项目生成的音频是通过物理建模算法（Karplus-Strong）实时计算生成的，而非播放原版录音，但在公开传播生成结果时，仍需尊重原曲作者的著作权。

### 3. 图像与音频资源
*   `assets/` 文件夹下的**图片素材**（如角色遮罩、背景图）及**语音片段**均来源于网络或官方素材的二创剪辑。
*   这些资源仅供本地运行测试使用，**请勿将包含受版权保护素材的代码库直接打包发布到公共平台**。如果您是开发者，建议在分发时移除 `assets` 下的版权文件。

### 4. 代码许可
*   本项目的**源代码**（`.py` / `.js` 逻辑部分）遵循 MIT 许可协议开源。您可以自由学习、修改算法逻辑。但此许可**不包含**项目中的任何美术资源和音乐文件。

---

## ✨ 功能特性 (Features)

### 🎵 物理建模音频引擎
*   **Karplus-Strong 算法**：不使用任何采样音源，纯数学算法模拟吉他、贝斯、钢琴的弦振动。
*   **多乐器支持**：支持 Guitar (爱音/乐奈)、Bass (素世)、Drums (立希)、Piano (灯/键盘)。
*   **参数化调节**：可实时调整 `亮度`、`拨弦位置`、`共鸣度`、`空间反射` 等物理参数。

### 🎭 AI 沉浸式小剧场
*   **DeepSeek/OpenAI 驱动**：基于 RAG (检索增强生成) 的动态剧本生成。
*   **多模式体验**：
    *   `日常` / `旅游` / `二创研讨` 等多种场景。
    *   **Tavily 联网搜索**：AI 可读取现实世界新闻或二创热梗，让角色“打破第四面墙”进行讨论。
    *   **Folium 地图联动**：在地图上选择日本真实坐标，生成基于当地风土人情的旅行剧本。

### 🎮 4K 下落式音游 (Rhythm Game)
*   **实时谱面生成**：后端解析 MIDI 文件的音高、力度和时间戳，算法自动生成 4K 谱面。
*   **专业手感优化**：包含防纵连 (Anti-Jack)、重音双押补偿、伪随机交互流生成。
*   **Web 游戏引擎**：基于 HTML5 Canvas 的高性能渲染，支持 `S D J K` 键位，包含连击 (Combo)、粒子特效、屏幕震动等 Juice 效果。
*   **本地成绩系统**：基于 `localStorage` 记录每首歌的最高分 (100万分制) 及 AP/FC 徽章。

---

## 🛠️ 安装与运行 (Installation)

### 1. 环境准备
确保已安装 Python 3.10 或更高版本。

```bash
# 克隆仓库
git clone https://github.com/your-username/karplus-strong-studio.git
cd karplus-strong-studio

# 安装依赖
pip install -r requirements.txt
```

### 2. 资源文件配置
由于版权原因，仓库可能不包含部分素材。请确保根目录下有 `assets` 文件夹，并按以下结构放入文件（文件名需与代码中 `INSTRUMENT_MASKS` 和 `INSTRUMENT_VOICES` 对应）：

```text
assets/
├── masks/             # 角色立绘遮罩图 (png)
│   ├── mygo.png
│   ├── 爱猫.png
│   └── ...
├── voices/            # 角色语音 (mp3)
├── covers/            # 歌曲封面 (jpg/png，文件名需与midi一致)
└── *.mid              # 演示用 MIDI 文件
```

### 3. API 密钥配置
在项目根目录创建 `.streamlit/secrets.toml` 文件：

```toml
# .streamlit/secrets.toml

# LLM 配置 (用于剧本生成)
LLM_API_KEY = "sk-xxxxxxxx"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"

# Tavily 配置 (用于联网搜索功能)
TAVILY_API_KEY = "tvly-xxxxxxxx"
```

### 4. 启动应用

```bash
streamlit run mygo.py
```

---

## 🎹 操作指南

1.  **Landing Page**: 选择想要互动的角色（对应不同乐器）。
2.  **Params**: (可选) 调节物理建模参数，改变音色。
3.  **MIDI Selection**:
    *   上传自己的 MIDI 文件。
    *   或选择内置曲目。
4.  **Game/Theatre Mode**:
    *   **🎭 剧场模式**: 开启 AI 剧场，配置场景（旅游/日常/Meta），AI 将根据音乐生成剧本并演绎。
    *   **🎮 音游模式**: 开启 4K 音游，选择难度 (Normal/Challenge)，使用键盘 `S D J K` 进行游玩。
5.  **Render**: 点击渲染，等待 Python 后端合成音频及谱面，进入沉浸式播放/游玩界面。

---

## 🤝 贡献与致谢

*   **Audio Engine**: 基于 Karplus-Strong 弦乐合成算法。感谢UCB CS61B的优质课程。
*   **UI Design**: 采用了 Glassmorphism (玻璃拟态) 风格设计。
*   **Community**: 感谢所有 BanG Dream! 社区的二创作者提供的灵感。

---
