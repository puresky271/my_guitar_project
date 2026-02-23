
from openai import OpenAI
import json
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
import random
import folium
from streamlit_folium import st_folium



# ========== 音游模式：专业 4K 谱面生成引擎 (全面 4K 交互流优化版) ==========
def generate_beatmap(midi_bytes, difficulty='normal'):
    """基于时间窗聚合与专业键型库的 4K 谱面生成器"""
    try:
        import mido
        import random
    except ImportError:
        st.warning("缺少 mido 库，无法生成音游谱面")
        return []

    try:
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        best_track = None
        max_notes = 0

        # 1. 寻找主旋律/音符最多的轨道
        for track in mid.tracks:
            cnt = sum(1 for msg in track if msg.type == 'note_on' and msg.velocity > 0)
            if cnt > max_notes:
                max_notes = cnt
                best_track = track

        if not best_track: return []

        # 2. 提取所有原始音符并计算绝对时间
        tempo = 500000
        ticks_per_beat = mid.ticks_per_beat
        current_time = 0.0
        raw_notes = []

        for msg in best_track:
            current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                raw_notes.append({'t': current_time, 'note': msg.note, 'vel': msg.velocity})

        if not raw_notes: return []

        # 3. 时间窗聚合 (Chord Grouping)
        TIME_TOLERANCE = 0.02  # 20ms 内的音符打包视为同一拍
        chords = []
        for n in raw_notes:
            if not chords:
                chords.append({'t': n['t'], 'notes': [n['note']], 'max_vel': n['vel']})
            else:
                last_chord = chords[-1]
                if n['t'] - last_chord['t'] <= TIME_TOLERANCE:
                    last_chord['notes'].append(n['note'])
                    last_chord['max_vel'] = max(last_chord['max_vel'], n['vel'])
                else:
                    chords.append({'t': n['t'], 'notes': [n['note']], 'max_vel': n['vel']})

        # 4. 难度配置
        if difficulty == 'challenge':
            MIN_GAP = 0.08  # 允许极高密度流 (16分/32分音符)
            JACK_LIMIT = 0.18  # 纵连限制(极快连续音符禁止同轨)
            ACCENT_VELOCITY = 100  # 触发重音补偿的力度阈值 (调高，减少随机双押)
        else:  # normal
            MIN_GAP = 0.16  # 过滤高频碎音，保留主节奏
            JACK_LIMIT = 0.30  # 宽松的防纵连，避免手部疲劳
            ACCENT_VELOCITY = 115  # 极难触发重音补偿

        # 4K 专业音游双押库 (0:S, 1:D, 2:J, 3:K)
        DOUBLE_PATTERNS = [(0, 3), (1, 2), (0, 1), (2, 3), (0, 2), (1, 3)]

        beatmap = []
        last_time = -999.0
        last_lanes = []

        # 使用歌曲信息生成固定随机种子，保证同一首曲子谱面永远一致
        seed_val = sum(n['note'] for n in raw_notes[:50]) + len(raw_notes)
        rng = random.Random(seed_val)

        # 5. 核心制谱循环
        for chord in chords:
            t = chord['t']

            # 密度过滤
            if t - last_time < MIN_GAP:
                continue

            num_notes = len(chord['notes'])
            vel = chord['max_vel']

            # 【核心改动1】：高速流强制降级为单点
            # 如果当前音符距离上一个极近(<0.18s)，无论原曲有几个音，强制变为1个音！
            # 这样可以在副歌高潮段形成纯粹的 4K 瀑布流 (Stream)，而不是卡手的连续多押
            if t - last_time < 0.18:
                num_notes = 1
            else:
                # 【核心改动2】：克制化重音补偿
                # 只有在前方有足够长空白期(>0.4s)且力度极大时，单点才升格为双押
                if num_notes == 1 and vel >= ACCENT_VELOCITY and (t - last_time) > 0.4:
                    num_notes = 2

                # 限制多押规模：绝大多数情况最多双押。只有Challenge模式中极大力度才允许三押
                if num_notes > 2:
                    if difficulty == 'challenge' and vel >= 110 and (t - last_time) > 0.5:
                        num_notes = 3
                    else:
                        num_notes = 2

            chosen_lanes = []

            # 分配轨道
            if num_notes == 1:
                pitch = chord['notes'][0]
                if pitch < 60:
                    base = 0
                elif pitch < 67:
                    base = 1
                elif pitch < 74:
                    base = 2
                else:
                    base = 3
                lane = base

                # 【核心改动3】：专业防纵连与双手交互流 (Trill/Stream)
                if t - last_time < JACK_LIMIT and last_lanes:
                    if lane in last_lanes:
                        # 强制换手交互：如果上次是左手(0,1)，这次优先分配给右手(2,3)
                        last_lane = last_lanes[0]
                        if last_lane <= 1:
                            available = [2, 3]
                        else:
                            available = [0, 1]
                        lane = rng.choice(available)
                chosen_lanes.append(lane)

            elif num_notes == 2:
                valid_patterns = DOUBLE_PATTERNS.copy()
                # 如果间隔较短，双押尽量不使用和上次完全一样的轨道
                if t - last_time < 0.35 and last_lanes:
                    valid_patterns = [p for p in valid_patterns if not set(p).issubset(set(last_lanes))]
                if not valid_patterns: valid_patterns = DOUBLE_PATTERNS
                chosen_lanes = list(rng.choice(valid_patterns))

            elif num_notes == 3:
                # 三押固定使用漏一键的配置
                missing_lane = rng.randint(0, 3)
                chosen_lanes = [l for l in range(4) if l != missing_lane]

            else:  # 保底
                chosen_lanes = [0, 1, 2, 3]

            # 写入谱面数据
            for l in chosen_lanes:
                beatmap.append({"t": round(t, 3), "l": l, "hit": False})

            last_time = t
            last_lanes = chosen_lanes

        return beatmap
    except Exception as e:
        print(f"MIDI Error: {e}")
        return []
# ── 前端 31 个精选坐标点 到 具体景点列表的映射 ──
DESTINATION_MAP = {
    "札幌": ["札幌大通公园", "札幌钟楼", "北海道大学银杏大道", "札幌伏见稻荷神社", "定山溪温泉", "旭山动物园"],
    "小樽": ["小樽运河", "小樽堺町通"],
    "富良野": ["北海道富良野薰衣草田", "富良野薰衣草田", "美瑛青池", "美瑛拼布之路"],
    "函馆": ["函馆山夜景", "五棱郭公园", "函馆元町教堂群"],
    "青森·十和田": ["奥入濑溪流", "十和田湖", "弘前城", "八甲田山", "白神山地"],
    "银山温泉": ["银山温泉", "山寺（立石寺）", "最上川", "藏王树冰", "藏王御釜", "藏王温泉"],
    "松岛": ["松岛", "瑞巖寺", "仙台城迹（青叶城）", "秋保大瀑布"],
    "会津·五色沼": ["大内宿", "鹤城", "五色沼", "猪苗代湖"],
    "东京市区": list(TOKYO_SPOTS.keys()),
    "镰仓·江之岛": list(KAMAKURA_SPOTS.keys()),
    "箱根": list(HAKONE_SPOTS.keys()),
    "富士山·河口湖": list(FUJI_SPOTS.keys()),
    "名古屋": ["名古屋城", "热田神宫", "大须商店街", "名古屋港水族馆"],
    "高山·飞驒": ["高山老街（三町通）", "高山阵屋", "飞驒国分寺", "飞驒之里"],
    "白川乡": ["白川乡合掌村", "荻町城迹展望台"],
    "金泽": ["兼六园", "金泽城公园", "近江町市场", "东茶屋街", "21世纪美术馆"],
    "京都": list(KYOTO_SPOTS.keys()),
    "大阪": ["大阪城", "道顿堀", "心斋桥筋商店街", "通天阁"],
    "奈良": ["奈良公园", "东大寺", "春日大社"],
    "神户": ["神户港塔", "北野异人馆街", "有马温泉"],
    "熊野古道": ["熊野古道", "和歌山城", "白良滨海滩"],
    "高松·直岛": ["栗林公园", "金刀比罗宫", "直岛", "中津万象园丸龟城"],
    "松山·道后": ["道后温泉本馆", "松山城", "下滩站", "内子町"],
    "高知·四万十": ["弘人市场", "桂浜", "四万十川", "足摺岬"],
    "德岛·祖谷": ["鸣门漩涡", "阿波舞会馆", "祖谷蔓桥", "大步危小步危"],
    "福冈·博多": ["太宰府天满宫", "栉田神社", "博多运河城", "中洲屋台"],
    "长崎": ["稲佐山展望台", "哥拉巴园", "大浦天主堂", "豪斯登堡"],
    "熊本·阿苏": ["熊本城", "阿苏火山", "草千里", "黑川温泉", "高千穗峡"],
    "大分·由布院": ["别府地狱巡游", "由布院", "金鳞湖"],
    "鹿儿岛·樱岛": ["樱岛", "仙岩园", "指宿砂浴", "雾岛", "雾岛神宫"],
    "冲绳本岛": list(OKINAWA_SPOTS.keys())
}

# 目的地坐标映射
REGION_COORDINATES = {
    "札幌": (43.0618, 141.3545),
    "小樽": (43.1907, 140.9947),
    "富良野": (43.3421, 142.3832),
    "函馆": (41.7687, 140.7288),
    "青森·十和田": (40.8222, 140.7474),
    "银山温泉": (38.5705, 140.5303),
    "松岛": (38.3713, 141.0664),
    "会津·五色沼": (37.4949, 139.9297),
    "东京市区": (35.6895, 139.6917),
    "镰仓·江之岛": (35.3191, 139.5467),
    "箱根": (35.2324, 139.1069),
    "富士山·河口湖": (35.3606, 138.7274),
    "名古屋": (35.1815, 136.9066),
    "高山·飞驒": (36.1408, 137.2513),
    "白川乡": (36.2566, 136.9043),
    "金泽": (36.5613, 136.6562),
    "京都": (35.0116, 135.7681),
    "大阪": (34.6937, 135.5023),
    "奈良": (34.6851, 135.8048),
    "神户": (34.6901, 135.1955),
    "熊野古道": (33.5097, 135.9141),
    "高松·直岛": (34.3428, 134.0466),
    "松山·道后": (33.8392, 132.7655),
    "高知·四万十": (33.5581, 133.5312),
    "德岛·祖谷": (34.0704, 134.5548),
    "福冈·博多": (33.5902, 130.4017),
    "长崎": (32.7503, 129.8777),
    "熊本·阿苏": (32.8031, 130.7079),
    "大分·由布院": (33.2647, 131.3571),
    "鹿儿岛·樱岛": (31.5966, 130.5571),
    "冲绳本岛": (26.2124, 127.6809)
}

MYGO_CHARACTERS = {
    "灯": {"color": "#93c5fd", "align": "left", "emoji": "🎤"},
    "爱音": {"color": "#f9a8d4", "align": "right", "emoji": "🎸"},
    "乐奈": {"color": "#86efac", "align": "left", "emoji": "🎸"},
    "素世": {"color": "#fde68a", "align": "right", "emoji": "🎸"},
    "立希": {"color": "#c4b5fd", "align": "left", "emoji": "🥁"},
}

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
    .main { background-color: transparent !important; color: #f0f2f6; }
    [data-testid="stAppViewContainer"] { background-color: transparent !important; }
    [data-testid="stSidebar"] { background-color: #161920; border-right: 1px solid #303030; }

    /* =================================
       UI 组件美化
       ================================= */
    h1, h2, h3 { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-weight: 600; }
    .stButton>button {
        border-radius: 6px; font-weight: 600;
        border: 1px solid rgba(255, 75, 75, 0.5);
        background-color: rgba(255, 75, 75, 0.1);
        color: #ff4b4b; transition: all 0.2s ease-in-out; height: 45px;
    }
    .stButton>button:hover {
        background-color: #ff4b4b; color: white;
        border-color: #ff4b4b; transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }

    /* Search Form UI Reset */
    [data-testid="stForm"] { background: transparent; border: none; padding: 0; }

    .metric-container {
        background: rgba(255, 255, 255, 0.03); 
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
        display: flex; flex-direction: column; gap: 8px;
    }
    .metric-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.95rem; color: #aaa; }
    .metric-val { font-family: 'SF Mono', 'Consolas', monospace; color: #fff; font-weight: 500; }
    [data-testid="stDownloadButton"] > button,
    div[data-testid="stHorizontalBlock"] .stButton > button {
        height: 48px !important; display: flex !important; align-items: center !important;
        justify-content: center !important; border-radius: 8px !important; transition: all 0.3s ease !important;
    }
    [data-testid="stDownloadButton"] > button { height: 48px !important; width: 100%; }
    div[data-testid="column"] button div p { font-size: 14px !important; font-weight: 500 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 背景图加载逻辑（含角色高亮层）---
INSTRUMENT_MASKS = {
    "full_band": "assets/masks/mygo.png",
    "guitar": "assets/masks/爱猫.png",
    "bass": "assets/masks/素世.png",
    "drums": "assets/masks/立希1.png",
    "piano": None,
}

INSTRUMENT_VOICES = {
    "full_band": [("assets/voices/咕咕嘎嘎.mp3", 1), ("assets/voices/灵感菇.mp3", 1)],
    "guitar": [("assets/voices/唐哭.mp3", 1), ("assets/voices/唐笑.mp3", 1), ("assets/voices/有趣的女人.mp3", 1)],
    "bass": [("assets/voices/希腊奶.mp3", 1)],
    "drums": [("assets/voices/我要拉黑他.mp3", 1)],
    "piano": [],
}

BASS_SPECIAL_VOICE = "assets/voices/为什么要演奏春日影.mp3"


def pick_voice(instrument: str) -> str | None:
    candidates = INSTRUMENT_VOICES.get(instrument, [])
    candidates = [(p, w) for p, w in candidates if os.path.exists(p)]
    if not candidates: return None
    paths, weights = zip(*candidates)
    return random.choices(paths, weights=weights, k=1)[0]


def inject_voice(path: str | None):
    if not path or not os.path.exists(path): return
    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    components.html(
        f"""<audio autoplay style="display:none"><source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>""",
        height=1, scrolling=False)


# ========== 搜索功能 (Tavily AI) ==========
def search_web(query, max_results=15):
    """使用 Tavily AI 搜索，限定国内网站并过滤负面内容"""
    api_key = st.secrets.get("TAVILY_API_KEY")
    if not api_key:
        return [{"title": "API Key 缺失", "content": "请在 .streamlit/secrets.toml 中配置 TAVILY_API_KEY", "url": "#"}]

    try:
        client = TavilyClient(api_key=api_key)

        # 优化检索质量：如果用户输入较短或没有带企划名称，强制附加 MyGO/BanG Dream 关键词保障相关性
        keywords = ["MyGO", "Ave Mujica", "BanG Dream", "邦邦", "祥子", "素世", "灯", "立希", "爱音", "乐奈"]
        if not any(k in query for k in keywords):
            query = f"{query} MyGO OR BanG Dream"

        # 强制限定国内域及常见二次元域（新增萌娘百科增强设定检索）
        safe_query = f"{query} (site:bilibili.com OR site:weibo.com OR site:tieba.baidu.com OR site:nga.cn OR site:moegirl.org.cn)"
        response = client.search(
            query=safe_query,
            search_depth="advanced",
            max_results=max_results,
            exclude_domains=["twitter.com", "facebook.com", "youtube.com"]
        )

        # 二次安全过滤（扩充常见网络垃圾与节奏词汇）
        blacklist = ["黑料", "炎上", "丑闻", "恶心", "崩坏", "抄袭", "辱华", "政治", "死", "节奏", "挂人", "避雷",
                     "争议"]
        safe_results = []
        for r in response.get('results', []):
            full_text = r.get("title", "") + r.get("content", "")
            if not any(bad in full_text for bad in blacklist):
                safe_results.append(r)
        return safe_results
    except Exception as e:
        return [{"title": "搜索发生错误", "content": str(e), "url": "#"}]


# ========== 动态场景生成器 ==========

def generate_fullband_scene(category_override=None):
    def is_plausible(category, theme, location, num_people):
        weird_pairs = {
            "看演唱会": ["图书馆", "练习室", "学校教室", "便利店", "宿舍"],
            "一起去卡拉OK": ["图书馆", "浅草寺", "明治神宫", "奈良公园", "富士山"],
            "玩桌游": ["livehouse后台", "箱根温泉街", "富士山", "严岛神社（宫岛）"],
            "露营": ["练习室", "学校教室", "图书馆", "卡拉OK包厢", "商场"],
            "去水族馆": ["练习室", "卡拉OK包厢", "livehouse后台", "图书馆"],
            "去温泉": ["学校教室", "练习室", "图书馆", "便利店", "车站"],
            "赏花": ["livehouse后台", "卡拉OK包厢", "游戏中心", "宿舍"],
            "看电影": ["图书馆", "练习室", "便利店", "明治神宫"],
            "参加学园祭": ["箱根温泉街", "冲绳美之海水族馆", "富士山", "某人家中"],
            "排练": ["水族馆", "电影院", "神社", "东京塔", "浅草寺", "明治神宫", "富士山", "奈良公园", "京都清水寺"],
            "堆雪人": ["livehouse后台", "卡拉OK包厢", "食堂", "便利店"],
            "去海边玩": ["练习室", "图书馆", "宿舍", "卡拉OK包厢", "商场"],
        }
        if theme in weird_pairs and location in weird_pairs[theme]: return False
        need_at_least = {"玩桌游": 3, "过生日": 3, "庆祝节日": 3, "一起去卡拉OK": 2}
        if theme in need_at_least and num_people < need_at_least[theme]: return False
        meta_ok = set(INDOOR_LOCATIONS) | {"天台", "屋顶", "咖啡厅", "便利店"}
        if category == "meta" and location not in meta_ok: return False
        return True

    category = category_override if category_override in ("real", "fantasy", "meta", "outing") else random.choice(
        ["real", "fantasy", "meta"])
    members = ["爱音", "素世", "灯", "立希", "乐奈"]

    for _ in range(20):
        detail_hint = ""
        if category == "outing":
            theme = random.choice(OUTING_THEMES)
            location = random.choice(random.choice([TOKYO_LOCATIONS, JAPAN_LOCATIONS]))
        elif category == "real":
            theme = random.choice(REAL_THEMES)
            forced = THEME_LOCATION_TYPE.get(theme)
            if forced == "indoor":
                location = random.choice(INDOOR_LOCATIONS)
            elif forced == "tokyo":
                location = random.choice(TOKYO_LOCATIONS)
            elif forced == "japan":
                location = random.choice(JAPAN_LOCATIONS)
            else:
                pk = random.choices(["indoor", "tokyo", "japan"], weights=CATEGORY_WEIGHTS["real"], k=1)[0]
                location = random.choice(
                    {"indoor": INDOOR_LOCATIONS, "tokyo": TOKYO_LOCATIONS, "japan": JAPAN_LOCATIONS}[pk])
        elif category == "fantasy":
            if random.random() < 0.6:
                scenario = random.choice(FANTASY_IF_LINES)
                theme = "畅想未来·if线"
                detail_hint = f"，情节方向参考：{scenario}"
            else:
                theme = random.choice(FANTASY_ABSTRACT)
            pk = random.choices(["indoor", "tokyo", "japan"], weights=CATEGORY_WEIGHTS["fantasy"], k=1)[0]
            location = random.choice(
                {"indoor": INDOOR_LOCATIONS, "tokyo": TOKYO_LOCATIONS, "japan": JAPAN_LOCATIONS}[pk])
        else:
            cat_keys, cat_weights = list(META_CATEGORY_WEIGHTS.keys()), list(META_CATEGORY_WEIGHTS.values())
            meta_cat = random.choices(cat_keys, weights=cat_weights, k=1)[0]
            theme, hint_text = random.choice(META_CONTENT_POOLS[meta_cat])
            detail_hint = f"，具体情节参考（仅作方向，请自由延伸）：{hint_text}"
            pk = random.choices(["indoor", "tokyo", "japan"], weights=CATEGORY_WEIGHTS["meta"], k=1)[0]
            location = random.choice(
                {"indoor": INDOOR_LOCATIONS, "tokyo": TOKYO_LOCATIONS, "japan": JAPAN_LOCATIONS}[pk])

        num_people = random.randint(2, 5)
        selected = random.sample(members, num_people)
        random.shuffle(selected)
        people_str = "、".join(selected)

        if not is_plausible(category, theme, location, num_people): continue
        deep_part = "，" + random.choice(DEEP_RELATIONS) if random.random() < 0.3 else ""
        cross_part = f"，对话中偶然提到{random.choice(OTHER_MEMBERS)}" if random.random() < 0.2 else ""
        spot_hint = get_spot_info(location, mode="brief")
        return f"场景主题：{theme}。地点：{location}{spot_hint}。在场人物：{people_str}。{detail_hint}{deep_part}{cross_part}"
    return "场景主题：日常闲聊。地点：练习室。在场人物：爱音、素世、灯。"


def generate_mygo_chat(instrument_type, category_override=None):
    fallback_chat = [
        {"name": "爱音", "text": "诶？好像网络有点波动呢。"},
        {"name": "立希", "text": "真是的，关键时刻掉链子。"},
        {"name": "素世", "text": "没关系，我们就先用这套备用方案吧。"},
        {"name": "乐奈", "text": "抹茶...没有了吗？"},
        {"name": "灯", "text": "那个...只要大家在一起...就好。"}
    ]
    api_key = st.secrets.get("LLM_API_KEY")
    base_url = st.secrets.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model_name = st.secrets.get("LLM_MODEL", "deepseek-chat")
    if not api_key: return fallback_chat

    specific_scene = generate_fullband_scene(
        category_override=category_override) if instrument_type == "full_band" else random.choice(
        SUB_SCENES.get(instrument_type, ["休息室的日常对话"]))

    system_prompt = f"""
    你是一个MyGO!!!!!同人剧本生成器，精通每个角色的性格、说话方式和人际关系。
    请根据给定的场景，生成一段3~20句的群像剧对话（JSON格式）。

    【基本原则】（**请严格遵循！！！**）
    1. ⚠️ **话题多样性（核心要求）**：每组对话必须有独特的核心事件，严禁在不同轮次中出现重复或高度相似的主题！
    - **不要拘泥于我给你的场景字面描述！！！**
    2. 参与人数必须随机：每次对话可以是2人、3人、4人或5人，**绝对禁止每次都让所有角色出场!**
    3. 对话要有真实的情感张力（重力感），不能平淡如水。
    4. 绝对不要让所有的对话主题都是音乐或者合奏！一定要有生活场景的对话！
    5. 绝对不要让立希一直嘴臭！她是会好好说话的！
    6. 减少要乐奈的说话频率，只在有趣的时候开口。
    7. 除了要乐奈以外，剩下的成员所占的对话比例要尽可能的平均！
    8. 不要出现任何有关学习的话题。
    9. 对话中应适当使用日本当地的生活用语、称呼方式，符合日本校园生活场景。
    10. **背景共识**：MyGO的五名成员都已经知晓Ave Mujica的成立，也对CRYCHIC的历史（包括祥子离开的原因）有基本了解。因此，在涉及这些话题时，其他成员不会表现出惊讶或完全无知，而是以“已知”为前提进行对话。素世偶尔流露的沉重，其他人能敏锐察觉并可能默默关心，但不会大惊小怪。

{MYGO_PROFILES}

    【使用说明】
    当对话场景涉及排练、演出、选曲等音乐相关话题时，可以从已知曲目中随机选择参考。

    【对话生成要求】
    1. 输出一个JSON对象，包含两个字段："title"（10字到15字、有画面感的本轮剧情标题）和 "dialogue"（对话数组，每个元素含 "name" 和 "text"）。
    2. 在不叫外号的情况下，名字必须严格使用：「爱音」「素世」「灯」「立希」「乐奈」。
    3. 当提到Ave Mujica成员时，用「祥子」「睦」「初华」「海铃」「若麦」。
    4. 台词数3~12句，要体现角色间的互动和关系。
    5. 不要输出任何额外文字，只输出这个JSON对象。
    """
    user_prompt = f"当前场景参考：{specific_scene}。场景只是方向，对话内容可以自然延伸，不要拘泥于场景字面描述。请根据上述角色档案和要求自由生成对话。"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
        messages = [{"role": "system", "content": system_prompt}] + DEFAULT_EXAMPLES + [
            {"role": "user", "content": user_prompt}]
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=1.4, max_tokens=1500,
                                                  response_format={"type": "json_object"})
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"): content = content[:-3]

        import re
        obj_match = re.search(r'\{.*\}', content, re.DOTALL)
        if obj_match: content = obj_match.group()

        data = json.loads(content)
        title = data.get("title", "今日一幕")
        lines = data.get("dialogue") or data.get("lines")
        if not lines and isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    lines = val;
                    break
        elif isinstance(data, list):
            lines = data
        return {"title": title, "lines": lines} if lines else {"title": title, "lines": fallback_chat}
    except Exception as e:
        st.sidebar.error(f"DeepSeek API 调用失败: {e}")
        return {"title": "日常相处", "lines": fallback_chat}


def generate_travel_arc(landmark):
    fixed = {cat: random.choice(pool) for cat, pool in TRAVEL_FIXED_NODES.items()}
    spot_data = ALL_SPOTS.get(landmark, {})
    search_text = " ".join([spot_data.get("description", ""), spot_data.get("nature", ""), spot_data.get("vibe", ""),
                            spot_data.get("customs", ""), " ".join(spot_data.get("foods", [])),
                            " ".join(spot_data.get("spots", []))])
    feature_keywords = {
        "温泉": ["温泉", "足汤", "硫磺", "泡汤"],
        "古建筑": ["神社", "寺", "鸟居", "城", "古", "历史", "江户", "明治", "参道"],
        "海边": ["海", "海岸", "海湾", "海滩", "珊瑚", "渔"], "美食": ["美食", "料理", "小吃", "名产", "特产", "食"],
        "自然景观": ["山", "湖", "森林", "竹", "花", "瀑布", "火山", "草原", "公园", "自然"]
    }
    applicable_features = [feat for feat, keywords in feature_keywords.items() if
                           any(kw in search_text for kw in keywords)]
    spot_events = list(dict.fromkeys(
        random.choice(TRAVEL_SPOT_EVENTS[f]) for f in applicable_features if f in TRAVEL_SPOT_EVENTS)) or [
                      random.choice(TRAVEL_SPOT_EVENTS["美食"])]
    emotional = random.choice(TRAVEL_EMOTIONAL_BEATS) if random.random() < 0.4 else ""
    wildcard = random.choice(TRAVEL_WILDCARDS) if random.random() < 0.2 else ""

    lines = [
        "【旅途中可能发生的事（仅作方向参考，请根据角色人设自行分配给合适的人物，不必全部使用）】",
        f"· 出发阶段：{fixed['出发']}", f"· 途中：{fixed['交通']}", f"· 住宿：{fixed['住宿']}",
        f"· 景点互动：{'；'.join(spot_events)}", f"· 返程：{fixed['返程']}"
    ]
    if emotional: lines.append(f"· 情感暗线（可自然融入）：{emotional}")
    if wildcard: lines.append(f"· 意外变量（可自然融入）：{wildcard}")
    lines += ["", "【叙事要求】", "- 鼓励非线性叙事，避免流水账结构", "- 只有五名主角，背景路人可出现但无台词"]
    return "\n".join(lines)


def generate_episode_chat(episode_type, destination=None, search_context=None):
    fallback_lines = [{"name": "爱音", "text": "诶，好像信号不太好……"}, {"name": "立希", "text": "等一下。"},
                      {"name": "素世", "text": "没关系，稍等一下。"}, {"name": "灯", "text": "那个……"},
                      {"name": "乐奈", "text": "……"}]
    fallback = {"intro": "连接中，请稍候……",
                "episodes": [{"title": f"第{i + 1}幕", "lines": fallback_lines} for i in range(20)]}

    api_key = st.secrets.get("LLM_API_KEY")
    base_url = st.secrets.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model_name = st.secrets.get("LLM_MODEL", "deepseek-chat")
    if not api_key: return fallback

    search_inject = ""
    if search_context:
        search_inject = f"\n【互联网实时资讯/二创数据（必须参考）】\n{search_context}\n"

    if episode_type == "travel":
        if destination and destination in DESTINATION_MAP:
            landmark = random.choice(DESTINATION_MAP[destination])
        else:
            landmark = random.choice(list(ALL_SPOTS.keys()))
        spot_detail = get_spot_info(landmark, mode="full")
        travel_events = generate_travel_arc(landmark)
        arc_hint = f"目的地：【{landmark}】（对话中必须自然出现此地名）\n\n【目的地介绍】\n{spot_detail}\n\n{travel_events}"
        type_label = "出门旅游"

    elif episode_type == "fancreation":
        arc_hint = """剧情起点：五人无意间发现B站上大量关于她们的二创内容，开始追更、吐槽、被感动。
注意：本集只有爱音·素世·灯·立希·乐奈五人，绝对不出现其他角色。
⚠️ 严禁出现任何旅游、出行、外出度假、景点游览的内容——本集场景固定在室内（宿舍/练习室/某人家中）。"""
        if search_context:
            arc_hint += f"\n\n【⚠️ 必须参考的互联网真实搜索数据（将这些内容自然融入剧本对话中）】\n{search_context}\n【注意】提取有趣的部分进行互动，忽略负面内容。"
        type_label = "二创研讨"

    elif episode_type == "news":
        arc_hint = "剧情起点：五人看到了关于MyGO!!!!!或Ave Mujica的最新现实资讯。"
        if search_context:
            arc_hint += f"\n\n【⚠️ 必须参考的互联网最新资讯（将这些内容视为平行世界里她们真实参与的活动）】\n{search_context}\n【注意】请基于这些真实的情报、地点和事件进行对话。"
        type_label = "一手资讯讨论"

    else:
        triggers_text = get_random_triggers(2)
        arc_hint = f"""剧情起点：某个契机触发五人频繁回忆过去。

    {triggers_text}

    重要：背景成员（祥子·睦·初华·海铃·若麦）在本集要多以回忆/偶然出现/发消息等方式参与，让她们有真实存在感，而不只是被提及的名字。
    ⚠️ 严禁出现任何旅游、出行、外出度假、景点游览的内容——本集所有场景发生在日常固定地点（家中/学校/练习室），核心驱动力是"记忆与情感"而非"移动与旅行"。\n"""
        type_label = "回忆与现实交织"

    if episode_type in ["fancreation", "news"]:
        system_prompt = f"""你是专业的日本动画剧本作家，熟悉BanG Dream! MyGO!!!!!的角色设定。

    本次任务：生成【{type_label}】完整连续剧集。

    【剧情设定】
    {arc_hint}
    {search_inject}

    【角色档案】
    {MYGO_PROFILES}

    【二创/Meta创作指南】
    {FANCREATION_GUIDE}

    【输出格式——严格遵守】
    输出一个JSON对象，包含两个字段：
    1. "intro"：100-200字开场白，第三人称旁白风格，介绍本集故事背景和氛围，有文学感
    2. "episodes"：幕次数组，每个元素为：
        {{"title": "第X幕·标题（15-25字，有画面感，点出本幕情绪核心）", "dialogue": [{{"name": "角色名", "text": "台词"}}]}}

    每幕要求：
    - 台词8-15句，五名主角每幕都要有发言。
    - 幕与幕之间要有时间感和剧情推进，不要停在同一状态。
    - ⚠️【强制要求】每幕dialogue数组中**必须**插入1-3条 name为""的氛围描述行（旁白/场景描写），分散在对话之间。
    - 氛围描述行要有文学感，用第三人称旁白视角，10-25字，描写环境/氛围/角色状态。
    - 对话中严禁出现任何日文文字（包括假名、汉字词、罗马字）。
    - 不要输出JSON以外的任何内容。
        """
    elif episode_type == "memories":
        system_prompt = f"""你是专业的日本动画剧本作家，熟悉BanG Dream! MyGO!!!!!的角色设定。

    本次任务：生成【{type_label}】完整连续剧集。

    【剧情设定】
    {arc_hint}

    【角色档案】
    {MYGO_PROFILES}

    【回忆与现实交织 · 深度手册】
    {MEMORIES_GUIDE}

    【输出格式——严格遵守】
    输出一个JSON对象，包含两个字段：
    1. "intro"：100-200字开场白，第三人称旁白风格，介绍本集故事背景和氛围，有文学感
    2. "episodes"：幕次数组，每个元素为：
        {{"title": "第X幕·标题（15-25字，有画面感，点出本幕情绪核心）", "dialogue": [{{"name": "角色名", "text": "台词"}}]}}

    每幕要求：
    - 台词8-15句，五名主角每幕都要有发言。
    - 幕与幕之间要有时间感和剧情推进，不要停在同一状态。
    - ⚠️【强制要求】每幕dialogue数组中**必须**插入1-3条 name为""的氛围描述行（旁白/场景描写），分散在对话之间。
    - 对话中严禁出现任何日文文字。
    - 回忆场景和当下场景要交替出现，避免连续三幕都沉浸在过去。
    - 背景角色（祥子、睦等）以消息、回忆、偶遇等方式参与，但不抢占MyGO五人的情感空间。
        """
    else:
        system_prompt = f"""你是专业的日本动画剧本作家，熟悉BanG Dream! MyGO!!!!!的角色设定。

    本次任务：生成【{type_label}】完整连续剧集。

    【剧情设定】
    {arc_hint}

    【角色档案】
    {MYGO_PROFILES}

        【输出格式——严格遵守】
        输出一个JSON对象，包含两个字段：
        1. "intro"：100-200字开场白，第三人称旁白风格，介绍本集故事背景和氛围，有文学感
        2. "episodes"：幕次数组，每个元素为：
           {{"title": "第X幕·标题（15-25字，有画面感，点出本幕情绪核心）", "dialogue": [{{"name": "角色名", "text": "台词"}}]}}

        每幕要求：
        - 台词8-15句，五名主角每幕都要有发言。
        - 幕与幕之间要有时间感和剧情推进，不要停在同一状态。
        - ⚠️【强制要求】每幕dialogue数组中**必须**插入1-3条 name为""的氛围描述行（旁白/场景描写）。
        - 氛围描述行要有文学感，用第三人称旁白视角。
        - 对话中严禁出现任何日文文字。
        - 严禁逐条照搬上述剧情锚点的顺序，你的任务是围绕锚点自由创作，而非填空。
        """

    try:
        import json as _j
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

        if episode_type == "travel":
            prompt_p1 = "请生成开场白，以及第1幕到第10幕的对话（共10幕）。第1-10幕只负责覆盖旅程的前半段：出发、途中、抵达、住宿安顿、第一天的景点与用餐。此时五人仍在目的地，绝对不能出现返程或回到东京的内容。输出JSON对象，包含 'intro' 和 'episodes' 两个字段，episodes数组长度恰好为10。"
        else:
            prompt_p1 = "请生成开场白，以及第1幕到第10幕的对话（共10幕）。第1-10幕负责铺垫故事背景、引入核心情境，节奏可以相对轻松，为后半段蓄力。⚠️ 本集绝对不包含任何旅游、外出度假、景点游览的内容，请严格遵守。输出JSON对象，包含 'intro' 和 'episodes' 两个字段，episodes数组长度恰好为10。"

        resp1 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, *DEFAULT_EXAMPLES,
                      {"role": "user", "content": prompt_p1}],
            temperature=1.4, max_tokens=8000, response_format={"type": "json_object"}
        )
        c1 = resp1.choices[0].message.content.strip()
        if c1.startswith("```json"):
            c1 = c1[7:]
        elif c1.startswith("```"):
            c1 = c1[3:]
        if c1.endswith("```"): c1 = c1[:-3]
        d1 = _j.loads(c1)
        intro = d1.get("intro", "")
        episodes_raw = d1.get("episodes", [])

        last_lines_ctx = "，".join(f"{l.get('name', '')}：{l.get('text', '')}" for l in
                                  (episodes_raw[-1].get("dialogue") or episodes_raw[-1].get("lines") or [])[-3:] if
                                  l.get("text")) if episodes_raw else ""

        if episode_type == "travel":
            prompt_p2 = f"前10幕已经生成完毕，五人仍在目的地，最后的场景是：{last_lines_ctx or '（请自然延续）'}。请继续生成第11幕到第20幕（共10幕），覆盖旅程后半段：第二天活动、深层次的夜晚对话、最后的景点体验，以及第18幕之后才开始出现返程与告别，第20幕在回程途中或抵达后收尾。保持与前10幕的剧情连贯，输出JSON对象，只包含 'episodes' 字段，数组长度恰好为10，幕次标题从第11幕开始。"
        else:
            prompt_p2 = f"前10幕已经生成完毕，故事正在推进，最后的场景是：{last_lines_ctx or '（请自然延续）'}。请继续生成第11幕到第20幕（共10幕），情感强度逐渐推进，第18-20幕走向高潮或深度共鸣的收尾。⚠️ 本集绝对不包含任何旅游、外出度假、景点游览的内容，请严格遵守。保持与前10幕的剧情连贯，输出JSON对象，只包含 'episodes' 字段，数组长度恰好为10，幕次标题从第11幕开始。"

        resp2 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, *DEFAULT_EXAMPLES,
                      {"role": "user", "content": prompt_p1}, {"role": "assistant", "content": c1},
                      {"role": "user", "content": prompt_p2}],
            temperature=1.4, max_tokens=8000, response_format={"type": "json_object"}
        )
        c2 = resp2.choices[0].message.content.strip()
        if c2.startswith("```json"):
            c2 = c2[7:]
        elif c2.startswith("```"):
            c2 = c2[3:]
        if c2.endswith("```"): c2 = c2[:-3]
        d2 = _j.loads(c2)
        episodes_raw += d2.get("episodes", [])

        episodes = [{"title": ep.get("title", ""),
                     "lines": [{"name": l.get("name", ""), "text": l.get("text", "")} for l in
                               (ep.get("dialogue") or ep.get("lines") or []) if l.get("text", "")]} for ep in
                    episodes_raw]
        return {"intro": intro, "episodes": episodes} if episodes else fallback

    except Exception as e:
        st.sidebar.error(f"剧集生成失败: {e}")
        return fallback


@st.cache_data(show_spinner=False)
def load_image_b64(path):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_audio_b64(path):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def set_background(current_instrument: str):
    valid_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for directory in ["assets", ".", "./assets"]:
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        st.warning("⚠️ 背景图未生效：请在 assets 文件夹放入一张图片")
        return

    bg_path = image_files[0]
    bg_b64 = load_image_b64(bg_path)
    if not bg_b64: return

    mask_path = INSTRUMENT_MASKS.get(current_instrument, "")
    mask_b64 = load_image_b64(mask_path) if mask_path else None

    if mask_b64:
        highlight_layer = f"""
            [data-testid="stAppViewContainer"]::after {{
                content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                background-image: url(data:image/jpeg;base64,{bg_b64}); background-size: cover; background-position: center; background-repeat: no-repeat;
                opacity: 0.88; z-index: -1; pointer-events: none; transition: opacity 0.4s ease;
                filter: saturate(180%) contrast(80%) brightness(115%);
                -webkit-mask-image: url(data:image/png;base64,{mask_b64}); mask-image: url(data:image/png;base64,{mask_b64});
                -webkit-mask-mode: luminance; mask-mode: luminance;
                -webkit-mask-size: cover; mask-size: cover; -webkit-mask-position: center; mask-position: center; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
            }}
        """
    else:
        highlight_layer = """[data-testid="stAppViewContainer"]::after { content: none; }"""

    style = f"""
        <style>
        .stApp {{ background: transparent !important; }}
        [data-testid="stAppViewContainer"] {{ background: transparent !important; }}
        .main {{ background: transparent !important; }}
        [data-testid="stAppViewContainer"]::before {{
            content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background-image: url(data:image/jpeg;base64,{bg_b64}); background-size: cover; background-position: center; background-repeat: no-repeat;
            opacity: 0.5; filter: saturate(130%) contrast(80%); z-index: -2; pointer-events: none;
        }}
        {highlight_layer}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


DEFAULT_PARAMS = {
    "guitar": {"brightness": 0.60, "pluck_position": 0.25, "body_mix": 0.15, "reflection": 0.15, "coupling": 0.005},
    "bass": {"brightness": 0.65, "pluck_position": 1.8, "body_mix": 0.3, "reflection": 0.1, "coupling": 0.0},
    "piano": {"brightness": 0.65, "pluck_position": 1.0, "body_mix": 0.3, "reflection": 0.15, "coupling": 2.5},
    "drums": {"brightness": 0.7, "pluck_position": 1.2, "body_mix": 0.4, "reflection": 0.2, "coupling": 2.0},
    "full_band": {"brightness": 0.7, "pluck_position": 1.5, "body_mix": 0.35, "reflection": 0.18, "coupling": 52}
}


def get_local_midi_files():
    search_paths = ["assets/*.mid", "assets/*.midi", "../assets/*.mid", "../assets/*.midi", "./*.mid", "./*.midi"]
    files = []
    for pattern in search_paths: files.extend(glob.glob(pattern))
    return sorted(list(set(files)))


@st.cache_data(show_spinner=False)
def midi_to_audio_cached(file_bytes, instrument, brightness, pluck_pos, body_mix, reflection, coupling):
    try:
        if instrument == "guitar":
            from instruments import guitar as engine_module
            midi_stream = io.BytesIO(file_bytes)
            result = engine_module.midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling)
            if result is None or not isinstance(result, tuple) or result[0] is None: return None
            return result[0]
        elif instrument == "bass":
            from instruments import bass as engine_module
            midi_stream = io.BytesIO(file_bytes)
            try:
                result = engine_module.midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling,
                                                     solo_mode=True)
            except TypeError:
                midi_stream = io.BytesIO(file_bytes)
                result = engine_module.midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling)
            if result is None or not isinstance(result, tuple) or result[0] is None: return None
            return result[0]
        elif instrument == "drums":
            from instruments import drums as engine_module
            midi_stream = io.BytesIO(file_bytes)
            result = engine_module.midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling)
            if result is None or not isinstance(result, tuple) or result[0] is None: return None
            return result[0]
        elif instrument == "full_band":
            from instruments import guitar, bass, drums
            from scipy import signal

            original_data = file_bytes
            midi_stream_guitar = io.BytesIO(original_data)
            result_guitar = guitar.midi_to_audio(midi_stream_guitar, brightness * 1.05, 0.25, body_mix * 0.85,
                                                 reflection * 0.9, 0.005)

            midi_stream_bass = io.BytesIO(original_data)
            result_bass = bass.midi_to_audio(midi_stream_bass, brightness * 0.85, 1.8, body_mix * 1.15,
                                             reflection * 0.85, 0.0)

            midi_stream_drums = io.BytesIO(original_data)
            result_drums = drums.midi_to_audio(midi_stream_drums, brightness * 0.95, 1.2, body_mix * 0.7,
                                               reflection * 0.6, coupling)

            if not (result_guitar and result_bass and result_drums): return None
            guitar_samples, bass_samples, drums_samples = result_guitar[1], result_bass[1], result_drums[1]
            if guitar_samples is None or bass_samples is None or drums_samples is None: return None

            max_len = max(len(guitar_samples), len(bass_samples), len(drums_samples))
            if len(guitar_samples) < max_len: guitar_samples = np.pad(guitar_samples,
                                                                      (0, max_len - len(guitar_samples)))
            if len(bass_samples) < max_len: bass_samples = np.pad(bass_samples, (0, max_len - len(bass_samples)))
            if len(drums_samples) < max_len: drums_samples = np.pad(drums_samples, (0, max_len - len(drums_samples)))

            sos_g_hp = signal.butter(2, 100, 'hp', fs=48000, output='sos')
            guitar_samples = signal.sosfilt(sos_g_hp, guitar_samples)
            sos_b_lp = signal.butter(2, 3000, 'lp', fs=48000, output='sos')
            bass_samples = signal.sosfilt(sos_b_lp, bass_samples)
            b_bmud, a_bmud = signal.iirnotch(280, 8, 48000)
            bass_samples = signal.lfilter(b_bmud, a_bmud, bass_samples) * 0.85 + bass_samples * 0.15

            base_guitar, base_bass, base_drums = 0.65, 0.40, 0.15
            if pluck_pos < 1.5:
                guitar_vol, bass_vol, drums_vol = base_guitar * 1.1, base_bass * 0.9, base_drums * 0.95
            elif pluck_pos > 1.5:
                guitar_vol, bass_vol, drums_vol = base_guitar * 0.9, base_bass * 1.1, base_drums * 1.05
            else:
                guitar_vol, bass_vol, drums_vol = base_guitar, base_bass, base_drums

            mixed = (guitar_samples * guitar_vol + bass_samples * bass_vol + drums_samples * drums_vol)
            mixed = np.nan_to_num(mixed)
            peak = np.max(np.abs(mixed))
            if peak > 0.01: mixed = mixed / peak * 0.96

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
            from instruments import piano as engine_module
            midi_stream = io.BytesIO(file_bytes)
            result = engine_module.midi_to_audio(midi_stream, brightness, pluck_pos, body_mix, reflection, coupling)
            if result is None or not isinstance(result, tuple) or result[0] is None: return None
            return result[0]
    except Exception as e:
        st.error(f"渲染引擎错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


_PARAM_MAP = {
    'guitar': {'g_bright': 'brightness', 'g_pluck': 'pluck_position', 'g_res': 'body_mix', 'g_verb': 'reflection',
               'g_coup': 'coupling'},
    'bass': {'b_bright': 'brightness', 'b_force': 'pluck_position', 'b_body': 'body_mix', 'b_verb': 'reflection'},
    'drums': {'d_tension': 'brightness', 'd_impact': 'pluck_position', 'd_shell': 'body_mix', 'd_verb': 'reflection',
              'd_comp': 'coupling'},
    'piano': {'p_bright': 'brightness', 'p_hammer': 'pluck_position', 'p_board': 'body_mix', 'p_verb': 'reflection',
              'p_vel': 'coupling'},
    'full_band': {'f_bright': 'brightness', 'f_dyn': 'pluck_position', 'f_res': 'body_mix', 'f_verb': 'reflection',
                  'f_split': 'coupling'},
}


def _resolve_params(instrument: str, settings: dict) -> dict:
    defaults = DEFAULT_PARAMS.get(instrument, DEFAULT_PARAMS['guitar'])
    result = dict(defaults)
    if settings.get('use_custom_params') == '1':
        mapping = _PARAM_MAP.get(instrument, {})
        for html_key, backend_key in mapping.items():
            val_str = settings.get(f'p_{html_key}')
            if val_str is not None:
                try:
                    result[backend_key] = float(val_str)
                except ValueError:
                    pass
    return result


def _generate_theatre_data(settings: dict) -> list:
    import concurrent.futures
    fmt = settings.get('th_format', 'mini')
    depth = settings.get('th_depth', 'casual')
    scenario = settings.get('th_scenario', 'travel')
    destination = settings.get('th_destination', '')

    # 获取搜索内容
    search_context = st.session_state.get('final_search_text', '')

    if fmt == 'episode':
        _ep_map = {'travel': 'travel', 'daily': 'travel', 'fancreation': 'fancreation', 'news': 'news',
                   'memories': 'memories'}
        ep_type = _ep_map.get(scenario, 'travel')

        if ep_type == 'travel':
            ep_result = generate_episode_chat(ep_type, destination=destination)
        elif ep_type in ['fancreation', 'news']:
            ep_result = generate_episode_chat(ep_type, search_context=search_context)
        else:
            ep_result = generate_episode_chat(ep_type)

        intro_card = {"title": "__intro__", "lines": [], "intro": ep_result.get("intro", "")}
        return [intro_card] + ep_result.get("episodes", [])
    else:
        _num = {'casual': 5, 'deep': 10, 'lifetime': 20}.get(depth, 5)
        _cat_map = {'daily': 'real', 'travel': 'outing', 'future': 'fantasy', 'break_wall': 'meta', 'research': None,
                    'create_world': None}
        _cat = _cat_map.get(scenario, None)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(generate_mygo_chat, 'full_band', _cat) for _ in range(_num)]
                return [f.result() for f in concurrent.futures.as_completed(futures)]
        except Exception:
            return [{"title": "备用场景", "lines": [{"name": "爱音", "text": "诶？好像网络有点抖动呢，再等等？"}]}]


# ══════════════════════════════════════════════════════════
# 路由逻辑处理
# ══════════════════════════════════════════════════════════

if 'app_page' not in st.session_state:
    st.session_state.app_page = 'landing'

if st.query_params.get('go_landing') == '1':
    st.session_state.app_page = 'landing'
    for k in ['audio_out', 'multi_chat', 'upload_file_bytes', 'upload_file_name']:
        st.session_state.pop(k, None)
    st.query_params.clear()
    st.rerun()

if st.query_params.get('go_map') == '1':
    st.session_state.app_page = 'map'
    # 保存前端传过来的上下文参数，防止返回 landing 页面时状态丢失
    st.session_state.saved_context = {
        's_inst': st.query_params.get('s_inst', ''),
        's_voice': st.query_params.get('s_voice', '0'),
        's_theatre': st.query_params.get('s_theatre', '0'),
        's_rhythm': st.query_params.get('s_rhythm', '0'),
        's_midi': st.query_params.get('s_midi', ''),
        's_song': st.query_params.get('s_song', ''),
        's_fmt': st.query_params.get('s_fmt', 'mini'),
        's_scenario': st.query_params.get('s_scenario', 'random')
    }
    st.query_params.clear()
    st.rerun()

if st.query_params.get('go_search') == '1':
    # 记录模式
    st.session_state.search_mode = st.query_params.get('search_mode', 'manual')
    st.session_state.app_page = 'search_ui'
    # 保存前端传过来的上下文参数
    st.session_state.saved_context = {
        's_inst': st.query_params.get('s_inst', ''),
        's_voice': st.query_params.get('s_voice', '0'),
        's_theatre': st.query_params.get('s_theatre', '0'),
        's_rhythm': st.query_params.get('s_rhythm', '0'),
        's_midi': st.query_params.get('s_midi', ''),
        's_song': st.query_params.get('s_song', ''),
        's_fmt': st.query_params.get('s_fmt', 'mini'),
        's_scenario': st.query_params.get('s_scenario', 'random')
    }
    st.query_params.clear()
    st.rerun()

if st.query_params.get('go_upload') == '1':
    st.session_state.app_page = 'upload'
    # 统一暂存状态，防止返回时丢失
    st.session_state.saved_context = {
        's_inst': st.query_params.get('s_inst', ''),
        's_voice': st.query_params.get('s_voice', '0'),
        's_theatre': st.query_params.get('s_theatre', '0'),
        's_rhythm': st.query_params.get('s_rhythm', '0'),
        's_midi': 'upload',  # 锁定为上传模式
        's_song': '',
        's_fmt': st.query_params.get('s_fmt', 'mini'),
        's_scenario': st.query_params.get('s_scenario', 'random')
    }
    # 提取渲染必需的参数
    st.session_state.upload_settings = {
        'instrument': st.query_params.get('s_inst', 'guitar'),
        'voice': st.query_params.get('s_voice', '0'),
        'theatre': st.query_params.get('s_theatre', '0'),
        'rhythm': st.query_params.get('s_rhythm', '0'),
        'custom_params': st.query_params.get('use_custom_params', '0'),
    }
    for k, v in st.query_params.items():
        if k.startswith('p_') or k.startswith('th_'):
            st.session_state.upload_settings[k] = v
    st.query_params.clear()
    st.rerun()

_app_page = st.session_state.get('app_page', 'landing')

# ══════════════════════════════════════════════════════════
# 地图选择页 (Map Page)
# ══════════════════════════════════════════════════════════
if _app_page == 'map':
    bg_container = st.empty()
    with bg_container:
        set_background("")

    st.markdown("""
<style>
header[data-testid="stHeader"], footer, [data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { max-width: 1100px !important; padding-top: 6vh !important; padding-bottom: 0 !important; }
.map-title { font-size: 42px; font-weight: 800; color: #fff; text-align: center; text-shadow: 0 4px 15px rgba(0,0,0,0.6); margin-bottom: 8px; font-family: 'Segoe UI', sans-serif; letter-spacing: 1px; }
.map-subtitle { font-size: 14px; color: rgba(255,255,255,0.7); text-align: center; margin-bottom: 40px; letter-spacing: 2px; text-shadow: 0 2px 4px rgba(0,0,0,0.8); }
div[data-testid="column"] { background: rgba(30, 30, 35, 0.65); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 20px; padding: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }
div[data-testid="column"]:nth-child(2) { background: transparent; backdrop-filter: none; border: none; box-shadow: none; }
div.stButton > button { background: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(255, 255, 255, 0.3) !important; color: #fff !important; border-radius: 30px !important; font-weight: 600 !important; width: 100% !important; height: 48px !important; }
div.stButton > button:hover { background: #fff !important; color: #000 !important; border-color: #fff !important; transform: translateY(-2px) !important; box-shadow: 0 4px 15px rgba(255,255,255,0.2) !important; }
</style>""", unsafe_allow_html=True)

    st.markdown('<div class="map-title">Select Destination</div>', unsafe_allow_html=True)
    st.markdown('<div class="map-subtitle">地图选点系统 · 为「再看一集」配置旅行坐标</div>', unsafe_allow_html=True)

    col1, col_gap, col2 = st.columns([2.8, 0.15, 1.2])
    with col1:
        m = folium.Map(location=[36.5, 138], zoom_start=5, tiles="CartoDB positron")
        for region, coords in REGION_COORDINATES.items():
            spots = DESTINATION_MAP.get(region, [])
            tooltip_html = f"<div style='font-family:sans-serif; font-size:13px;'><b>{region}</b><br>包含 {len(spots)} 个事件点</div>"
            folium.Marker(location=coords, popup=region, tooltip=tooltip_html,
                          icon=folium.Icon(color="lightblue", icon="location-dot", prefix="fa")).add_to(m)

        output = st_folium(m, width=800, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

        if output and output["last_object_clicked"]:
            lat, lng = output["last_object_clicked"]["lat"], output["last_object_clicked"]["lng"]
            selected_region = None
            for region, coords in REGION_COORDINATES.items():
                if abs(coords[0] - lat) < 0.001 and abs(coords[1] - lng) < 0.001:
                    selected_region = region
                    break
            if selected_region:
                st.session_state.th_destination = selected_region
                st.session_state.app_page = 'landing'
                st.session_state.initial_step = 3
                st.rerun()

    with col2:
        st.markdown(
            "<div style='margin-bottom: 12px; font-weight: 700; color: #ddd; font-size: 14px; letter-spacing: 1px;'>📍 CURRENT DESTINATION</div>",
            unsafe_allow_html=True)
        curr = st.session_state.get('th_destination', '')
        if curr:
            st.markdown(
                f"<div style='background:rgba(67, 97, 238, 0.4); border:1px solid rgba(100, 149, 237, 0.6); padding:16px; border-radius:14px; color:#fff; text-align:center; font-size:20px; font-weight:800; margin-bottom: 24px;'>{curr}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:rgba(255, 255, 255, 0.05); border:1px solid rgba(255, 255, 255, 0.2); padding:16px; border-radius:14px; color:#aaa; text-align:center; font-size:16px; margin-bottom: 24px;'>未选择 (全国随机)</div>",
                unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size: 13px; color: #bbb; margin-bottom: 40px; line-height: 1.7; text-align: justify;'>👈 拖动左侧地图并点击蓝色标记，即可将该地区设为本次演出的目的地。<br><br>后续生成的长剧场将基于当地的真实风土人情与景点展开。</div>",
            unsafe_allow_html=True)

        if st.button("🎲 随机选择 (默认)"):
            st.session_state.th_destination = ""
            st.session_state.app_page = 'landing'
            st.session_state.initial_step = 3
            st.rerun()
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("⬅️ 返回剧场设置"):
            st.session_state.app_page = 'landing'
            st.session_state.initial_step = 3
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════
# 搜索 UI 界面 (Search UI)
# ══════════════════════════════════════════════════════════
if _app_page == 'search_ui':
    bg_container = st.empty()
    with bg_container:
        set_background("")

    # 注入高级 UI 样式
    st.markdown("""
<style>
/* 基础布局清理 */
header[data-testid="stHeader"], footer, [data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { 
    max-width: 900px !important; 
    padding-top: 8vh !important; 
    padding-bottom: 0 !important; 
}

/* 标题体系 */
.page-title {
    font-size: 42px; font-weight: 800; color: #fff; text-align: center;
    text-shadow: 0 4px 15px rgba(0,0,0,0.6); margin-bottom: 8px;
    font-family: 'Segoe UI', sans-serif; letter-spacing: 1px;
}
.page-subtitle {
    font-size: 14px; color: rgba(255,255,255,0.7); text-align: center;
    margin-bottom: 40px; letter-spacing: 2px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
}

/* 玻璃面板 */
.glass-panel {
    background: rgba(30, 30, 35, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

/* 输入框与按钮对齐黑科技 */
[data-testid="stHorizontalBlock"] { align-items: center !important; }

/* 输入框美化 */
[data-testid="stTextInput"] label { display: none; } /* 隐藏 Label */
[data-testid="stTextInput"] input {
    height: 50px !important;
    border-radius: 25px !important;
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    padding: 0 24px !important;
    font-size: 15px !important;
    transition: all 0.3s;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(147, 197, 253, 0.8) !important;
    box-shadow: 0 0 15px rgba(147, 197, 253, 0.2) !important;
    background: rgba(0, 0, 0, 0.5) !important;
}

/* 搜索按钮 */
div.stButton > button {
    background: linear-gradient(135deg, rgba(67, 97, 238, 0.4), rgba(67, 97, 238, 0.2)) !important;
    border: 1px solid rgba(100, 149, 237, 0.5) !important;
    color: #fff !important;
    border-radius: 25px !important;
    font-weight: 700 !important;
    height: 50px !important;
    transition: 0.3s;
    width: 100%;
    margin-top: 0px !important; /* 强制对齐 */
}
div.stButton > button:hover {
    background: linear-gradient(135deg, rgba(67, 97, 238, 0.6), rgba(67, 97, 238, 0.4)) !important;
    transform: scale(1.02);
    box-shadow: 0 4px 20px rgba(67, 97, 238, 0.3) !important;
}

/* 搜索结果卡片 */
.result-card-container {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.2s;
}
.result-card-container:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(147, 197, 253, 0.3);
    transform: translateX(4px);
}
.res-title { font-size: 16px; font-weight: 700; color: #93c5fd; margin-bottom: 6px; }
.res-body { font-size: 13px; color: #ddd; line-height: 1.5; margin-bottom: 6px; }
.res-meta { font-size: 11px; color: #777; font-family: monospace; }

/* 底部返回按钮容器 */
.bottom-actions {
    margin-top: 30px;
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

    # 获取模式
    mode = st.session_state.get('search_mode', 'manual')

    # 标题区
    title_str = "Topic Search" if mode == 'manual' else "Breaking News Scan"
    subtitle_str = "输入关键词进行二创话题研讨" if mode == 'manual' else "正在连接互联网检索 MyGO!!!!! / Ave Mujica 最新资讯"

    st.markdown(f'<div class="page-title">{title_str}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle_str}</div>', unsafe_allow_html=True)

    # 主功能区
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

        # 搜索栏布局
        col1, col2 = st.columns([3.5, 1.2])
        with col1:
            if mode == 'manual':
                query = st.text_input("Search Keyword", placeholder="例如：羊宫妃那 迷星叫 / 祥子 梗 / 邦邦新企划",
                                      label_visibility="collapsed")
            else:
                st.info("锁定搜索范围：MyGO!!!!! / Ave Mujica / BanG Dream! (国内源)")
                query = "BanG Dream MyGO Ave Mujica 最新资讯 live"

        with col2:
            search_btn = st.button("🚀 开始检索")

        # 触发搜索
        if search_btn:
            if not query.strip():
                st.error("请输入检索词！")
            else:
                with st.spinner("正在连接 Tavily 国内节点扫描全网..."):
                    results = search_web(query, max_results=12)
                    st.session_state.raw_search_results = results

        # 结果展示与表单选择
        if 'raw_search_results' in st.session_state and st.session_state.raw_search_results:
            results = st.session_state.raw_search_results
            st.write("---")
            st.markdown(
                f"<div style='text-align:left; color:#aaa; font-size:13px; margin-bottom:15px;'>共找到 {len(results)} 条相关结果，请勾选您希望 AI 参考的内容：</div>",
                unsafe_allow_html=True)

            with st.form("selection_form"):
                selections = []
                for i, item in enumerate(results):
                    # 使用 HTML 渲染卡片
                    st.markdown(f"""
                    <div class="result-card-container">
                        <div class="res-title">{item.get('title', 'No Title')}</div>
                        <div class="res-body">{item.get('content', '')}</div>
                        <div class="res-meta">SOURCE: {item.get('url', 'Unknown')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Checkbox 紧跟在卡片下方
                    checked = st.checkbox(f"✅ 采纳此条情报", key=f"chk_{i}")
                    selections.append((item, checked))
                    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # Spacer

                st.markdown("---")
                if st.form_submit_button("💾 确认并使用选中内容生成剧本"):
                    selected_items = [item for item, chk in selections if chk]
                    if not selected_items:
                        st.error("请至少勾选一条内容！")
                    else:
                        final_text = "\n\n".join(
                            [f"【标题】{it.get('title')}\n【内容】{it.get('content')}" for it in selected_items])

                        # 保存状态并跳转
                        st.session_state.final_search_text = final_text
                        st.session_state.search_ready = True

                        st.session_state.app_page = 'landing'
                        st.session_state.initial_step = 3

                        # 清理原始结果以释放内存
                        st.session_state.pop('raw_search_results', None)
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)  # End glass panel

    # 底部返回按钮
    st.write("")
    st.write("")
    col_a, col_b, col_c = st.columns([1.5, 2, 1.5])
    with col_b:
        if st.button("⬅️ 取消并返回舞台设置", key="back_btn"):
            # 如果取消，清除搜索状态，视为未完成搜索
            st.session_state.search_ready = False
            st.session_state.pop('final_search_text', None)
            st.session_state.pop('raw_search_results', None)

            st.session_state.app_page = 'landing'
            st.session_state.initial_step = 3
            st.rerun()

    st.stop()

# ══════════════════════════════════════════════════════════
# 极简上传页
# ══════════════════════════════════════════════════════════
if _app_page == 'upload':
    bg_container = st.empty()
    with bg_container:
        set_background("")

    # 注入高级 UI 样式
    st.markdown("""
<style>
/* 基础布局清理 */
header[data-testid="stHeader"], footer, [data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { 
    max-width: 900px !important; 
    padding-top: 10vh !important; 
    padding-bottom: 0 !important; 
}

/* 标题体系 */
.page-title {
    font-size: 42px; font-weight: 800; color: #fff; text-align: center;
    text-shadow: 0 4px 15px rgba(0,0,0,0.6); margin-bottom: 8px;
    font-family: 'Segoe UI', sans-serif; letter-spacing: 1px;
}
.page-subtitle {
    font-size: 14px; color: rgba(255,255,255,0.7); text-align: center;
    margin-bottom: 50px; letter-spacing: 2px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
}

/* --- Streamlit 文件上传组件深度美化 --- */
[data-testid="stFileUploader"] {
    padding-top: 0px;
}

/* 上传区域核心样式 */
[data-testid="stFileUploader"] section {
    background-color: rgba(30, 30, 35, 0.75) !important; /* 深色半透明背景 */
    backdrop-filter: blur(16px); /* 磨砂玻璃效果 */
    -webkit-backdrop-filter: blur(16px);
    border: 1px dashed rgba(255, 255, 255, 0.3) !important;
    border-radius: 20px !important;
    padding: 60px 20px !important; /* 增加内部空间 */
    transition: all 0.3s ease;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

/* 悬停效果 */
[data-testid="stFileUploader"] section:hover {
    background-color: rgba(40, 40, 45, 0.85) !important;
    border-color: #93c5fd !important;
    transform: scale(1.01);
}

/* 上传小图标 */
[data-testid="stFileUploader"] svg {
    fill: #93c5fd !important;
    width: 48px !important;
    height: 48px !important;
    margin-bottom: 10px;
}

/* 文字颜色 */
[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
    color: #eee !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] small {
    color: #aaa !important;
}

/* 内部 "Browse files" 按钮 */
[data-testid="stFileUploader"] button {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: #fff !important;
    border-radius: 20px !important;
    padding: 8px 24px !important;
    margin-top: 10px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #fff !important;
    color: #000 !important;
    border-color: #fff !important;
}

/* 底部返回按钮样式 */
div.stButton > button {
    background: linear-gradient(135deg, rgba(67, 97, 238, 0.4), rgba(67, 97, 238, 0.2)) !important;
    border: 1px solid rgba(100, 149, 237, 0.5) !important;
    color: #fff !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    height: 50px !important;
    transition: 0.3s;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, rgba(67, 97, 238, 0.6), rgba(67, 97, 238, 0.4)) !important;
    transform: scale(1.02);
    box-shadow: 0 4px 20px rgba(67, 97, 238, 0.3) !important;
}
</style>""", unsafe_allow_html=True)

    # 标题区
    st.markdown('<div class="page-title">Upload Custom MIDI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">上传本地 MIDI 文件以驱动物理建模引擎 · 支持多轨解析</div>',
                unsafe_allow_html=True)

    # 主功能区
    with st.container():
        f = st.file_uploader("拖拽文件至此区域", type=["mid", "midi"], label_visibility="visible")

    # 渲染逻辑
    if f:
        _upload_bytes = f.read()
        _settings = st.session_state.get('upload_settings', {})
        _inst = _settings.get('instrument', 'guitar')
        _params = _resolve_params(_inst, _settings)

        st.write("")  # Spacer
        with st.status("🚀 正在启动音频渲染引擎...", expanded=True) as _status:
            if _settings.get('rhythm') == '1':
                st.write("正在解析 MIDI 事件流生成 4K 下落式音游谱面...")
            else:
                st.write("解析 MIDI 事件流...")
            time.sleep(0.3)
            st.write(f"加载 {_inst.upper()} 物理模型参数...")

            audio_bytes = midi_to_audio_cached(
                _upload_bytes, _inst,
                _params['brightness'], _params['pluck_position'],
                _params['body_mix'], _params['reflection'], _params['coupling']
            )

            if audio_bytes:
                st.session_state.audio_out = audio_bytes
                st.session_state.render_instrument = _inst

                if _settings.get('theatre') == '1':
                    st.write("正在生成剧场对话数据...")
                    _theatre_data = _generate_theatre_data(_settings)
                    st.session_state.multi_chat = _theatre_data
                else:
                    st.session_state.multi_chat = []

                _status.update(label="✅ 渲染成功！正在跳转至播放界面...", state="complete")
                st.session_state.app_page = 'landing'
                st.session_state._from_upload = True
                time.sleep(0.8)
                st.rerun()
            else:
                _status.update(label="❌ 渲染失败，请检查 MIDI 文件格式", state="error")

    # 返回按钮区
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        if st.button("⬅️ 取消并返回"):
            st.session_state.app_page = 'landing'
            st.session_state.initial_step = 2
            st.rerun()

    st.stop()

# ══════════════════════════════════════════════════════════
# 落地页 (Landing Page - 唯一主页面)
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
header[data-testid="stHeader"], [data-testid="stSidebar"], footer,
[data-testid="stDecoration"] { display: none !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: transparent !important; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
</style>""", unsafe_allow_html=True)

_sel_inst = st.query_params.get('sel_instrument') or st.session_state.get('render_instrument', '')
_sel_midi = st.query_params.get('sel_midi')
_from_upload = st.session_state.pop('_from_upload', False)
_rendered_audio_b64 = ''
_chat_data_json = '[]'
_is_theatre = False

_initial_step = st.session_state.pop('initial_step', 1)
_current_dest = st.session_state.get('th_destination', '')
_search_ready = st.session_state.get('search_ready', False)

if _from_upload and 'audio_out' in st.session_state:
    _rendered_audio_b64 = base64.b64encode(st.session_state.audio_out).decode()
    _chat_data_json = json.dumps(st.session_state.get('multi_chat', []), ensure_ascii=False)
    _is_theatre = bool(st.session_state.get('multi_chat'))

elif _sel_inst and _sel_midi:
    _inst = _sel_inst
    bg_container = st.empty()
    with bg_container:
        set_background("")

    _settings = dict(st.query_params)
    _params = _resolve_params(_inst, _settings)

    _midi_bytes = None
    if _sel_midi == 'haruhikage':
        for p in ["assets/春日影-mygo.mid", "../assets/春日影-mygo.mid", "春日影-mygo.mid"]:
            if os.path.exists(p):
                with open(p, "rb") as f: _midi_bytes = f.read(); break
    elif _sel_midi == 'preset':
        _song_name = st.query_params.get('sel_song', '')
        if _song_name:
            for _fp in get_local_midi_files():
                if os.path.basename(_fp) == _song_name:
                    with open(_fp, "rb") as f: _midi_bytes = f.read(); break

    if _midi_bytes:
        loading_container = st.empty()
        tuning_texts = {"guitar": "爱音正在确认效果器踏板，乐奈在一旁随意地拨弄琴弦...",
                        "bass": "素世正垂下眼眸，细致地调整着贝斯音色与线路...",
                        "drums": "立希紧皱眉头，正在检查小军鼓的张力与踩槌角度...",
                        "piano": "正在进行键盘触键响应测试与物理反馈确认..."}
        tuning_text = tuning_texts.get(_inst, "灯紧紧握着麦克风，全员正在进行最后的调音确认...")


        def get_loading_html(state="loading", inst="full_band", subtext="", animate_base=False, progress=0):
            C_PINK, C_GREEN, C_YELLOW, C_PURPLE, C_BLUE = "#f9a8d4", "#86efac", "#fde68a", "#c4b5fd", "#93c5fd"
            if inst == "guitar":
                marks, comp_style = [C_PINK, C_PINK, C_PINK, C_GREEN, C_GREEN], ""
            elif inst == "bass":
                marks, comp_style = [C_YELLOW] * 5, f"color: {C_YELLOW};"
            elif inst == "drums":
                marks, comp_style = [C_PURPLE] * 5, f"color: {C_PURPLE};"
            else:
                marks, comp_style = [C_PINK, C_GREEN, C_YELLOW, C_PURPLE, C_BLUE], f"color: {C_BLUE};"

            chars_html = "".join([
                f'<span style="display:inline-block; width: 18px;"></span>' if char == " " else f'<span style="display:inline-block; {"opacity:0; animation: charFade 0.4s forwards " + str(0.05 * i) + "s;" if animate_base else "opacity:1;"}">{char}</span>'
                for i, char in enumerate("It's MyGO")])
            for i, m_color in enumerate(
                    marks): chars_html += f'<span style="display:inline-block; color:{m_color}; {"opacity:0; animation: popIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;" if i == progress - 1 else "opacity:1;" if i < progress else "opacity:0;"}">!</span>'

            if state == "completed":
                if inst == "guitar":
                    comp_text = f'<span style="color: {C_PINK};">Rendering</span> <span style="color: {C_GREEN};">Completed!</span>'
                else:
                    comp_text = f'<span style="{comp_style}">Rendering Completed!</span>'

                main_content = f'<div style="font-size: 52px; font-weight: 800; animation: compFadeIn 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards; text-shadow: 0 4px 15px rgba(0,0,0,0.6); letter-spacing: 2px;">{comp_text}</div>'
                sub_content = ""
            else:
                main_content = f'<div style="font-size: 60px; font-weight: 800; color: white; text-shadow: 0 4px 15px rgba(0,0,0,0.6); letter-spacing: 4px;">{chars_html}</div>'
                sub_content = f'<div style="margin-top: 32px; font-size: 15px; color: rgba(255,255,255,0.7); letter-spacing: 2.5px; opacity:0; animation: subFade 0.6s forwards; text-shadow: 0 2px 4px rgba(0,0,0,0.8);">{subtext}</div>'

            return f"""<div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 9999; background: transparent; pointer-events: none; font-family: 'Segoe UI', sans-serif;">{main_content}{sub_content}<style>@keyframes charFade {{0% {{opacity: 0; transform: translateY(12px); filter: blur(4px);}} 100% {{opacity: 1; transform: translateY(0); filter: blur(0);}}}} @keyframes subFade {{0% {{opacity: 0; transform: translateY(8px);}} 100% {{opacity: 1; transform: translateY(0);}}}} @keyframes popIn {{0% {{opacity: 0; transform: scale(0.5) translateY(10px);}} 60% {{opacity: 1; transform: scale(1.2) translateY(0);}} 100% {{opacity: 1; transform: scale(1) translateY(0);}}}} @keyframes compFadeIn {{0% {{opacity: 0; transform: scale(0.9) translateY(10px); filter: blur(8px);}} 100% {{opacity: 1; transform: scale(1) translateY(0); filter: blur(0);}}}}</style></div>"""


        loading_container.markdown(get_loading_html("loading", _inst, tuning_text, animate_base=True, progress=0),
                                   unsafe_allow_html=True);
        time.sleep(0.5)
        loading_container.markdown(get_loading_html("loading", _inst, "解析乐曲参数与环境配置...", progress=1),
                                   unsafe_allow_html=True);
        time.sleep(0.3)
        loading_container.markdown(
            get_loading_html("loading", _inst, "正在初始化 Karplus-Strong 引擎渲染高保真音频...", progress=2),
            unsafe_allow_html=True)

        audio_bytes = midi_to_audio_cached(_midi_bytes, _inst, _params['brightness'], _params['pluck_position'],
                                           _params['body_mix'], _params['reflection'], _params['coupling'])

        if audio_bytes:
            st.session_state.audio_out = audio_bytes
            _rendered_audio_b64 = base64.b64encode(audio_bytes).decode()
            loading_container.markdown(get_loading_html("loading", _inst, "音频合成完毕！引擎冷却中...", progress=3),
                                       unsafe_allow_html=True);
            time.sleep(0.3)

            if st.query_params.get('sel_theatre') == '1':
                theatre_text = random.choice(
                    ["灯正在翻阅随身携带的笔记本，整理着零散的思绪...", "大家正在休息室里漫无目的地闲聊，气氛逐渐活跃...",
                     "乐奈正盯着桌上的抹茶点心发呆...", "正在捕捉现实与回忆交织的情感碎片...",
                     "爱音正在刷着手机，偶尔抬起头参与对话..."])
                loading_container.markdown(get_loading_html("loading", _inst, theatre_text, progress=4),
                                           unsafe_allow_html=True)
                _theatre_data = _generate_theatre_data(dict(st.query_params))
                st.session_state.multi_chat = _theatre_data
                _chat_data_json = json.dumps(_theatre_data, ensure_ascii=False)
                _is_theatre = True
                loading_container.markdown(get_loading_html("loading", _inst, "剧本与演出数据同步完成！", progress=5),
                                           unsafe_allow_html=True)
            elif st.query_params.get('sel_rhythm') == '1':
                loading_container.markdown(
                    get_loading_html("loading", _inst, "正在解析多轨 MIDI 生成 4K 下落式谱面...", progress=4),
                    unsafe_allow_html=True)
                time.sleep(0.4)
                loading_container.markdown(
                    get_loading_html("loading", _inst, "音游系统初始化完成！全员准备就绪！", progress=5),
                    unsafe_allow_html=True)
                _chat_data_json = '[]'
            else:
                _chat_data_json = '[]'
                loading_container.markdown(get_loading_html("loading", _inst, "整理舞台设备...", progress=4),
                                           unsafe_allow_html=True);
                time.sleep(0.2)
                loading_container.markdown(get_loading_html("loading", _inst, "全员准备就绪！", progress=5),
                                           unsafe_allow_html=True)

            time.sleep(0.5)
            loading_container.markdown(get_loading_html("completed", _inst), unsafe_allow_html=True)
            time.sleep(1.2)
        loading_container.empty();
        bg_container.empty()

    if st.query_params.get('sel_voice') == '1':
        st.session_state.pending_voice = BASS_SPECIAL_VOICE if _inst == 'bass' and _sel_midi == 'haruhikage' else pick_voice(
            _inst)

_valid_ext = ["*.jpg", "*.jpeg", "*.png"]
_img_files = []
for _d in ["assets", ".", "./assets"]:
    for _e in _valid_ext: _img_files.extend(glob.glob(os.path.join(_d, _e)))
_img_files = sorted(list(set(_img_files)))
_bg_b64 = load_image_b64(_img_files[0]) if _img_files else None

_mask_b64_map = {k: load_image_b64(p) for k, p in INSTRUMENT_MASKS.items() if p and load_image_b64(p)}

_intro_voice_b64 = ""
for _vp in ["assets/voices/咕咕嘎嘎.mp3", "assets/voices/灵感菇.mp3"]:
    if os.path.exists(_vp):
        with open(_vp, "rb") as _vf: _intro_voice_b64 = base64.b64encode(_vf.read()).decode(); break

_song_list = [os.path.basename(p) for p in get_local_midi_files()]

_char_positions = {
    "素世": {"top": "28%", "left": "12%", "anchor": "right"}, "爱音": {"top": "55%", "left": "12%", "anchor": "right"},
    "灯": {"top": "75%", "left": "33%", "anchor": "right"}, "立希": {"top": "48%", "left": "85%", "anchor": "left"},
    "乐奈": {"top": "18%", "left": "63%", "anchor": "left"},
}
_char_cfg = {name: {"color": cfg["color"], "emoji": cfg["emoji"]} for name, cfg in MYGO_CHARACTERS.items()}

_covers_map = {}
for _midi_path in get_local_midi_files():
    _song_filename = os.path.basename(_midi_path)
    _base_name = os.path.splitext(_song_filename)[0]
    for _ext in [".jpg", ".png", ".jpeg"]:
        _cover_path = os.path.join("assets", "covers", _base_name + _ext)
        if not os.path.exists(_cover_path): _cover_path = os.path.join("../assets", "covers", _base_name + _ext)
        if os.path.exists(_cover_path):
            _b64 = load_image_b64(_cover_path)
            if _b64: _covers_map[_song_filename] = _b64; break

_mask_js_entries = ",\n  ".join('"{k}": "data:image/png;base64,{v}"'.format(k=k, v=v) for k, v in _mask_b64_map.items())
_bg_css = f"background-image:url('data:image/jpeg;base64,{_bg_b64}');background-size:cover;background-position:center;" if _bg_b64 else "background:#0a0614;"
_audio_tag = f"<audio id='introAudio' style='display:none'><source src='data:audio/mpeg;base64,{_intro_voice_b64}' type='audio/mpeg'></audio>" if _intro_voice_b64 else ""

import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _landing_page import build_landing_html

# 获取当前保存的状态，用于恢复前端
_saved_ctx = st.session_state.pop('saved_context', {})
_upload_settings = st.session_state.get('upload_settings', {})


def _get_bool_setting(ctx_key, upload_key, query_key):
    if ctx_key in _saved_ctx:
        return _saved_ctx[ctx_key] == '1'
    if upload_key in _upload_settings:
        return _upload_settings[upload_key] == '1'
    return st.query_params.get(query_key) == '1'


def _get_str_setting(ctx_key, upload_key, query_key, default):
    if ctx_key in _saved_ctx:
        return _saved_ctx[ctx_key]
    if upload_key in _upload_settings:
        return _upload_settings[upload_key]
    if st.query_params.get(query_key):
        return st.query_params.get(query_key)
    return default


_saved_inst = _saved_ctx.get('s_inst') or _upload_settings.get('instrument', '') or st.query_params.get('s_inst', '')
if not _saved_inst: _saved_inst = _sel_inst

_saved_voice = _get_bool_setting('s_voice', 'voice', 's_voice')
_saved_theatre = _get_bool_setting('s_theatre', 'theatre', 's_theatre')
_saved_rhythm = _get_bool_setting('s_rhythm', 'rhythm', 's_rhythm') or st.query_params.get('sel_rhythm') == '1'
_saved_diff = st.query_params.get('sel_difficulty', 'normal')
_saved_midi = _get_str_setting('s_midi', 'sel_midi', 's_midi', '')
_saved_song = _get_str_setting('s_song', 'sel_song', 's_song', '')
_saved_fmt = _get_str_setting('s_fmt', 'th_format', 's_fmt', 'mini')
_saved_scenario = _get_str_setting('s_scenario', 'th_scenario', 's_scenario', 'random')

# ================= 生成音游谱面 =================
_beatmap_json = '[]'
_is_rhythm_mode = False

if _saved_rhythm and _sel_midi and _midi_bytes:
    _beatmap_data = generate_beatmap(_midi_bytes, difficulty=_saved_diff)
    if _beatmap_data:
        _beatmap_json = json.dumps(_beatmap_data, ensure_ascii=False)
        _is_rhythm_mode = True
        _is_theatre = False

_landing_html = build_landing_html(
    bg_css=_bg_css,
    mask_js_entries=_mask_js_entries,
    audio_tag=_audio_tag,
    rendered_audio_b64=_rendered_audio_b64,
    song_list_json=json.dumps(_song_list, ensure_ascii=False),
    song_covers_json=json.dumps(_covers_map, ensure_ascii=False),
    theatre_data_json=_chat_data_json,
    beatmap_json=_beatmap_json,
    is_rhythm=_is_rhythm_mode,
    char_positions_json=json.dumps(_char_positions, ensure_ascii=False),
    char_cfg_json=json.dumps(_char_cfg, ensure_ascii=False),
    is_theatre=_is_theatre,
    selected_instrument=_sel_inst or '',
    initial_step=_initial_step,
    current_dest=_current_dest,
    search_ready=_search_ready,
    saved_inst=_saved_inst,
    saved_voice=_saved_voice,
    saved_theatre=_saved_theatre,
    saved_rhythm=_saved_rhythm,
    saved_midi=_saved_midi,
    saved_song=_saved_song,
    saved_fmt=_saved_fmt,
    saved_scenario=_saved_scenario
)

_voice = st.session_state.pop('pending_voice', None)
inject_voice(_voice)

components.html(_landing_html, height=1000, scrolling=False)

st.stop()

