"""
_landing_page.py  v10.1
核心改动：
- 4K 下落式音游 (Rhythm Game) 引擎深度 Juice 调优
- 加入屏幕震动 (Screen Shake)、粒子火花 (Particles)、轨道高亮光柱
- 巨型居中 Combo 显示与动感爆出动画 (Pop Animation)
- 左右两侧轨道分离颜色：左边粉色(爱音)，右边蓝色(灯)
- 提高速度上限，优化打击判定手感
"""

import json

_TITLE_COLORS = {
    "guitar": "#ff8899", "bass": "#ffdd88", "drums": "#7777aa",
    "full_band": "#3388bb", "default": "#ffffff"
}

_LABEL_TEXTS = {
    "guitar": "千早爱音&要乐奈 · 吉他", "bass": "长崎素世 · 贝斯",
    "drums": "椎名立希 · 架子鼓", "full_band": "高松灯 · 乐队全员",
    "default": "请选择角色进入"
}

_PARAMS_CONFIG = {
    "guitar": [
        {"key": "g_bright", "label": "亮度", "def": 0.60, "min": 0.2, "max": 0.8, "step": 0.02},
        {"key": "g_pluck", "label": "拨弦位置", "def": 0.25, "min": 0.08, "max": 0.40, "step": 0.01},
        {"key": "g_res", "label": "琴箱共鸣", "def": 0.15, "min": 0.0, "max": 0.6, "step": 0.02},
        {"key": "g_verb", "label": "空间反射", "def": 0.15, "min": 0.0, "max": 0.3, "step": 0.01},
    ],
    "bass": [
        {"key": "b_bright", "label": "明亮度", "def": 0.65, "min": 0.2, "max": 0.7, "step": 0.05},
        {"key": "b_force", "label": "拨弦力度", "def": 1.8, "min": 1.2, "max": 2.5, "step": 0.1},
        {"key": "b_body", "label": "箱体共鸣", "def": 0.3, "min": 0.0, "max": 0.6, "step": 0.05},
        {"key": "b_verb", "label": "房间混响", "def": 0.1, "min": 0.0, "max": 0.3, "step": 0.02},
    ],
    "drums": [
        {"key": "d_tension", "label": "鼓皮硬度", "def": 0.7, "min": 0.3, "max": 0.9, "step": 0.05},
        {"key": "d_impact", "label": "打击响应", "def": 1.2, "min": 0.5, "max": 2.0, "step": 0.1},
        {"key": "d_shell", "label": "腔体共鸣", "def": 0.4, "min": 0.0, "max": 0.8, "step": 0.05},
        {"key": "d_comp", "label": "压缩感", "def": 2.0, "min": 1.0, "max": 3.0, "step": 0.1},
    ],
    "piano": [
        {"key": "p_bright", "label": "明亮度", "def": 0.65, "min": 0.3, "max": 0.9, "step": 0.05},
        {"key": "p_hammer", "label": "琴槌硬度", "def": 1.0, "min": 0.5, "max": 2.0, "step": 0.1},
        {"key": "p_board", "label": "音板共鸣", "def": 0.3, "min": 0.0, "max": 0.5, "step": 0.05},
    ],
    "full_band": [
        {"key": "f_bright", "label": "整体明亮", "def": 0.7, "min": 0.4, "max": 0.9, "step": 0.05},
        {"key": "f_dyn", "label": "动态平衡", "def": 1.5, "min": 0.8, "max": 2.5, "step": 0.1},
        {"key": "f_res", "label": "乐器共鸣", "def": 0.35, "min": 0.0, "max": 0.6, "step": 0.05},
        {"key": "f_verb", "label": "混响", "def": 0.18, "min": 0.0, "max": 0.4, "step": 0.02},
    ],
}


def build_landing_html(bg_css: str, mask_js_entries: str, audio_tag: str = '',
                       rendered_audio_b64: str = '',
                       song_list_json: str = '[]',
                       song_covers_json: str = '{}',
                       theatre_data_json: str = '[]',
                       beatmap_json: str = '[]',
                       is_rhythm: bool = False,
                       char_positions_json: str = '{}',
                       char_cfg_json: str = '{}',
                       is_theatre: bool = False,
                       selected_instrument: str = '',
                       initial_step: int = 1,
                       current_dest: str = '',
                       search_ready: bool = False,
                       saved_inst: str = '',
                       saved_voice: bool = False,
                       saved_theatre: bool = False,
                       saved_rhythm: bool = False,
                       saved_midi: str = '',
                       saved_song: str = '',
                       saved_fmt: str = 'mini',
                       saved_scenario: str = 'random') -> str:
    safe_audio_tag = audio_tag if (audio_tag and audio_tag.strip()) else ""
    has_audio = bool(rendered_audio_b64)
    has_audio_str = "true" if has_audio else "false"
    is_theatre_str = "true" if is_theatre else "false"
    is_rhythm_str = "true" if is_rhythm else "false"
    search_ready_str = "true" if search_ready else "false"

    saved_voice_str = "true" if saved_voice else "false"
    saved_theatre_str = "true" if (saved_theatre or is_theatre) else "false"
    saved_rhythm_str = "true" if (saved_rhythm or is_rhythm) else "false"

    audio_source = (
        "<source src='data:audio/wav;base64," + rendered_audio_b64 + "' type='audio/wav'>"
        if has_audio else ""
    )

    title_colors_json = json.dumps(_TITLE_COLORS, ensure_ascii=False)
    label_texts_json = json.dumps(_LABEL_TEXTS, ensure_ascii=False)
    params_config_json = json.dumps(_PARAMS_CONFIG, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ width:100%; height:100vh; overflow:hidden; font-family:'Segoe UI',sans-serif; background:#000; user-select:none; color:#fff; }}
.bg-dark {{ position:fixed; inset:0; {bg_css} opacity:0.55; filter:saturate(130%) contrast(80%); z-index:0; transition:opacity 0.8s ease; pointer-events:none; }}
.bg-hi {{ position:fixed; inset:0; {bg_css} opacity:0; filter:saturate(180%) contrast(80%) brightness(108%); z-index:1; transition:opacity .35s ease; -webkit-mask-size:cover; mask-size:cover; -webkit-mask-position:center; mask-position:center; -webkit-mask-mode:luminance; mask-mode:luminance; -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat; pointer-events:none; }}

/* 容器通用样式 */
.step-container {{ position:fixed; top:50%; left:50%; transform:translate(-50%,-50%); z-index:20; width:100%; max-width:900px; text-align:center; opacity:1; transition:opacity 0.45s ease, transform 0.45s cubic-bezier(0.2,0.8,0.2,1); pointer-events:auto; }}
.step-container.hidden {{ opacity:0; pointer-events:none; transform:translate(-50%,-60%); }}
#step3 {{ max-width: 1000px; }}
#step25 {{ width: 100%; max-width: 100%; height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; background: transparent; }}

.back-btn {{ position:fixed; top:30px; left:30px; z-index:100; display:flex; align-items:center; gap:8px; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.2); padding:8px 16px; border-radius:20px; cursor:pointer; font-size:13px; font-weight:600; opacity:0; pointer-events:none; transition:all 0.3s ease; }}
.back-btn:hover {{ background:rgba(255,255,255,0.15); border-color:#fff; }}
.back-btn.show {{ opacity:1; pointer-events:auto; }}
.back-arrow {{ width:0;height:0;border-top:5px solid transparent;border-bottom:5px solid transparent;border-right:8px solid #fff; }}

.app-title {{ font-size:48px; font-weight:800; margin-bottom:8px; text-shadow:0 4px 12px rgba(0,0,0,0.6); transition:color 0.3s ease; }}
.app-sub {{ font-size:14px; color:rgba(255,255,255,0.8); margin-bottom:30px; text-shadow:0 2px 4px rgba(0,0,0,0.8); }}
.sec-label {{ font-size:16px; font-weight:700; letter-spacing:2px; color:rgba(255,255,255,0.9); margin-bottom:20px; text-shadow:0 1px 3px rgba(0,0,0,0.5); }}

.toggle-row {{ display:flex; gap:20px; justify-content:center; margin-bottom:22px; transition:opacity 0.4s ease; }}
.tgl {{ display:flex; align-items:center; gap:8px; cursor:pointer; padding:8px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.2); backdrop-filter:blur(4px); transition:background 0.2s; }}
.tgl:hover {{ background:rgba(0,0,0,.4); }}
.tgl input {{ display:none; }}
.tgl-track {{ width:36px; height:20px; border-radius:10px; background:rgba(255,255,255,.2); position:relative; margin-right:8px; }}
.tgl-thumb {{ position:absolute; top:3px; left:3px; width:14px; height:14px; border-radius:7px; background:#888; transition:left .22s ease; }}
.tgl-lbl {{ font-size:12.5px; text-shadow:0 1px 2px rgba(0,0,0,0.5); }}
.hint-text {{ font-size:11px; color:rgba(255,255,255,0.7); margin-top:20px; text-shadow:0 1px 3px rgba(0,0,0,0.8); transition:opacity 0.4s ease; }}

.params-panel {{ background:rgba(30,30,35,0.95); border:1px solid rgba(255,255,255,0.1); border-radius:16px; padding:30px; width:100%; max-width:500px; margin:0 auto 30px; text-align:left; box-shadow:0 10px 40px rgba(0,0,0,0.6); }}
.slider-row {{ margin-bottom:24px; }}
.slider-header {{ display:flex; justify-content:space-between; margin-bottom:8px; }}
.slider-label {{ font-size:13px; font-weight:600; color:#ddd; }}
.slider-val {{ font-size:13px; color:#ff6b6b; font-family:monospace; font-weight:700; }}
input[type=range] {{ -webkit-appearance:none; width:100%; background:transparent; }}
input[type=range]:focus {{ outline:none; }}
input[type=range]::-webkit-slider-runnable-track {{ width:100%; height:4px; cursor:pointer; background:rgba(255,255,255,0.2); border-radius:2px; }}
input[type=range]::-webkit-slider-thumb {{ height:16px; width:16px; border-radius:50%; background:#ff6b6b; cursor:pointer; -webkit-appearance:none; margin-top:-6px; box-shadow:0 0 10px rgba(255,107,107,0.4); }}

.options-grid {{ display:flex; gap:20px; justify-content:center; margin-bottom:30px; flex-wrap:wrap; }}
.option-card {{ width:160px; height:130px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.2); border-radius:16px; display:flex; flex-direction:column; align-items:center; justify-content:center; cursor:pointer; transition:all 0.25s; backdrop-filter:blur(8px); }}
.option-card:hover {{ background:rgba(255,255,255,0.15); transform:translateY(-4px); }}
.option-card.selected {{ background:rgba(124,80,232,0.5); border-color:#bb99ff; box-shadow:0 0 25px rgba(124,80,232,0.4); }}
.option-card.special-haru.selected {{ background:rgba(230,57,70,0.6); border-color:#ff99ac; box-shadow:0 0 25px rgba(230,57,70,0.5); }}
.card-icon {{ font-size:36px; margin-bottom:12px; }}
.card-title {{ font-size:14px; font-weight:700; }}
.card-desc {{ font-size:11px; color:rgba(255,255,255,0.6); margin-top:4px; }}

.confirm-btn {{ padding:12px 48px; background:#fff; color:#000; border:none; border-radius:30px; font-weight:800; cursor:pointer; font-size:14px; box-shadow:0 4px 15px rgba(255,255,255,0.2); transition:transform 0.2s; }}
.confirm-btn:hover {{ transform:scale(1.05); }}

.panorama-viewport {{ position: relative; width: 100%; height: 420px; perspective: 900px; overflow: hidden; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
.panorama-stage {{ position: relative; width: 100%; height: 280px; transform-style: preserve-3d; display: flex; align-items: center; justify-content: center; }}
.p-item {{ position: absolute; width: 440px; height: 260px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); transition: transform 0.6s cubic-bezier(0.19, 1, 0.22, 1), opacity 0.6s ease; cursor: pointer; display: flex; flex-direction: column; align-items: center; justify-content: center; -webkit-backface-visibility: hidden; backface-visibility: hidden; backdrop-filter: blur(5px); }}
.p-content {{ position: absolute; bottom: 20px; left: 20px; text-align: left; z-index: 2; pointer-events: none; }}
.p-title {{ font-size: 20px; font-weight: 800; color: #fff; text-shadow: 0 2px 10px rgba(0,0,0,0.8); margin-bottom: 4px; letter-spacing: 1px; }}
/* 新增：分数显示与弹窗样式 */
.p-score-row {{ display:flex; gap:12px; margin-top:6px; }}
.p-score-item {{ display:flex; flex-direction:column; align-items:flex-start; }}
.p-badge {{ font-size:10px; font-weight:800; padding:2px 6px; border-radius:4px; margin-bottom:2px; letter-spacing:1px; }}
.p-val {{ font-size:16px; font-weight:800; font-family:'Courier New', monospace; text-shadow:0 0 5px rgba(255,255,255,0.5); }}
.badge-ap {{ background:linear-gradient(135deg, #fcd34d, #f59e0b); color:#000; box-shadow:0 0 10px rgba(245,158,11,0.6); }}
.badge-fc {{ background:linear-gradient(135deg, #93c5fd, #3b82f6); color:#fff; box-shadow:0 0 10px rgba(59,130,246,0.6); }}
.badge-norm {{ background:rgba(255,255,255,0.2); color:#ddd; }}
.badge-chal {{ background:rgba(239,68,68,0.3); color:#fca5a5; border:1px solid #ef4444; }}

#diffModal {{ position:fixed; inset:0; background:rgba(0,0,0,0.8); backdrop-filter:blur(8px); z-index:300; display:none; flex-direction:column; align-items:center; justify-content:center; opacity:0; transition:0.3s; }}
.diff-card {{ width:500px; background:rgba(30,30,40,0.95); border:1px solid rgba(255,255,255,0.1); border-radius:24px; padding:40px; text-align:center; transform:scale(0.9); transition:0.3s cubic-bezier(0.19, 1, 0.22, 1); pointer-events:auto; }}
#diffModal.show {{ opacity:1; }}
#diffModal.show .diff-card {{ transform:scale(1); }}
.diff-opts {{ display:flex; gap:20px; justify-content:center; margin-top:20px; }}
.diff-btn {{ flex:1; height:140px; border-radius:16px; border:2px solid transparent; cursor:pointer; display:flex; flex-direction:column; align-items:center; justify-content:center; transition:0.2s; background:rgba(255,255,255,0.05); }}
.diff-btn:hover {{ transform:translateY(-5px); }}
.db-norm {{ border-color:#93c5fd; }}
.db-norm:hover {{ background:rgba(147,197,253,0.15); box-shadow:0 0 30px rgba(147,197,253,0.2); }}
.db-chal {{ border-color:#fca5a5; }}
.db-chal:hover {{ background:rgba(252,165,165,0.15); box-shadow:0 0 30px rgba(252,165,165,0.2); }}

.p-item.active {{ background: rgba(255, 255, 255, 0.1); box-shadow: 0 20px 60px rgba(0,0,0,0.7); z-index: 100; }}
.p-item.active {{ background: rgba(255, 255, 255, 0.1); box-shadow: 0 20px 60px rgba(0,0,0,0.7); z-index: 100; }}
.panorama-controls {{ margin-top: 40px; display: flex; align-items: center; gap: 30px; z-index: 200; }}
.pan-btn {{ width: 48px; height: 48px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.3); background: rgba(0,0,0,0.3); color: #fff; font-size: 20px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; backdrop-filter: blur(4px); }}
.pan-btn:hover {{ background: #fff; color: #000; transform: scale(1.1); }}
.pan-confirm {{ padding: 12px 40px; border-radius: 30px; background: #fff; color: #000; font-weight: 800; font-size: 14px; border: none; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 20px rgba(255,255,255,0.2); }}
.pan-confirm:hover {{ transform: scale(1.05); }}

/* ================= STEP 3 THEATRE LAYOUT ================= */
.theatre-layout {{ display: flex; gap: 60px; align-items: center; justify-content: center; margin-bottom: 30px; width: 100%; }}
.theatre-left {{ flex: 1; max-width: 520px; text-align: left; }}
.theatre-right {{ flex: 0 0 320px; position: relative; display: flex; align-items: center; justify-content: center; }}
.theatre-form {{ background:rgba(40,40,40,0.85); border:1px solid rgba(255,255,255,0.15); border-radius:20px; padding:30px 40px; width:100%; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }}
.form-group {{ margin-bottom:25px; }}
.form-label {{ font-size:13px; color:rgba(255,255,255,0.6); margin-bottom:12px; display:block; font-weight:600; letter-spacing:1px; }}
.capsule-group {{ display:flex; gap:10px; flex-wrap:wrap; }}
.capsule-opt {{ padding:8px 16px; background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2); border-radius:20px; font-size:13px; cursor:pointer; display:flex; align-items:center; gap:6px; transition:0.2s; }}
.capsule-opt.active {{ background:#fff; color:#000; font-weight:700; }}
.dot-indicator {{ width:8px; height:8px; border-radius:50%; background:rgba(255,255,255,0.4); }}
.capsule-opt.active .dot-indicator {{ background:#4361ee; }}
.input-area {{ width:100%; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.3); border-radius:10px; padding:12px; color:#fff; resize:none; outline:none; }}
.input-collapse {{ max-height:0; overflow:hidden; opacity:0; transition:all 0.4s; }}
.input-collapse.show {{ max-height:150px; opacity:1; margin-top:15px; }}

.dest-display {{ margin-top: 15px; font-size: 12px; color: #ccc; background: rgba(0,0,0,0.3); padding: 8px 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); width: fit-content; display: none; transition: 0.3s; }}
.dest-display span {{ color: #fff; font-weight: bold; margin-left: 5px; }}

.map-open-btn, .search-open-btn {{
    margin-top: 15px; background: linear-gradient(135deg, rgba(67, 97, 238, 0.2), rgba(67, 97, 238, 0.05));
    border: 1px solid rgba(100, 149, 237, 0.3); border-radius: 12px; padding: 12px 16px; color: #fff;
    font-size: 13px; font-weight: 600; cursor: pointer; display: none; align-items: center; gap: 8px;
    transition: all 0.3s ease; width: 100%; justify-content: center;
}}
.map-open-btn:hover, .search-open-btn:hover {{ background: rgba(67, 97, 238, 0.3); border-color: rgba(100, 149, 237, 0.8); transform: translateY(-2px); }}
.dest-status {{ font-size: 12px; color: #ccc; margin-left: auto; font-weight: 400; }}
.data-status-badge {{ background: rgba(46, 204, 113, 0.15); border: 1px solid rgba(46, 204, 113, 0.5); color: #2ecc71; font-size: 13px; font-weight: 600; padding: 10px 16px; border-radius: 12px; margin-top: 15px; display: none; align-items: center; justify-content: center; gap: 8px; box-shadow: 0 4px 15px rgba(46, 204, 113, 0.1); }}

/* --- ANIMATION STAGE (右侧演示区) --- */
.anim-stage {{ width: 280px; height: 280px; position: relative; }}
.anim-ring {{ fill: none; stroke: rgba(255,255,255,0.8); stroke-width: 4; stroke-linecap: round; transition: stroke-dasharray 0.8s cubic-bezier(0.4, 0, 0.2, 1); transform-origin: 140px 140px; animation: spinRing 24s linear infinite; }}
@keyframes spinRing {{ 100% {{ transform: rotate(360deg); }} }}
.center-anim {{ opacity: 0; visibility: hidden; transform-origin: 140px 140px; transform: scale(0.8); transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1); }}
.center-anim.active {{ opacity: 1; visibility: visible; transform: scale(1); }}
@keyframes tumble {{ 100% {{ transform: rotate(360deg); }} }}
.dice-spin {{ transform-origin: 140px 140px; animation: tumble 6s infinite linear; }}
@keyframes orbitPlane {{ 0% {{ transform: rotate(0deg) translateY(-55px) rotate(90deg); }} 100% {{ transform: rotate(360deg) translateY(-55px) rotate(90deg); }} }}
.plane-orbit {{ transform-origin: 140px 140px; animation: orbitPlane 4s linear infinite; }}
@keyframes floatNote {{ 0%,100%{{ transform: translateY(0); }} 50%{{ transform: translateY(-8px); }} }}
.float-note {{ animation: floatNote 2s ease-in-out infinite; }}
@keyframes flicker {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
.tv-flicker {{ animation: flicker 0.2s infinite; }}
@keyframes channelSwitch {{ 0%, 48% {{ fill: rgba(255,255,255,0.1); }} 50%, 98% {{ fill: rgba(100,200,255,0.3); }} 100% {{ fill: rgba(255,255,255,0.1); }} }}
.tv-screen {{ animation: channelSwitch 2s infinite; }}
@keyframes popBubble {{ 0%, 20% {{ opacity: 0; transform: scale(0.5); }} 40%, 80% {{ opacity: 1; transform: scale(1); }} 100% {{ opacity: 0; transform: scale(1.1); }} }}
.b-pop1 {{ transform-origin: 155px 120px; animation: popBubble 3s infinite; animation-delay: 0s; }}
.b-pop2 {{ transform-origin: 180px 90px;  animation: popBubble 3s infinite; animation-delay: 0.3s; }}
.b-pop3 {{ transform-origin: 220px 50px;  animation: popBubble 3s infinite; animation-delay: 0.6s; }}
@keyframes waveArm {{ 0% {{ transform: rotate(-25deg); }} 100% {{ transform: rotate(45deg); }} }}
.wave-arm {{ transform-origin: 140px 150px; animation: waveArm 0.6s ease-in-out infinite alternate; }}
@keyframes handSpin {{ 100% {{ transform: rotate(360deg); }} }}
.min-hand {{ transform-origin: 140px 140px; animation: handSpin 12s linear infinite; }}
.sec-hand {{ transform-origin: 140px 140px; animation: handSpin 1s steps(60) infinite; }}

/* ================= 音游模式样式 ================= */
#rhythmContainer {{ position:absolute; inset:0; z-index:25; display:none; flex-direction:column; align-items:center; justify-content:center; pointer-events:none; }}
#rhythmCanvas {{ position:absolute; width:100%; height:100%; top:0; left:0; pointer-events:none; }}

/* 顶部/居中 UI */
.rhythm-ui-layer {{ position:absolute; top:8%; width:100%; display:flex; justify-content:center; align-items:flex-start; pointer-events:none; z-index:30; }}
.score-box {{ position:absolute; right:40px; top:20px; display:flex; flex-direction:column; align-items:flex-end; }}
.score-num {{ font-size:42px; font-weight:900; color:#fff; font-family:'Courier New', Courier, monospace; text-shadow: 0 0 15px rgba(255,255,255,0.5); }}
.score-title {{ font-size:14px; color:#ccc; font-weight:800; letter-spacing:4px; }}

/* 巨型连击数 */
.combo-box {{ display:flex; flex-direction:column; align-items:center; opacity:0; transition:opacity 0.3s; transform:scale(1); margin-top: 5vh; }}
.combo-box.active {{ opacity:1; }}
.combo-num {{ font-size:110px; font-weight:900; color:#fff; text-shadow:0 0 40px rgba(255,255,255,0.9), 0 0 10px rgba(255,255,255,0.5); font-style:italic; line-height: 1; }}
.combo-title {{ font-size:22px; color:rgba(255,255,255,0.8); font-weight:800; letter-spacing:8px; margin-top:-10px; z-index:2; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }}
@keyframes comboPop {{
    0% {{ transform: scale(1.3); }}
    100% {{ transform: scale(1); }}
}}
.combo-pop {{ animation: comboPop 0.15s cubic-bezier(0.2, 0.8, 0.2, 1) forwards; }}

/* 结算界面 */
#settlementScreen {{ position:absolute; inset:0; background:rgba(0,0,0,0.85); backdrop-filter:blur(10px); z-index:100; display:none; flex-direction:column; align-items:center; justify-content:center; opacity:0; transition:0.5s; }}
.set-card {{ background:rgba(30,30,40,0.8); border:1px solid rgba(147,197,253,0.3); border-radius:24px; padding:40px 60px; text-align:center; box-shadow:0 15px 50px rgba(0,0,0,0.5), inset 0 0 20px rgba(147,197,253,0.1); width:90%; max-width:500px; pointer-events:auto; }}
.set-title {{ font-size:32px; font-weight:800; color:#fff; margin-bottom:10px; }}
.set-rank {{ font-size:90px; font-weight:900; background:-webkit-linear-gradient(#fde68a, #f59e0b); -webkit-background-clip:text; -webkit-text-fill-color:transparent; filter:drop-shadow(0 0 25px rgba(245,158,11,0.6)); margin-bottom:20px; font-style:italic; }}
.set-stats {{ display:flex; flex-direction:column; gap:12px; margin-bottom:35px; font-size:20px; color:#ddd; font-weight:600; }}
.set-stat-row {{ display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:5px; }}
.stat-perfect {{ color:#86efac; font-weight:bold; }}
.stat-great {{ color:#93c5fd; font-weight:bold; }}
.stat-miss {{ color:#fca5a5; font-weight:bold; }}

/* ============================================================== */

.player-wrapper {{ display:flex; flex-direction:column; align-items:center; justify-content:center; position:relative;height: auto; min-height: 280px; }}
.play-circle {{ width:110px; height:110px; border-radius:50%; background:#444; border:3px solid rgba(255,255,255,0.1); display:flex; align-items:center; justify-content:center; cursor:default; transition:all 1.2s; position:relative; margin:105px auto 5px; }}
.s4-play-icon {{ width:0; height:0; border-style:solid; border-width:14px 0 14px 24px; border-color:transparent transparent transparent rgba(255,255,255,0.3); margin-left:4px; }}
.play-circle.ready {{ cursor:pointer; transform:scale(1.1); }}
.play-circle.ready.mode-normal {{ background:#e63946; border-color:#ff99ac; box-shadow:0 0 40px rgba(230,57,70,0.4); }}
.play-circle.ready.mode-theatre {{ background:#4361ee; border-color:#99aaff; box-shadow:0 0 40px rgba(67,97,238,0.4); }}
.play-circle.ready .s4-play-icon {{ border-color:transparent transparent transparent #fff; }}
#renderOverlay {{ display:none; position:fixed; inset:0; z-index:9999; background:rgba(6,4,16,0.92); flex-direction:column; align-items:center; justify-content:center; gap:24px; }}
#renderOverlay.show {{ display:flex; }}
.ro-spinner {{ width:48px; height:48px; border:3px solid rgba(255,255,255,0.12); border-top-color:#ff4b4b; border-radius:50%; animation:roSpin 0.8s linear infinite; }}
.ro-text {{ font-size:13px; letter-spacing:3px; color:rgba(255,255,255,0.6); }}
@keyframes roSpin {{ to {{ transform:rotate(360deg); }} }}
.actions-row {{ display:flex; gap:20px; justify-content:center; opacity:0; transform:translateY(20px); transition:all 0.6s ease 0.2s; pointer-events:none; }}
.actions-row.visible {{ opacity:1; transform:translateY(0); pointer-events:auto; }}
.action-btn {{ font-size:13px; color:rgba(255,255,255,0.7); background:transparent; border:1px solid rgba(255,255,255,0.3); padding:8px 18px; border-radius:20px; cursor:pointer; transition:0.2s; }}
.action-btn:hover {{ background:rgba(255,255,255,0.1); color:#fff; border-color:#fff; }}
.render-status {{ font-size:13px; color:rgba(255,255,255,0.5); margin-top:-30px; min-height:20px; letter-spacing:1px; text-align:center; }}
#pureMode {{ position:fixed; inset:0; z-index:200; display:none; align-items:center; justify-content:center; background:transparent; }}
#pureMode.active {{ display:flex; }}
.pure-exit {{ position:fixed; top:24px; right:28px; z-index:300; width:36px; height:36px; border-radius:50%; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.25); color:rgba(255,255,255,0.7); font-size:16px; cursor:pointer; display:flex; align-items:center; justify-content:center; transition:0.25s; pointer-events:auto; }}
.pure-exit:hover {{ background:rgba(255,75,75,0.25); border-color:#ff4b4b; transform:scale(1.1); }}
.pure-player-wrap {{ position:relative; z-index:10; display:flex; flex-direction:column; align-items:center; gap:20px; }}
.pure-player-container {{ position:relative; width:220px; height:220px; display:flex; align-items:center; justify-content:center; }}
.progress-ring {{ position:absolute; transform:rotate(-90deg); z-index:10; pointer-events:none; }}
.progress-ring-circle {{ transition:stroke-dashoffset 0.1s linear; stroke:#ff4b4b; stroke-linecap:round; stroke-dasharray:408.4; stroke-dashoffset:408.4; }}
.progress-ring-bg {{ stroke:rgba(255,255,255,0.1); }}
.pure-play-btn {{ position:absolute; width:110px; height:110px; background:linear-gradient(135deg,#ff4b4b,#ff6b6b); border:none; border-radius:50%; cursor:pointer; display:flex; align-items:center; justify-content:center; transition:all 0.3s; box-shadow:0 10px 30px rgba(255,75,75,0.4); z-index:20; }}
.pure-play-btn:hover {{ transform:scale(1.06); }}
.pp-icon,.pp-pause {{ width:48px; height:48px; fill:white; }}
.pp-pause {{ display:none; }}
.pure-play-btn.playing .pp-icon {{ display:none; }}
.pure-play-btn.playing .pp-pause {{ display:block; }}
.vis-container {{ position:absolute; width:100%; height:100%; display:flex; align-items:center; justify-content:center; pointer-events:none; }}
.bar {{ position:absolute; bottom:50%; width:2px; background:#ff4b4b; transform-origin:center bottom; border-radius:2px; opacity:0.6; }}
.pure-time {{ color:rgba(255,255,255,0.55); font-size:13px; letter-spacing:2px; }}
.scene-title {{ font-size:12px; letter-spacing:2.5px; color:rgba(255,255,255,0.9); background:linear-gradient(135deg,rgba(20,15,40,0.85),rgba(40,25,60,0.85)); backdrop-filter:blur(16px); padding:7px 22px; border-radius:24px; border:1px solid rgba(180,140,255,0.3); box-shadow:0 0 16px rgba(150,100,255,0.2); opacity:0; pointer-events:none; margin-bottom:16px; }}
.bubble-wrap {{ position:absolute; max-width:210px; min-width:110px; pointer-events:none; }}
.bubble-name {{ font-size:11px; margin-bottom:5px; letter-spacing:1px; font-weight:600; text-shadow:0 1px 4px rgba(0,0,0,0.8); }}
.bubble-box {{ position:relative; background:rgba(15,12,26,0.78); color:#fff; padding:9px 13px; border-radius:14px; font-size:13px; line-height:1.65; backdrop-filter:blur(12px); word-break:break-all; text-shadow:0 1px 3px rgba(0,0,0,0.7); }}
.bubble-tail {{ position:absolute; }}
.narration-box {{font-size: 13px;line-height: 1.6;letter-spacing: 1px;color: rgba(255, 255, 255, 0.95);text-align: center; background: linear-gradient(135deg, rgba(15, 10, 30, 0.95), rgba(30, 20, 50, 0.95)); border: 1px solid rgba(180, 140, 255, 0.3); border-radius: 16px; padding: 12px 24px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 0 20px rgba(150, 100, 255, 0.15); transform: translateZ(0); }}
@keyframes titleFadeIn {{ 0%{{ opacity:0; transform:translateY(-6px); }} 100%{{ opacity:1; transform:translateY(0); }} }}
@keyframes titleFadeOut {{ 0%{{ opacity:1; }} 100%{{ opacity:0; transform:translateY(-6px); }} }}
@keyframes popIn {{ 0%{{ opacity:0; transform:translateY(10px) scale(0.95); }} 100%{{ opacity:1; transform:translateY(0) scale(1); }} }}
@keyframes fadeOut {{ 0%{{ opacity:1; }} 100%{{ opacity:0; transform:translateY(-8px); }} }}
.global-footer {{
    position: fixed;
    bottom: 15px;
    left: 0;
    width: 100%;
    text-align: center;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    letter-spacing: 1px;
    pointer-events: none;
    z-index: 10;
    transition: opacity 0.4s ease;
}}
</style>
</head>
<body>

<div class="bg-dark" id="mainBg"></div>
<div class="bg-hi" id="bgHi"></div>
<div class="global-footer" id="globalFooter">
  免责声明：本项目为非营利性二创交流作品，仅供技术探讨。BanG Dream! 及 MyGO!!!!! 相关版权归 Bushiroad 等原权利人所有。MIDI来源：MidiShow (https://www.midishow.com/)
</div>
<div class="back-btn" id="globalBackBtn" onclick="goBack()"><div class="back-arrow"></div> BACK</div>
<div id="renderOverlay"><div class="ro-spinner"></div><div class="ro-text">RENDERING...</div></div>
<!-- 难度选择弹窗 -->
<div id="diffModal">
  <div class="diff-card">
    <div style="font-size:24px; font-weight:800; letter-spacing:2px; color:#fff;">SELECT DIFFICULTY</div>
    <div class="diff-opts">
      <div class="diff-btn db-norm" onclick="selectDifficulty('normal')">
        <div style="font-size:24px; font-weight:900; color:#93c5fd; margin-bottom:8px;">NORMAL</div>
        <div style="font-size:12px; opacity:0.7; color:#fff;">标准模式 · 无多押</div>
      </div>
      <div class="diff-btn db-chal" onclick="selectDifficulty('challenge')">
        <div style="font-size:24px; font-weight:900; color:#fca5a5; margin-bottom:8px;">CHALLENGE</div>
        <div style="font-size:12px; opacity:0.7; color:#fff;">重音双押 · 高密度</div>
      </div>
    </div>
    <div style="margin-top:20px; font-size:12px; color:#aaa; cursor:pointer;" onclick="closeDiffModal()">CANCEL</div>
  </div>
</div>

<!-- STEP 1 -->
<div class="step-container" id="step1">
  <div class="app-title">Karplus-Strong Studio</div>
  <div class="app-sub">物理建模合成 · BanG Dream It's MyGO !!!!! · 高保真弦振动</div>
  <div class="sec-label" id="roleLabel">请选择角色进入</div>
  <div class="toggle-row" id="toggleRow1">
    <label class="tgl" id="tglVoice"><input type="checkbox" id="voiceCk"><div class="tgl-track"><div class="tgl-thumb"></div></div><span class="tgl-lbl">🔊 语音</span></label>
    <label class="tgl" id="tglTheatre"><input type="checkbox" id="theatreCk"><div class="tgl-track"><div class="tgl-thumb"></div></div><span class="tgl-lbl">🎭 剧场</span></label>
    <label class="tgl" id="tglRhythm"><input type="checkbox" id="rhythmCk"><div class="tgl-track"><div class="tgl-thumb"></div></div><span class="tgl-lbl">🎮 音游</span></label>
    <label class="tgl" id="tglPiano"><input type="checkbox" id="pianoCk"><div class="tgl-track"><div class="tgl-thumb"></div></div><span class="tgl-lbl">🎹 钢琴</span></label>
    <label class="tgl" id="tglCustom"><input type="checkbox" id="customCk"><div class="tgl-track"><div class="tgl-thumb"></div></div><span class="tgl-lbl">🎛️ 参数</span></label>
  </div>
  <div class="hint-text" id="hintText">悬停角色高亮，点击立绘进入</div>
</div>

<!-- STEP 1.5 -->
<div class="step-container hidden" id="step15">
  <div class="app-title" id="paramTitle">Instrument Params</div>
  <div class="app-sub">精细化调整物理建模参数</div>
  <div class="params-panel" id="paramsPanel"></div>
  <button class="confirm-btn" id="btnParamConfirm">应用设置 · APPLY</button>
</div>

<!-- STEP 1.6 音游设置 -->
<div class="step-container hidden" id="step16">
  <div class="app-title">Rhythm Settings</div>
  <div class="app-sub">4K 下落式音游参数调节</div>
  <div class="params-panel">
    <div class="slider-row">
      <div class="slider-header"><span class="slider-label">下落速度 (Speed)</span><span class="slider-val" id="val_speed">4.0</span></div>
      <input type="range" min="2.0" max="8.0" step="0.5" value="4.0" oninput="STATE.rhythmSpeed=parseFloat(this.value); document.getElementById('val_speed').innerText=this.value.includes('.') ? this.value : this.value + '.0';">
    </div>
    <div class="slider-row">
      <div class="slider-header"><span class="slider-label">判定偏移 (Offset ms)</span><span class="slider-val" id="val_offset">0</span></div>
      <input type="range" min="-200" max="200" step="10" value="0" oninput="STATE.rhythmOffset=parseInt(this.value); document.getElementById('val_offset').innerText=this.value;">
    </div>
  </div>
  <button class="confirm-btn" onclick="goTo(2)">确认设置 · CONFIRM</button>
</div>

<!-- STEP 2: MIDI Source Selection -->
<div class="step-container hidden" id="step2">
  <div class="app-title">Select MIDI Source</div>
  <div class="app-sub">请选择乐曲的驱动方式</div>
  <div class="options-grid" id="midiGrid">
    <div class="option-card special-haru" onclick="selectMidi('haruhikage',this)">
      <div class="card-icon">😡</div><div class="card-title">为什么要演奏春日影</div><div class="card-desc">SPECIAL PRESET</div>
    </div>
    <div class="option-card" onclick="selectMidi('preset',this)">
      <div class="card-icon">💿</div><div class="card-title">内置 MIDI 库</div><div class="card-desc">BUILT-IN LIBRARY</div>
    </div>
    <div class="option-card" onclick="selectMidi('upload',this)">
      <div class="card-icon">📂</div><div class="card-title">上传自己的 MIDI</div><div class="card-desc">CUSTOM UPLOAD</div>
    </div>
  </div>
  <button class="confirm-btn" id="midiConfirmBtn" style="display:none" onclick="onMidiConfirm()">确认 · CONFIRM</button>
</div>

<!-- STEP 2.5: Full Screen Panorama Song Selection -->
<div class="step-container hidden" id="step25">
  <div class="app-title" style="margin-bottom:40px;">Select a Song</div>
  <div class="panorama-viewport">
    <div class="panorama-stage" id="panoramaStage"></div>
  </div>
  <div class="panorama-controls">
    <div class="pan-btn" onclick="movePanorama(-1)">&#8592;</div>
    <button class="pan-confirm" onclick="onPanoramaConfirm()">确认选择</button>
    <div class="pan-btn" onclick="movePanorama(1)">&#8594;</div>
  </div>
  <div class="back-btn show" style="top:30px;left:30px;" onclick="goTo(2)"><div class="back-arrow"></div> BACK</div>
</div>

<!-- STEP 3: THEATRE SETTINGS -->
<div class="step-container hidden" id="step3">
  <div class="app-title">Global Settings</div>
  <div class="app-sub">小剧场模式 · 即兴演出设定</div>

  <div class="theatre-layout">
    <!-- 左侧面板 -->
    <div class="theatre-left">
      <div class="theatre-form">
        <div class="form-group"><span class="form-label">剧场格式 · FORMAT</span>
          <div class="capsule-group" id="fmtGroup">
            <div class="capsule-opt active" data-val="mini" onclick="pickCapsule(this,'fmt')"><div class="dot-indicator"></div>迷你小剧场</div>
            <div class="capsule-opt" data-val="episode" onclick="pickCapsule(this,'fmt')"><div class="dot-indicator"></div>再看一集</div>
          </div>
        </div>
        <div class="form-group" id="depthGroup"><span class="form-label">小剧场模式 · MODE</span>
          <div class="capsule-group" id="depthCapsules">
            <div class="capsule-opt active" data-val="casual" onclick="pickCapsule(this,'depth')"><div class="dot-indicator"></div>随便聊聊</div>
            <div class="capsule-opt" data-val="deep" onclick="pickCapsule(this,'depth')"><div class="dot-indicator"></div>深度聊聊</div>
            <div class="capsule-opt" data-val="lifetime" onclick="pickCapsule(this,'depth')"><div class="dot-indicator"></div>聊一辈子</div>
          </div>
        </div>

        <div class="form-group"><span class="form-label">请选择将要发生的事情 · SCENARIO</span>
          <!-- 迷你小剧场面板 -->
          <div id="scenarioWrapMini">
            <div class="capsule-group">
              <div class="capsule-opt active" data-val="random" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>🎲 全随机</div>
              <div class="capsule-opt" data-val="travel" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>✈️ 出游旅行</div>
              <div class="capsule-opt" data-val="daily" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>🏠 日常生活</div>
            </div>
            <div class="capsule-group" style="margin-top:10px">
              <div class="capsule-opt" data-val="fancreation" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>📚 二创研讨</div>
              <div class="capsule-opt" data-val="future" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>🔮 畅想未来</div>
              <div class="capsule-opt" data-val="break_wall" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>🎭 打破第四面墙</div>
              <div class="capsule-opt" data-val="custom" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>✍️ 自定义剧本</div>
            </div>
          </div>

          <!-- 再看一集面板 -->
          <div id="scenarioWrapEpisode" style="display:none;">
            <div class="capsule-group">
              <div class="capsule-opt active" data-val="travel" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>✈️ 出游旅行</div>
              <div class="capsule-opt" data-val="fancreation" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>📚 二创研讨</div>
              <div class="capsule-opt" data-val="news" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>📰 一手资讯</div>
              <div class="capsule-opt" data-val="memories" onclick="pickCapsule(this,'scenario')"><div class="dot-indicator"></div>🌌 回忆与现实交织</div>
            </div>

            <div id="mapEntryBtn" class="map-open-btn" onclick="navigateToMap()">
              <span>📍 打开地图选择目的地</span>
              <span class="dest-status" id="destStatus">{current_dest if current_dest else "未选择 (全国随机)"}</span>
            </div>
            <div id="searchEntryBtn" class="search-open-btn" onclick="navigateToSearch('manual')">
              <span>🔍 搜索特定话题</span>
            </div>
            <div id="newsEntryBtn" class="search-open-btn" onclick="navigateToSearch('auto')">
              <span>🌐 获取最新资讯</span>
            </div>
            <div id="dataReadyBadge" class="data-status-badge">
                <span>✅ 已获取网络数据，可以开始渲染</span>
            </div>
          </div>
        </div>

        <div class="input-collapse" id="customPromptArea">
          <textarea class="input-area" id="customPrompt" rows="3" placeholder="在此输入自定义世界观或具体情境..." oninput="STATE.th_prompt=this.value"></textarea>
        </div>
      </div>
    </div>

    <div class="theatre-right">
      <div class="anim-stage">
        <svg viewBox="0 0 280 280" width="280" height="280" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
          <!-- 动画内容省略... -->
          <circle id="animRing" class="anim-ring" cx="140" cy="140" r="130" stroke-dasharray="150 50" />
          <g id="anim-random" class="center-anim active"><g class="dice-spin"><rect x="90" y="90" width="100" height="100" rx="16"/><circle cx="120" cy="120" r="4" fill="#fff" stroke="none"/><circle cx="160" cy="160" r="4" fill="#fff" stroke="none"/><circle cx="120" cy="160" r="4" fill="#fff" stroke="none"/><circle cx="160" cy="120" r="4" fill="#fff" stroke="none"/><circle cx="140" cy="140" r="4" fill="#fff" stroke="none"/></g></g>
          <g id="anim-travel" class="center-anim"><circle cx="140" cy="140" r="50"/><ellipse cx="140" cy="140" rx="20" ry="50"/><line x1="90" y1="140" x2="190" y2="140"/><g class="plane-orbit"><path d="M140 50 L150 75 L130 75 Z" fill="#fff" stroke-linejoin="miter"/></g></g>
          <g id="anim-daily" class="center-anim" stroke-width="3"><circle cx="140" cy="115" r="8"/><path d="M140 123 v 20 M140 143 l -10 15 M140 143 l 10 15 M130 130 h 20"/><g transform="translate(-35, -15)"><circle cx="140" cy="115" r="8"/><path d="M140 123 v 20 M140 143 l -10 15 M140 143 l 10 15 M130 130 h 20"/></g><g transform="translate(35, -10)"><circle cx="140" cy="115" r="8"/><path d="M140 123 v 20 M140 143 l -10 15 M140 143 l 10 15 M130 130 h 20"/></g><g transform="translate(-25, 30)"><circle cx="140" cy="115" r="8"/><path d="M140 123 v 20 M140 143 l -10 15 M140 143 l 10 15 M130 130 h 20"/></g><g transform="translate(25, 35)"><circle cx="140" cy="115" r="8"/><path d="M140 123 v 20 M140 143 l -10 15 M140 143 l 10 15 M130 130 h 20"/></g><g class="float-note" transform="translate(-30, 0)"><path d="M150 80 v 20 M150 80 c 10 0 10 10 10 10" stroke-width="2"/><circle cx="147" cy="100" r="3" fill="#fff" stroke="none"/></g><g class="float-note" transform="translate(40, -20)" style="animation-delay:-1s;"><path d="M130 80 v 20 M130 80 c 10 0 10 10 10 10" stroke-width="2"/><circle cx="127" cy="100" r="3" fill="#fff" stroke="none"/></g></g>
          <g id="anim-fancreation" class="center-anim"><rect x="80" y="100" width="120" height="90" rx="8"/><path d="M110 100 L90 70 M170 100 L190 70"/><rect x="90" y="110" width="100" height="70" rx="4" class="tv-screen" fill="rgba(255,255,255,0.1)" stroke="none"/><circle cx="160" cy="145" r="8" stroke="none" fill="#fff" class="tv-flicker"/></g>
          <g id="anim-news" class="center-anim"><circle cx="140" cy="140" r="40"/><path d="M100 140 H180 M140 100 V180 M115 115 L165 165 M165 115 L115 165" stroke-width="2"/><circle cx="140" cy="140" r="60" stroke-dasharray="10 10" class="dice-spin"/></g>
          <g id="anim-future" class="center-anim"><circle cx="110" cy="150" r="12"/><path d="M110 162 v 40 M110 202 l -15 20 M110 202 l 15 20 M110 175 l 20 -10"/><circle cx="145" cy="125" r="5" class="b-pop1"/><circle cx="165" cy="100" r="10" class="b-pop2"/><circle cx="205" cy="65" r="25" class="b-pop3"/></g>
          <g id="anim-break_wall" class="center-anim"><circle cx="140" cy="130" r="16"/><path d="M140 146 v 50 M140 196 l -20 40 M140 196 l 20 40 M140 160 l -30 20"/><g class="wave-arm"><path d="M140 160 Q 180 150 180 110"/><circle cx="180" cy="110" r="4" fill="#fff"/></g></g>
          <g id="anim-memories" class="center-anim"><line x1="140" y1="140" x2="140" y2="70" stroke-width="6" class="min-hand"/><line x1="140" y1="150" x2="140" y2="50" stroke="#ff4b4b" stroke-width="2" class="sec-hand"/><circle cx="140" cy="140" r="6" fill="#fff" stroke="none"/></g>
          <g id="anim-custom" class="center-anim"><path d="M110 170 L160 120 L180 140 L130 190 Z" /><path d="M110 170 L90 190 L130 190 Z" fill="#fff"/><line x1="80" y1="200" x2="200" y2="200" stroke-width="2"/></g>
        </svg>
      </div>
    </div>
  </div>

  <button class="confirm-btn" id="renderBtn" onclick="goTo(4)">开始渲染 · RENDER</button>
</div>

<!-- STEP 4 -->
<div class="step-container hidden" id="step4">
  <div class="player-wrapper">
    <div class="play-circle" id="playCircle"><div class="s4-play-icon"></div></div>
    <div class="sec-label" id="s4Label" style="font-size:12px; opacity:0.6; margin-top:20px;">AUDIO RENDERING...</div>
    <div class="actions-row" id="actionsRow">
      <button class="action-btn" onclick="reRender()">🔄 重新渲染</button>
      <button class="action-btn" onclick="exitToHome()">🏠 返回首页</button>
    </div>
    <div class="render-status" id="renderStatus"></div>
  </div>
</div>

<!-- STEP 5: 纯净播放 / 音游模式 -->
<div id="pureMode">
  <div class="pure-exit" onclick="exitPure()" title="退出">✕</div>
  <div class="pure-player-wrap">
    <div class="scene-title" id="sceneTitle"></div>
    <div class="pure-player-container">
      <div class="vis-container" id="visualizer"></div>
      <svg class="progress-ring" width="220" height="220" viewBox="0 0 220 220">
        <circle class="progress-ring-bg" cx="110" cy="110" r="65" fill="none" stroke-width="4"/>
        <circle class="progress-ring-circle" id="progressCircle" cx="110" cy="110" r="65" fill="none" stroke-width="4"/>
      </svg>
      <button class="pure-play-btn" id="purePlayBtn">
        <svg class="pp-icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
        <svg class="pp-pause" viewBox="0 0 24 24"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/></svg>
      </button>
    </div>
    <div id="narrationBubble" style="opacity:0;pointer-events:none;max-width:280px;text-align:center;z-index:30;position:relative;"></div>
    <div class="pure-time" id="timeLabel">00:00 / 00:00</div>
  </div>

  <!-- 音游画布与 UI -->
  <div id="rhythmContainer">
    <canvas id="rhythmCanvas"></canvas>
    <div class="rhythm-ui-layer">
      <div class="combo-box" id="comboWrap">
        <div class="combo-num" id="rgCombo">0</div>
        <div class="combo-title">COMBO</div>
      </div>
      <div class="score-box">
        <div class="score-title">SCORE</div>
        <div class="score-num" id="rgScore">000000</div>
      </div>
    </div>
  </div>

  <!-- 结算界面 -->
  <div id="settlementScreen">
    <div class="set-card">
      <div class="set-title">STAGE CLEAR</div>
      <div class="set-rank" id="setRank">S</div>
      <div class="set-stats">
        <div class="set-stat-row"><span class="stat-perfect">PERFECT</span><span id="setPerf">0</span></div>
        <div class="set-stat-row"><span class="stat-great">GREAT</span><span id="setGr">0</span></div>
        <div class="set-stat-row"><span class="stat-miss">MISS</span><span id="setMiss">0</span></div>
        <div class="set-stat-row"><span style="color:#aaa;">MAX COMBO</span><span id="setMaxCombo">0</span></div>
      </div>
      <button class="confirm-btn" onclick="exitPure()">退出结算</button>
    </div>
  </div>

  <div id="bubbleLayer" style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:20;"></div>
</div>

<audio id="mainAudio" crossorigin="anonymous">{audio_source}</audio>
{safe_audio_tag}

<script>
// 获取 Python 注入的状态
var HAS_SEARCH_DATA = {search_ready_str};

var STATE = {{
  step: {initial_step},
  instrument: '{saved_inst}' ? '{saved_inst}' : (SELECTED_INST ? SELECTED_INST : null),
  voice: {saved_voice_str},
  theatre: {saved_theatre_str},
  rhythm: {saved_rhythm_str},
  rhythmSpeed: 4.0,
  rhythmOffset: 0,
  diff: 'normal',
  piano: false, 
  customParams: false,
  midi: '{saved_midi}' ? '{saved_midi}' : null,
  song: '{saved_song}' ? '{saved_song}' : null,
  params: {{}},
  th_format: '{saved_fmt}' || 'mini', 
  th_depth: 'casual', 
  th_scenario: '{saved_scenario}' || 'random', 
  th_prompt: '',
  th_destination: '{current_dest}'
}};

var HAS_AUDIO      = {has_audio_str};
var IS_THEATRE     = {is_theatre_str};
var IS_RHYTHM_MODE = {is_rhythm_str};
var SELECTED_INST  = '{selected_instrument}';
var SONG_LIST      = {song_list_json};
var SONG_COVERS    = {song_covers_json};
var ALL_CHATS      = {theatre_data_json};
var BEATMAP        = {beatmap_json};
var CHAR_POSITIONS = {char_positions_json};
var CHAR_CFG       = {char_cfg_json};
var MASKS          = {{ {mask_js_entries} }};

var TITLE_COLORS   = {title_colors_json};
var LABEL_TEXTS    = {label_texts_json};
var PARAMS_CONFIG  = {params_config_json};
var STEPS = ['step1','step15','step16','step2','step25','step3','step4'];

function goTo(stepNum) {{
  STEPS.forEach(function(id) {{
    var el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  }});
  var targetId;
  if      (stepNum === 1)   targetId = 'step1';
  else if (stepNum === 1.5) targetId = 'step15';
  else if (stepNum === 1.6) targetId = 'step16';
  else if (stepNum === 2)   targetId = 'step2';
  else if (stepNum === 2.5) targetId = 'step25';
  else if (stepNum === 3)   targetId = 'step3';
  else if (stepNum === 4)   targetId = 'step4';

  STATE.step = stepNum;
  var showGlobalBack = (stepNum > 1 && stepNum !== 2.5);
  document.getElementById('globalBackBtn').classList.toggle('show', showGlobalBack);
  var footer = document.getElementById('globalFooter');
  if (footer) footer.style.opacity = (stepNum === 1) ? '1' : '0';

  setTimeout(function() {{
    var target = document.getElementById(targetId);
    if (target) target.classList.remove('hidden');

    if (stepNum === 1) document.getElementById('mainBg').style.opacity = '0.55';
    else document.getElementById('bgHi').style.opacity = '0';

    if (stepNum === 2.5) initPanorama();
    if (stepNum === 3) {{
        var fmtOpt = document.querySelector('.capsule-opt[data-val="' + STATE.th_format + '"]');
        if (fmtOpt) pickCapsule(fmtOpt, 'fmt');
        var scnWrapId = STATE.th_format === 'episode' ? '#scenarioWrapEpisode' : '#scenarioWrapMini';
        var scnOpt = document.querySelector(scnWrapId + ' .capsule-opt[data-val="' + STATE.th_scenario + '"]');
        if (scnOpt) pickCapsule(scnOpt, 'scenario');
        updateTheatreAnim();
    }}
    if (stepNum === 4)   initStep4();
  }}, 80);
}}

function goBack() {{
  var s = STATE.step;
  if (s === 5) {{ exitPure(); return; }}
  if (s === 4 && HAS_AUDIO) {{ exitToHome(); return; }}
  if (s === 4) goTo(STATE.theatre ? 3 : 2);
  else if (s === 3) goTo(2);
  else if (s === 2.5) goTo(2); 
  else if (s === 2) goTo(STATE.rhythm ? 1.6 : (STATE.customParams ? 1.5 : 1));
  else if (s === 1.6) goTo(STATE.customParams ? 1.5 : 1);
  else if (s === 1.5) goTo(1);
  else goTo(1);
}}

var btnParamConfirm = document.getElementById('btnParamConfirm');
if(btnParamConfirm) {{
  btnParamConfirm.onclick = function() {{
      goTo(STATE.rhythm ? 1.6 : 2);
  }};
}}

var maskCanvases = {{}};
var masksLoaded = 0;
var maskKeys = Object.keys(MASKS);
var titleEl  = document.querySelector('.app-title');
var labelEl  = document.getElementById('roleLabel');
var toggleRow = document.getElementById('toggleRow1');
var hintEl   = document.getElementById('hintText');

maskKeys.forEach(function(key) {{
  var img = new Image();
  img.onload = function() {{
    var canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    var ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
    var winR = canvas.width / canvas.height;
    var imgR = img.width / img.height;
    var dW, dH, dX, dY;
    if (winR > imgR) {{ dW = canvas.width; dH = canvas.width / imgR; dX = 0; dY = (canvas.height - dH) / 2; }}
    else {{ dW = canvas.height * imgR; dH = canvas.height; dX = (canvas.width - dW) / 2; dY = 0; }}
    ctx.drawImage(img, dX, dY, dW, dH);
    maskCanvases[key] = ctx;
    masksLoaded++;
  }};
  img.src = MASKS[key];
}});

var currentHover = null;
function showMask(key) {{
  var hi = document.getElementById('bgHi');
  var m = MASKS[key];
  if (m) {{
    hi.style.webkitMaskImage = 'url(' + m + ')';
    hi.style.maskImage = 'url(' + m + ')';
    hi.style.opacity = '1';
  }} else {{
    hi.style.opacity = '0';
  }}
}}

document.addEventListener('mousemove', function(e) {{
  if (STATE.step !== 1) return;
  if (masksLoaded < maskKeys.length) return;
  var x = e.clientX, y = e.clientY;
  var detected = null;
  var order = ['guitar', 'bass', 'drums', 'full_band'];
  for (var i = 0; i < order.length; i++) {{
    var ctx = maskCanvases[order[i]];
    if (!ctx) continue;
    try {{
      var px = ctx.getImageData(x, y, 1, 1).data;
      if (px[0]*0.299 + px[1]*0.587 + px[2]*0.114 > 50) {{ detected = order[i]; break; }}
    }} catch(err) {{}}
  }}
  if (detected && detected !== currentHover) {{
    currentHover = detected;
    showMask(detected);
    document.body.style.cursor = 'pointer';
    if (titleEl)  titleEl.style.color = TITLE_COLORS[detected] || TITLE_COLORS['default'];
    if (labelEl)  labelEl.textContent = LABEL_TEXTS[detected] || LABEL_TEXTS['default'];
    if (toggleRow) toggleRow.style.opacity = '0.2';
    if (hintEl)   hintEl.style.opacity = '0.2';
  }} else if (!detected && currentHover) {{
    currentHover = null;
    showMask(null);
    document.body.style.cursor = 'default';
    if (titleEl)  titleEl.style.color = TITLE_COLORS['default'];
    if (labelEl)  labelEl.textContent = LABEL_TEXTS['default'];
    if (toggleRow) toggleRow.style.opacity = '1';
    if (hintEl)   hintEl.style.opacity = '1';
  }}
}});

document.addEventListener('click', function(e) {{
  if (STATE.step !== 1) return;
  if (e.target.closest('.tgl') || e.target.closest('.back-btn')) return;
  if (!currentHover) return;

  STATE.voice   = document.getElementById('voiceCk').checked;
  STATE.theatre = document.getElementById('theatreCk').checked;
  STATE.rhythm  = document.getElementById('rhythmCk').checked;
  STATE.piano   = document.getElementById('pianoCk').checked;
  STATE.customParams = document.getElementById('customCk').checked;
  STATE.instrument = STATE.piano ? 'piano' : currentHover;

  if (STATE.voice) {{
    var ia = document.getElementById('introAudio');
    if (ia && ia.querySelector('source')) {{ ia.currentTime = 0; ia.play().catch(function(){{}}); }}
  }}
  if (STATE.customParams) {{ 
      buildParamSliders(STATE.instrument); 
      goTo(1.5); 
  }} else if (STATE.rhythm) {{
      goTo(1.6);
  }} else {{
      goTo(2);
  }}
}});

['tglVoice','tglTheatre','tglRhythm','tglPiano','tglCustom'].forEach(function(id) {{
  var lbl = document.getElementById(id);
  if (!lbl) return;
  lbl.addEventListener('click', function(e) {{
    e.stopPropagation();
    var ck = this.querySelector('input');

    if (id === 'tglTheatre' && !ck.checked) {{
      document.getElementById('rhythmCk').checked = false;
      updateToggleVisual('tglRhythm', false);
    }}
    if (id === 'tglRhythm' && !ck.checked) {{
      document.getElementById('theatreCk').checked = false;
      updateToggleVisual('tglTheatre', false);
    }}

    ck.checked = !ck.checked;
    updateToggleVisual(id, ck.checked);
  }});
}});

function updateToggleVisual(id, isChecked) {{
  var lbl = document.getElementById(id);
  var thumb = lbl.querySelector('.tgl-thumb');
  thumb.style.left = isChecked ? '18px' : '3px';
  lbl.querySelector('.tgl-track').style.background = isChecked ? '#7c50e8' : 'rgba(255,255,255,0.2)';
  thumb.style.background = isChecked ? '#fff' : '#888';
}}

function buildParamSliders(inst) {{
  var panel = document.getElementById('paramsPanel');
  var config = PARAMS_CONFIG[inst];
  var instTitles = {{"guitar":"Guitar Parameters","bass":"Bass Parameters","drums":"Drum Kit Parameters","piano":"Piano Parameters","full_band":"Band Mix Parameters"}};
  var pt = document.getElementById('paramTitle');
  if (pt) pt.textContent = instTitles[inst] || 'Parameters';
  if (!config) {{ panel.innerHTML = '<p style="color:#888">该乐器无可调参数</p>'; return; }}
  var html = '<div style="font-size:14px;color:#aaa;margin-bottom:20px;letter-spacing:1px">' + inst.toUpperCase() + ' 参数</div>';
  config.forEach(function(p) {{
    var prec = p.step < 0.01 ? 4 : (p.step < 0.1 ? 2 : 1);
    STATE.params[p.key] = p["def"];
    html += '<div class="slider-row"><div class="slider-header"><span class="slider-label">' + p.label + '</span><span class="slider-val" id="val_' + p.key + '">' + p["def"].toFixed(prec) + '</span></div><input type="range" min="' + p.min + '" max="' + p.max + '" step="' + p.step + '" value="' + p["def"] + '" oninput="onSlider(&quot;' + p.key + '&quot;,this.value,' + prec + ')"></div>';
  }});
  panel.innerHTML = html;
}}

function onSlider(key, val, prec) {{
  STATE.params[key] = parseFloat(val);
  var el = document.getElementById('val_' + key);
  if (el) el.textContent = parseFloat(val).toFixed(prec);
}}

var activePanIndex = 0;
var panItems = [];

function initPanorama() {{
  var stage = document.getElementById('panoramaStage');
  if (!stage) return;
  stage.innerHTML = '';
  panItems = [];

  SONG_LIST.forEach(function(s, i) {{
    var div = document.createElement('div');
    div.className = 'p-item';
    var displayName = s.replace(/\.midi?$|\.mid$/i, '');
    
    // 动态生成成绩 HTML (仅在音游模式下显示)
    var scoreHtml = '';
    if (STATE.rhythm) {{
        var kNorm = 'mygo_r_score_' + s + '_normal';
        var kChal = 'mygo_r_score_' + s + '_challenge';
        var scNorm = localStorage.getItem(kNorm) || '------';
        var scChal = localStorage.getItem(kChal) || '------';
        var badgeN = localStorage.getItem(kNorm+'_badge');
        var badgeC = localStorage.getItem(kChal+'_badge');
        
        var bN = badgeN ? '<span class="p-badge badge-'+badgeN.toLowerCase()+'">'+badgeN+'</span>' : '<span class="p-badge badge-norm">N</span>';
        var bC = badgeC ? '<span class="p-badge badge-'+badgeC.toLowerCase()+'">'+badgeC+'</span>' : '<span class="p-badge badge-chal">C</span>';
        scoreHtml = '<div class="p-score-row"><div class="p-score-item">'+bN+'<div class="p-val">'+scNorm+'</div></div><div class="p-score-item">'+bC+'<div class="p-val">'+scChal+'</div></div></div>';
    }}

    if (SONG_COVERS[s]) {{
      div.style.backgroundImage = "url('data:image/jpeg;base64," + SONG_COVERS[s] + "')";
      div.style.backgroundSize = "cover"; div.style.backgroundPosition = "center";
    }}
    div.innerHTML = '<div class="p-content"><div class="p-title">' + displayName + '</div>' + scoreHtml + '</div>';
    
    div.onclick = function() {{ activePanIndex = i; updatePanorama(); STATE.song = s; }};
    stage.appendChild(div);
    panItems.push({{ el: div, data: s }});
  }});
  if (panItems.length > 0) {{ STATE.song = panItems[0].data; updatePanorama(); }}
}}

function movePanorama(dir) {{
  if (!panItems.length) return;
  activePanIndex += dir;
  if (activePanIndex < 0) activePanIndex = panItems.length - 1;
  if (activePanIndex >= panItems.length) activePanIndex = 0;
  updatePanorama();
  STATE.song = panItems[activePanIndex].data;
}}

function updatePanorama() {{
  var count = panItems.length;
  var centerIdx = activePanIndex;
  var radius = 700; 
  var stepAngle = 20; 
  panItems.forEach(function(obj, i) {{
    var el = obj.el;
    var offset = i - centerIdx;
    if (offset > count / 2) offset -= count;
    if (offset < -count / 2) offset += count;

    var absOffset = Math.abs(offset);
    var theta = offset * stepAngle * (Math.PI / 180); 
    var x = radius * Math.sin(theta);
    var z = radius * Math.cos(theta) - radius; 
    var rotateY = -offset * stepAngle;
    var opacity = absOffset > 4 ? 0 : (1 - absOffset * 0.15);
    var zIndex = 100 - absOffset;
    el.style.transform = 'translate3d(' + x + 'px, 0, ' + z + 'px) rotateY(' + rotateY + 'deg)';
    el.style.opacity = opacity;
    el.style.zIndex = zIndex;
    if (i === centerIdx) el.classList.add('active');
    else el.classList.remove('active');
  }});
}}

function onPanoramaConfirm() {{
    if (STATE.theatre) goTo(3);
    else if (STATE.rhythm) {{
        document.getElementById('diffModal').style.display = 'flex';
        setTimeout(() => document.getElementById('diffModal').classList.add('show'), 10);
    }} else goTo(4);
}}

function closeDiffModal() {{
    document.getElementById('diffModal').classList.remove('show');
    setTimeout(() => document.getElementById('diffModal').style.display = 'none', 300);
}}

function selectDifficulty(diff) {{
    STATE.diff = diff;
    closeDiffModal();
    goTo(4); // 选完难度去渲染
}}

function selectMidi(type, cardEl) {{
  STATE.midi = type;
  document.querySelectorAll('#midiGrid .option-card').forEach(function(c) {{ c.classList.remove('selected'); }});
  cardEl.classList.add('selected');
  if (type === 'upload') {{ navigateToUpload(); }} 
  else if (type === 'preset') {{ goTo(2.5); }} 
  else {{ document.getElementById('midiConfirmBtn').style.display = 'inline-block'; }}
}}

function onMidiConfirm() {{ if (STATE.theatre) goTo(3); else goTo(4); }}

function getContextUrl() {{
    var p = new URLSearchParams();
    p.set('s_inst', STATE.instrument || '');
    p.set('s_voice', STATE.voice ? '1' : '0');
    p.set('s_theatre', STATE.theatre ? '1' : '0');
    p.set('s_rhythm', STATE.rhythm ? '1' : '0');
    p.set('s_midi', STATE.midi || '');
    if (STATE.song) p.set('s_song', STATE.song);
    p.set('s_fmt', STATE.th_format || 'mini');
    p.set('s_scenario', STATE.th_scenario || 'random');
    return p.toString();
}}

function updateTheatreAnim() {{
  var fmt = STATE.th_format; 
  var depth = STATE.th_depth; 
  var scenario = STATE.th_scenario;

  var ring = document.getElementById('animRing');
  var C = 2 * Math.PI * 130; 
  if (fmt === 'episode') {{
    ring.style.strokeDasharray = C + ' 0';
  }} else {{
    var segments = depth === 'casual' ? 5 : (depth === 'deep' ? 10 : 20);
    var dash = (C / segments) * 0.6;
    var gap = (C / segments) * 0.4;
    ring.style.strokeDasharray = dash + ' ' + gap;
  }}

  var allAnims = document.querySelectorAll('.center-anim');
  allAnims.forEach(function(el) {{ el.classList.remove('active'); }});

  var targetId = 'anim-' + scenario;
  if (fmt === 'episode' && scenario === 'memories') {{
    targetId = 'anim-memories';
  }}

  var targetEl = document.getElementById(targetId);
  if (targetEl) {{
    void targetEl.offsetWidth;
    targetEl.classList.add('active');
  }}

  var mapBtn = document.getElementById('mapEntryBtn');
  var searchBtn = document.getElementById('searchEntryBtn');
  var newsBtn = document.getElementById('newsEntryBtn');
  var renderBtn = document.getElementById('renderBtn');
  var badge = document.getElementById('dataReadyBadge');

  if (mapBtn) mapBtn.style.display = 'none';
  if (searchBtn) searchBtn.style.display = 'none';
  if (newsBtn) newsBtn.style.display = 'none';
  if (badge) badge.style.display = 'none';
  if (renderBtn) renderBtn.style.display = 'inline-block'; 

  if (fmt === 'episode') {{
      if (scenario === 'travel') {{
          if (mapBtn) mapBtn.style.display = 'flex';
      }} 
      else if (scenario === 'fancreation') {{
          if (HAS_SEARCH_DATA) {{
              if (badge) {{ badge.style.display = 'flex'; badge.innerHTML = "<span>✅ 已锁定二创话题，可开始渲染</span>"; }}
              if (searchBtn) {{ searchBtn.style.display = 'flex'; searchBtn.querySelector('span').innerText = "🔄 更换话题"; }}
          }} else {{
              if (renderBtn) renderBtn.style.display = 'none'; 
              if (searchBtn) searchBtn.style.display = 'flex';
          }}
      }} 
      else if (scenario === 'news') {{
          if (HAS_SEARCH_DATA) {{
              if (badge) {{ badge.style.display = 'flex'; badge.innerHTML = "<span>✅ 已获取最新资讯，可开始渲染</span>"; }}
              if (newsBtn) {{ newsBtn.style.display = 'flex'; newsBtn.querySelector('span').innerText = "🔄 刷新资讯"; }}
          }} else {{
              if (renderBtn) renderBtn.style.display = 'none';
              if (newsBtn) newsBtn.style.display = 'flex';
          }}
      }}
  }}
}}

function navigateToMap() {{
    navigateParent('?go_map=1&' + getContextUrl());
}}

function navigateToSearch(mode) {{
    navigateParent('?go_search=1&search_mode=' + mode + '&' + getContextUrl());
}}

function pickCapsule(el, group) {{
  if (group === 'scenario') {{
    el.closest('div[id^="scenarioWrap"]').querySelectorAll('.capsule-opt').forEach(function(c){{ c.classList.remove('active'); }});
  }} else {{
    el.parentElement.querySelectorAll('.capsule-opt').forEach(function(c){{ c.classList.remove('active'); }});
  }}
  el.classList.add('active');
  var val = el.dataset.val;

  if (group === 'fmt') {{
    STATE.th_format = val;
    var isEp = (val === 'episode');
    document.getElementById('depthGroup').style.display = isEp ? 'none' : '';

    var wrapMini = document.getElementById('scenarioWrapMini');
    var wrapEp   = document.getElementById('scenarioWrapEpisode');

    if (isEp) {{
      wrapMini.style.display = 'none';
      wrapEp.style.display   = 'block';
      var activeOpt = wrapEp.querySelector('.capsule-opt.active');
      if (activeOpt) STATE.th_scenario = activeOpt.dataset.val;
    }} else {{
      wrapMini.style.display = 'block';
      wrapEp.style.display   = 'none';
      var activeOpt = wrapMini.querySelector('.capsule-opt.active');
      if (activeOpt) STATE.th_scenario = activeOpt.dataset.val;
    }}
    document.getElementById('customPromptArea').classList.toggle('show', STATE.th_scenario === 'custom');

  }} else if (group === 'depth') {{ 
    STATE.th_depth = val; 
  }} else if (group === 'scenario') {{
    STATE.th_scenario = val;
    document.getElementById('customPromptArea').classList.toggle('show', val === 'custom');
  }}

  updateTheatreAnim();
}}

function initStep4() {{
  var circle = document.getElementById('playCircle');
  var label  = document.getElementById('s4Label');
  var actions = document.getElementById('actionsRow');
  var status = document.getElementById('renderStatus');
  if (HAS_AUDIO) {{
    circle.classList.add('ready', IS_THEATRE ? 'mode-theatre' : 'mode-normal');
    label.textContent = 'RENDER COMPLETE · READY';
    label.style.opacity = '0.9';
    label.style.letterSpacing = '2px';
    actions.classList.add('visible');
    status.textContent = '';
    circle.onclick = function() {{ enterPureMode(); }};
  }} else {{
    label.textContent = 'AUDIO RENDERING...';
    status.textContent = '点击播放按钮开始渲染';
    circle.style.cursor = 'pointer';
    circle.onclick = function() {{ submitRender(); }};
  }}
}}

function submitRender() {{
  var overlay = document.getElementById('renderOverlay');
  if (overlay) overlay.classList.add('show');
  document.getElementById('s4Label').textContent = 'AUDIO RENDERING...';
  document.getElementById('renderStatus').textContent = '请稍候，正在合成音频…';
  var circle = document.getElementById('playCircle');
  circle.style.cursor = 'default'; circle.onclick = null;
  var params = new URLSearchParams();
  params.set('sel_instrument', STATE.instrument);
  params.set('sel_voice', STATE.voice ? '1' : '0');
  params.set('sel_theatre', STATE.theatre ? '1' : '0');
  params.set('sel_rhythm', STATE.rhythm ? '1' : '0');
  if (STATE.rhythm) params.set('sel_difficulty', STATE.diff);
  params.set('sel_midi', STATE.midi || 'haruhikage');
  if (STATE.midi === 'preset' && STATE.song) params.set('sel_song', STATE.song);
  if (STATE.customParams) {{
    params.set('use_custom_params', '1');
    var entries = Object.entries(STATE.params);
    for (var i = 0; i < entries.length; i++) params.set('p_' + entries[i][0], String(entries[i][1]));
  }}
  if (STATE.theatre) {{
    params.set('th_format', STATE.th_format);
    params.set('th_depth', STATE.th_depth);
    params.set('th_scenario', STATE.th_scenario);
    if (STATE.th_prompt) params.set('th_prompt', STATE.th_prompt);
    if (STATE.th_destination) params.set('th_destination', STATE.th_destination);
  }}
  navigateParent('?' + params.toString());
}}

function reRender() {{ navigateParent('?go_landing=1'); }}
function exitToHome() {{ navigateParent('?go_landing=1'); }}
function navigateToUpload() {{ navigateParent('?go_upload=1&' + getContextUrl()); }}

function navigateParent(url) {{
  try {{
    var s = window.parent.document.createElement('script');
    s.textContent = 'window.location.search = "' + url.replace(/"/g, '\\\\"') + '"';
    window.parent.document.head.appendChild(s);
  }} catch(err) {{ window.location.href = url; }}
}}

var audio = document.getElementById('mainAudio');
var purePlayBtn = document.getElementById('purePlayBtn');
var progressCircle = document.getElementById('progressCircle');
var timeLabel = document.getElementById('timeLabel');
var circumference = 2 * Math.PI * 65;
var musicEnded = false;
var IS_EPISODE_MODE = ALL_CHATS.length > 0 && ALL_CHATS[0] && ALL_CHATS[0].title === '__intro__';
var EPISODE_INTRO = IS_EPISODE_MODE ? (ALL_CHATS[0].intro || '') : '';
var EPISODE_CHATS = IS_EPISODE_MODE ? ALL_CHATS.slice(1) : ALL_CHATS;
var sleep = function(ms) {{ return new Promise(function(r) {{ setTimeout(r, ms); }}); }};

var visualizer = document.getElementById('visualizer');
var barCount = 80, bars = [];
for (var i = 0; i < barCount; i++) {{
  var b = document.createElement('div'); b.className = 'bar'; visualizer.appendChild(b); bars.push(b);
}}
var animT = 0;
function drawVis() {{
  requestAnimationFrame(drawVis); animT += 0.02;
  for (var j = 0; j < barCount; j++) {{
    var r = j / barCount;
    var v = (Math.sin(r*Math.PI*6+animT) + Math.sin(r*Math.PI*10-animT*1.2)*0.4 + Math.cos(r*Math.PI*6+animT*0.5)*0.3 + 1.7) / 3.4;
    var vp = v * v;
    bars[j].style.height = (vp*30+3)+'px';
    bars[j].style.transform = 'rotate(' + (j*(360/barCount)) + 'deg) translateY(-73px) scaleY(' + (1+vp/2) + ')';
    bars[j].style.opacity = 0.2+vp*0.6;
  }}
}}
drawVis();

function enterPureMode() {{
  STATE.step = 5;
  STEPS.forEach(function(id) {{ var el = document.getElementById(id); if (el) el.classList.add('hidden'); }});
  document.getElementById('globalBackBtn').classList.remove('show');
  document.getElementById('mainBg').style.opacity = '0.60';
  var instKey = STATE.instrument || SELECTED_INST;
  if (instKey && MASKS[instKey]) showMask(instKey);

  document.getElementById('pureMode').classList.add('active');
  audio.currentTime = 0; 
  audio.play().catch(function(){{}});

  if (IS_RHYTHM_MODE && BEATMAP.length > 0) {{
      startRhythmGame();
  }} else if (IS_THEATRE && ALL_CHATS.length > 0) {{
      chatLoop();
  }}
}}

function exitPure() {{
  audio.pause();
  document.getElementById('pureMode').classList.remove('active');
  document.getElementById('mainBg').style.opacity = '0.55';
  document.getElementById('bgHi').style.opacity = '0';
  document.getElementById('bubbleLayer').innerHTML = '';
  document.getElementById('settlementScreen').style.display = 'none';
  document.getElementById('rhythmContainer').style.display = 'none';
  document.querySelector('.pure-player-wrap').style.display = 'flex';
  if(rgActive) stopRhythmGame(true);
  goTo(4);
}}

purePlayBtn.addEventListener('click', function() {{
  if (musicEnded) {{ musicEnded = false; audio.currentTime = 0; audio.play(); return; }}
  audio.paused ? audio.play() : audio.pause();
}});
audio.addEventListener('play', function() {{ purePlayBtn.classList.add('playing'); }});
audio.addEventListener('pause', function() {{ purePlayBtn.classList.remove('playing'); }});
audio.addEventListener('ended', function() {{ 
  purePlayBtn.classList.remove('playing'); 
  progressCircle.style.strokeDashoffset = circumference; 
  musicEnded = true; 
  if (rgActive) stopRhythmGame(false);
}});
audio.addEventListener('timeupdate', function() {{
  var p = (audio.currentTime / audio.duration) || 0;
  progressCircle.style.strokeDashoffset = circumference - p * circumference;
  var fmt = function(s) {{ return String(Math.floor(s/60)).padStart(2,'0') + ':' + String(Math.floor(s%60)).padStart(2,'0'); }};
  timeLabel.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration||0);
}});

function extractNarrations(lines) {{
  var cL = [], nT = [];
  lines.forEach(function(m) {{ if (CHAR_CFG[m.name]) cL.push(m); else if (m.text) nT.push(m.text); }});
  return {{ charLines: cL, narTexts: nT }};
}}
function renderBubbles(charLines) {{
  var layer = document.getElementById('bubbleLayer'); layer.innerHTML = '';
  charLines.forEach(function(msg, i) {{
    var name = msg.name, text = msg.text, cfg = CHAR_CFG[name]; if (!cfg) return;
    var pos = CHAR_POSITIONS[name] || {{top:'50%',left:'45%',anchor:'right'}};
    var color = cfg.color, emoji = cfg.emoji, anchor = pos.anchor, delay = i * 2.2;
    var tail = anchor === 'right'
      ? 'top:14px;left:-8px;border:8px solid transparent;border-right-color:'+color+'55;border-left:none;'
      : 'top:14px;right:-8px;border:8px solid transparent;border-left-color:'+color+'55;border-right:none;';
    var nAlign = anchor === 'right' ? 'text-align:left' : 'text-align:right';
    var w = document.createElement('div'); w.className = 'bubble-wrap';
    var posCSS;
    if (anchor === 'left') {{
      var rightPct = (100 - parseFloat(pos.left)) + '%';
      posCSS = 'top:'+pos.top+';right:'+rightPct+';left:auto;';
    }} else {{
      posCSS = 'top:'+pos.top+';left:'+pos.left+';';
    }}
    w.style.cssText = posCSS+'opacity:0;transform:translateY(10px) scale(0.95);animation:popIn 0.45s cubic-bezier(0.34,1.56,0.64,1) forwards;animation-delay:'+delay+'s;';
    w.innerHTML = '<div class="bubble-name" style="color:'+color+';'+nAlign+'">'+emoji+' '+name+'</div><div class="bubble-box" style="border:1.5px solid '+color+'66;box-shadow:0 4px 20px rgba(0,0,0,0.5),0 0 12px '+color+'22;"><div class="bubble-tail" style="'+tail+'"></div>'+text+'</div>';
    layer.appendChild(w);
  }});
}}
function showSceneTitle(title) {{
  var el = document.getElementById('sceneTitle'); if (!el || !title) return Promise.resolve();
  el.textContent = '\u2726 ' + title + ' \u2726';
  el.style.animation = 'titleFadeIn 0.5s ease forwards'; el.style.opacity = '1';
  return sleep(2000).then(function(){{ el.style.animation='titleFadeOut 0.5s ease forwards'; return sleep(500); }}).then(function(){{ el.style.opacity='0'; }});
}}
function showNarrations(narTexts) {{
  if (!narTexts || !narTexts.length) return Promise.resolve();
  var el = document.getElementById('narrationBubble'); if (!el) return Promise.resolve();
  var chain = Promise.resolve();
  narTexts.slice(0,3).forEach(function(txt) {{
    chain = chain.then(function() {{
      el.className='narration-box'; el.textContent='\u2014 '+txt+' \u2014';
      el.style.animation='none'; el.style.opacity='0';
      void el.offsetWidth;
      el.style.animation='titleFadeIn 0.4s ease forwards';
      setTimeout(function(){{ el.style.opacity='1'; }}, 450);
      return sleep(2800);
    }}).then(function(){{
      el.style.animation='titleFadeOut 0.4s ease forwards';
      return sleep(500);
    }}).then(function(){{ el.style.opacity='0'; el.className=''; }});
  }});
  return chain;
}}
function fadeOutAllBubbles() {{
  var layer = document.getElementById('bubbleLayer');
  layer.querySelectorAll('.bubble-wrap').forEach(function(w){{ w.style.animation='fadeOut 0.6s ease forwards'; }});
  return sleep(700).then(function(){{ layer.innerHTML=''; }});
}}
function showIntroCard(text) {{
  if (!text) return Promise.resolve();
  var card = document.createElement('div');
  card.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);max-width:520px;width:88%;z-index:50;background:linear-gradient(135deg,rgba(12,8,28,0.92),rgba(30,18,50,0.92));backdrop-filter:blur(20px);border:1px solid rgba(180,140,255,0.3);border-radius:20px;padding:28px 32px;box-shadow:0 0 40px rgba(130,80,255,0.2);color:rgba(255,255,255,0.88);font-size:14px;line-height:1.9;letter-spacing:0.4px;text-align:justify;animation:titleFadeIn 0.8s ease forwards;';
  card.textContent = text; document.body.appendChild(card);
  return sleep(6000).then(function(){{ card.style.animation='titleFadeOut 0.8s ease forwards'; return sleep(800); }}).then(function(){{ card.remove(); }});
}}
function chatLoop() {{
  var chats = IS_EPISODE_MODE ? EPISODE_CHATS : ALL_CHATS;
  if (!chats || !chats.length) return;
  function runEpisode() {{
    return showIntroCard(EPISODE_INTRO).then(function() {{
      var chain = Promise.resolve();
      chats.forEach(function(obj) {{
        chain = chain.then(function() {{
          var lines = obj.lines || obj, title = obj.title || '', res = extractNarrations(lines);
          return showSceneTitle(title).then(function(){{ return showNarrations(res.narTexts); }}).then(function() {{
            renderBubbles(res.charLines);
            return sleep(res.charLines.length * 4200 + 3000);
          }}).then(function(){{ return fadeOutAllBubbles(); }}).then(function(){{ return sleep(150); }});
        }});
      }});
      return chain;
    }});
  }}
  function runShuffle() {{
    var idx = []; for (var i = 0; i < chats.length; i++) idx.push(i);
    function next() {{
      if (!idx.length) return Promise.resolve();
      var ri = Math.floor(Math.random()*idx.length), obj = chats[idx.splice(ri,1)[0]];
      var lines = obj.lines||obj, title = obj.title||'', res = extractNarrations(lines);
      return showSceneTitle(title).then(function(){{ return showNarrations(res.narTexts); }}).then(function() {{
        renderBubbles(res.charLines);
        return sleep(res.charLines.length * 4200 + 2000);
      }}).then(function(){{ return fadeOutAllBubbles(); }}).then(function(){{ return sleep(200); }}).then(next);
    }}
    return next();
  }}
  (IS_EPISODE_MODE ? runEpisode() : runShuffle()).then(function() {{
    var ending = [
      {{"name":"爱音","text":"诶诶诶！？剧本用完了！？"}},
      {{"name":"立希","text":"退出去刷新页面，再进来就有新的了。"}},
      {{"name":"素世","text":"呵呵……总要给大家一点喘息的时间嘛。"}},
      {{"name":"灯","text":"谢谢你……陪我们到这里……"}},
      {{"name":"乐奈","text":"……抹茶。刷新。"}},
    ];
    renderBubbles(extractNarrations(ending).charLines);
  }});
}}

/* ================= 4K 伪3D斜轨音游引擎 ================= */
var rgCanvas, rgCtx;
var rgActive = false;
var rgStats = {{ perf: 0, great: 0, miss: 0, combo: 0, maxCombo: 0, score: 0 }};
// 修改：键位改为 S D K L
var laneKeys = ['S', 'D', 'J', 'K']; 
var laneColors = ['#f9a8d4', '#f9a8d4', '#93c5fd', '#93c5fd'];
var keyState = [false, false, false, false];
var hitEffects = [];
var floatTexts = []; 
var particles = [];

// --- 打击音效合成器 (修改为“咚”声) ---
var hitCtx = null;
function initAudioCtx() {{
    if (!hitCtx) hitCtx = new (window.AudioContext || window.webkitAudioContext)();
}}
function playHitSound() {{
    if (!hitCtx) initAudioCtx();
    if (hitCtx.state === 'suspended') hitCtx.resume();

    let t = hitCtx.currentTime;
    let osc = hitCtx.createOscillator();
    let gain = hitCtx.createGain();

    // 修改：使用正弦波模拟鼓声，频率从 150Hz 快速降到 40Hz
    osc.type = 'sine'; 
    osc.frequency.setValueAtTime(180, t); 
    osc.frequency.exponentialRampToValueAtTime(40, t + 0.1);

    // 包络：瞬间起音，快速衰减
    gain.gain.setValueAtTime(0.5, t); // 音量稍大一点
    gain.gain.exponentialRampToValueAtTime(0.01, t + 0.15);

    osc.connect(gain);
    gain.connect(hitCtx.destination);

    osc.start(t);
    osc.stop(t + 0.15);
}}

// --- 历史最高分管理 ---
function getHighScoreKey() {{
    // 优先使用 STATE.song (选曲模式)，如果是上传模式则用 'custom_upload'
    var songName = STATE.song;
    if (!songName && STATE.midi === 'upload') songName = 'custom_upload';
    if (!songName) songName = 'unknown_song';
    return 'mygo_rhythm_hiscore_' + songName;
}}

function startRhythmGame() {{
    initAudioCtx(); 
    document.getElementById('rhythmContainer').style.display = 'flex';
    document.querySelector('.pure-player-wrap').style.display = 'none'; 

    rgCanvas = document.getElementById('rhythmCanvas');
    rgCtx = rgCanvas.getContext('2d');
    rgCanvas.width = window.innerWidth;
    rgCanvas.height = window.innerHeight;

    rgActive = true;
    rgStats = {{ perf: 0, great: 0, miss: 0, combo: 0, maxCombo: 0, score: 0 }};
    updateRgUI();
    document.getElementById('comboWrap').classList.add('active');

    // 读取历史最高分显示（可选）
    // console.log("Current High Score:", localStorage.getItem(getHighScoreKey()));

    BEATMAP.forEach(n => n.hit = false);

    document.addEventListener('keydown', rgKeyDown);
    document.addEventListener('keyup', rgKeyUp);

    requestAnimationFrame(rgLoop);
}}

function stopRhythmGame(forceExit) {{
    rgActive = false;
    document.removeEventListener('keydown', rgKeyDown); document.removeEventListener('keyup', rgKeyUp);
    document.getElementById('comboWrap').classList.remove('active');
    if (forceExit) return;
    
    // 计算最终 100w 分
    let totalNotes = BEATMAP.length;
    let finalScore = totalNotes > 0 ? Math.floor(((rgStats.perf + rgStats.great * 0.65) / totalNotes) * 1000000) : 0;
    
    let rank = 'C';
    if (finalScore >= 950000) rank = 'S';
    else if (finalScore >= 850000) rank = 'A';
    else if (finalScore >= 700000) rank = 'B';
    
    // 保存逻辑
    var songName = STATE.song || (STATE.midi === 'upload' ? 'custom_upload' : 'unknown');
    var storeKey = 'mygo_r_score_' + songName + '_' + (STATE.diff || 'normal');
    var oldBest = parseInt(localStorage.getItem(storeKey) || '0');
    
    if (finalScore > oldBest) {{
        localStorage.setItem(storeKey, finalScore);
        oldBest = finalScore;
    }}
    
    // AP/FC 保存逻辑
    var badge = '';
    if (rgStats.miss === 0) {{
        badge = rgStats.great === 0 ? 'AP' : 'FC';
        var badgeKey = storeKey + '_badge';
        var oldBadge = localStorage.getItem(badgeKey);
        if (badge === 'AP' || (badge === 'FC' && oldBadge !== 'AP')) {{
            localStorage.setItem(badgeKey, badge);
        }}
    }}

    document.getElementById('setRank').innerText = rank;
    document.getElementById('setPerf').innerText = rgStats.perf;
    document.getElementById('setGr').innerText = rgStats.great;
    document.getElementById('setMiss').innerText = rgStats.miss;
    
    // 显示分数
    let mcEl = document.getElementById('setMaxCombo');
    mcEl.innerHTML = rgStats.maxCombo + '<br><div style="margin-top:12px;color:#fff;font-size:26px;">SCORE: ' + finalScore + '</div><div style="font-size:14px;color:#aaa;">BEST: ' + oldBest + '</div>';
    
    let setScreen = document.getElementById('settlementScreen');
    setScreen.style.display = 'flex';
    setTimeout(() => setScreen.style.opacity = '1', 50);
}}

function rgKeyDown(e) {{
    if(!rgActive) return;
    let code = e.code.replace('Key', '');
    let idx = laneKeys.indexOf(code);
    if (idx !== -1) {{
        if(!keyState[idx]) handleHit(idx);
        keyState[idx] = true;
    }}
}}
function rgKeyUp(e) {{
    let code = e.code.replace('Key', '');
    let idx = laneKeys.indexOf(code);
    if (idx !== -1) keyState[idx] = false;
}}

function handleHit(lane) {{
    let currTime = audio.currentTime - (STATE.rhythmOffset / 1000.0);
    let target = null;
    let minDiff = 999;

    for(let i=0; i<BEATMAP.length; i++) {{
        let n = BEATMAP[i];
        if (n.l === lane && !n.hit) {{
            let diff = Math.abs(n.t - currTime);
            if (diff < 0.2 && diff < minDiff) {{ minDiff = diff; target = n; }}
        }}
    }}

    if (target) {{
        target.hit = true;
        let diff = Math.abs(target.t - currTime);

        playHitSound(); // 播放“咚”

        // 在 target.hit = true; 下面替换判定部分
        if (diff < 0.06) {{ rgStats.perf++; rgStats.combo++; spawnFloatText("PERFECT", "#86efac"); }} 
        else if (diff < 0.12) {{ rgStats.great++; rgStats.combo++; spawnFloatText("GREAT", "#93c5fd"); }} 
        else {{ rgStats.miss++; rgStats.combo = 0; spawnFloatText("MISS", "#fca5a5"); }}
        
        if (rgStats.combo > rgStats.maxCombo) rgStats.maxCombo = rgStats.combo;
        
        // 实时 100万分计算
        let total = BEATMAP.length;
        rgStats.score = total > 0 ? Math.floor(((rgStats.perf + rgStats.great * 0.65) / total) * 1000000) : 0;
        
        updateRgUI(); // 刷新 UI
        hitEffects.push({{ l: lane, age: 0 }});
        for(let p=0; p<8; p++) {{
            particles.push({{
               l: lane, x: (Math.random()-0.5)*50, y: 0,
               vx: (Math.random()-0.5)*15, vy: -Math.random()*15 - 5, age: 0
            }});
        }}
    }}
}}

function spawnFloatText(txt, color) {{
    floatTexts.push({{ txt: txt, c: color, age: 0, y: rgCanvas.height * 0.55 }});
}}

function updateRgUI() {{
    let cb = document.getElementById('rgCombo');
    cb.innerText = rgStats.combo;

    let wrap = document.getElementById('comboWrap');
    wrap.classList.remove('combo-pop');
    void wrap.offsetWidth; 
    if(rgStats.combo > 0) wrap.classList.add('combo-pop');

    document.getElementById('rgScore').innerText = String(rgStats.score).padStart(6, '0');
}}

function rgLoop() {{
    if (!rgActive) return;
    rgCtx.clearRect(0, 0, rgCanvas.width, rgCanvas.height);

    let w = rgCanvas.width; let h = rgCanvas.height;
    let topY = h * 0.25; let botY = h * 0.85; 
    let topW = w * 0.35;  let botW = w * 0.85;  

    // 绘制轨道
    rgCtx.lineWidth = 2;
    for(let i=0; i<=4; i++) {{
        let tx = (w - topW)/2 + (topW/4)*i;
        let bx = (w - botW)/2 + (botW/4)*i;
        rgCtx.beginPath();
        rgCtx.moveTo(tx, topY); rgCtx.lineTo(bx, botY + 100); 
        rgCtx.strokeStyle = "rgba(255,255,255,0.15)";
        rgCtx.stroke();
    }}

    // 绘制判定线
    rgCtx.beginPath();
    rgCtx.moveTo((w - botW)/2, botY); rgCtx.lineTo((w + botW)/2, botY);
    rgCtx.strokeStyle = "rgba(147,197,253,0.8)";
    rgCtx.lineWidth = 4;
    rgCtx.shadowBlur = 15; rgCtx.shadowColor = "#93c5fd";
    rgCtx.stroke();
    rgCtx.shadowBlur = 0;

    // 绘制按键高亮和文字
    for(let i=0; i<4; i++) {{
        if(keyState[i]) {{
            let grad = rgCtx.createLinearGradient(0, topY, 0, botY);
            grad.addColorStop(0, "rgba(255,255,255,0)");
            grad.addColorStop(1, laneColors[i] + "88"); 
            rgCtx.fillStyle = grad; 

            rgCtx.beginPath();
            rgCtx.moveTo((w - topW)/2 + (topW/4)*i, topY);
            rgCtx.lineTo((w - topW)/2 + (topW/4)*(i+1), topY);
            rgCtx.lineTo((w - botW)/2 + (botW/4)*(i+1), botY+100);
            rgCtx.lineTo((w - botW)/2 + (botW/4)*i, botY+100);
            rgCtx.fill();
        }}
        rgCtx.fillStyle = "rgba(255,255,255,0.5)";
        rgCtx.font = "bold 20px Arial";
        let bx = (w - botW)/2 + (botW/4)*(i + 0.5);
        // 按键文字
        rgCtx.fillText(laneKeys[i], bx - 8, botY + 40);
    }}

    let currTime = audio.currentTime - (STATE.rhythmOffset / 1000.0);
    let approachTime = 2.0 / STATE.rhythmSpeed; 

    // 绘制音符
    for(let i=0; i<BEATMAP.length; i++) {{
        let n = BEATMAP[i];
        if (n.hit) continue;

        let timeDiff = n.t - currTime;
        if (timeDiff < -0.15) {{
            n.hit = true;
            rgStats.miss++; rgStats.combo = 0;
            spawnFloatText("MISS", "#fca5a5");
            updateRgUI();
            continue;
        }}

        if (timeDiff > 0 && timeDiff <= approachTime) {{
            let progress = 1.0 - (timeDiff / approachTime);
            let easeProg = progress * progress; 

            let cy = topY + (botY - topY) * easeProg;
            let currentW = topW + (botW - topW) * easeProg;
            let cx = (w - currentW)/2 + (currentW/4)*(n.l + 0.5);

            // 音符加厚
            let nSize = 25 + 50 * easeProg;
            let noteW = (currentW/4) * 0.8;

            rgCtx.fillStyle = laneColors[n.l];
            rgCtx.shadowBlur = 15; rgCtx.shadowColor = laneColors[n.l];
            rgCtx.beginPath();
            rgCtx.roundRect(cx - noteW/2, cy - nSize/2, noteW, nSize, nSize/2);
            rgCtx.fill();
            rgCtx.shadowBlur = 0;
        }}
    }}

    // 绘制打击反馈 (Shockwaves)
    for(let i=hitEffects.length-1; i>=0; i--) {{
        let eff = hitEffects[i];
        eff.age += 0.08;
        if(eff.age > 1) {{ hitEffects.splice(i, 1); continue; }}

        let cx = (w - botW)/2 + (botW/4)*(eff.l + 0.5);

        rgCtx.beginPath();
        rgCtx.ellipse(cx, botY, 80 * eff.age, 25 * eff.age, 0, 0, Math.PI*2);
        rgCtx.strokeStyle = laneColors[eff.l] + Math.floor((1-eff.age)*255).toString(16).padStart(2,'0');
        rgCtx.lineWidth = 8 * (1-eff.age);
        rgCtx.stroke();

        rgCtx.beginPath();
        rgCtx.ellipse(cx, botY, 40, 15, 0, 0, Math.PI*2);
        rgCtx.fillStyle = laneColors[eff.l] + Math.floor((1-eff.age)*255).toString(16).padStart(2,'0');
        rgCtx.fill();
    }}

    // 绘制粒子 (Particles)
    for(let i=particles.length-1; i>=0; i--) {{
        let p = particles[i];
        p.age += 0.04;
        p.x += p.vx;
        p.y += p.vy;
        if(p.age > 1) {{ particles.splice(i,1); continue; }}

        let cx = (w - botW)/2 + (botW/4)*(p.l + 0.5) + p.x;
        let cy = botY + p.y;

        rgCtx.fillStyle = laneColors[p.l] + Math.floor((1-p.age)*255).toString(16).padStart(2,'0');
        rgCtx.shadowBlur = 10; rgCtx.shadowColor = laneColors[p.l];
        rgCtx.beginPath();
        rgCtx.arc(cx, cy, 6*(1-p.age), 0, Math.PI*2);
        rgCtx.fill();
        rgCtx.shadowBlur = 0;
    }}

    // 绘制飘字 (Floating Text)
    rgCtx.textAlign = "center";
    rgCtx.font = "900 48px 'Segoe UI', Arial";
    for(let i=floatTexts.length-1; i>=0; i--) {{
        let ft = floatTexts[i];
        ft.age += 0.03;
        if(ft.age > 1) {{ floatTexts.splice(i, 1); continue; }}

        let scale = ft.age < 0.15 ? 1.0 + ft.age*3 : 1.45 - (ft.age-0.15)*0.5;
        let alpha = Math.floor((1-ft.age)*255).toString(16).padStart(2,'0');

        rgCtx.save();
        rgCtx.translate(w/2, ft.y - ft.age*40);
        rgCtx.scale(scale, scale);
        rgCtx.fillStyle = ft.c + alpha;
        rgCtx.shadowBlur = 20; rgCtx.shadowColor = ft.c;
        rgCtx.fillText(ft.txt, 0, 0);
        rgCtx.restore();
    }}

    requestAnimationFrame(rgLoop);
}}

(function init() {{
  if (SELECTED_INST) STATE.instrument = SELECTED_INST;
  if (STATE.step === 3) goTo(3);
  else if (HAS_AUDIO) goTo(4);
  else goTo(1);
}})();
</script>
</body>
</html>"""
