# -*- coding: utf-8 -*-
"""
睿聘智模 v3.1  ·  config.py
所有配置项均在此文件维护，engine/app 不含任何硬编码参数。
"""

# ── LLM API ──────────────────────────────────────────────────────
API_BASE    = "https://api.rcouyi.com/v1"
API_KEY     = "sk-lw29XaxfVg9MPVza2678Bd193b7248EeB6BbD65529459546"
MODEL_NAME  = "gpt-4o"
LLM_TIMEOUT = 120   # 秒

# ── Web 服务 ──────────────────────────────────────────────────────
SERVER_HOST  = "0.0.0.0"
SERVER_PORT  = 5000
DEBUG_MODE   = False
SECRET_KEY   = "ruipinzhimo-secret-key-2024"
MAX_UPLOAD_MB = 20

# ── 默认面试参数（用户可在 UI 设置页覆盖，每 session 独立）────────
DEFAULT_MAX_FOLLOWUP_ROUNDS      = 2     # 每题最多追问轮数 (1-4)
DEFAULT_DEPTH_SCORE_THRESHOLD    = 65    # 深度评分触发追问阈值 (0-100)
DEFAULT_MIN_STAR_ELEMENTS        = 3     # 最少STAR要素 (1-4)
DEFAULT_INTERVIEW_DURATION_MINUTES = 60  # 面试时长(分钟): 30|45|60|90|120
DEFAULT_INTERVIEW_MODE           = "standard"

# ── 面试官人设 ─────────────────────────────────────────────────────
INTERVIEWER_NAME      = "HR"
INTERVIEWER_TITLE     = "资深HR总监兼结构化面试专家"
INTERVIEWER_YEARS_EXP = 18
INTERVIEWER_STYLE     = "问题精准、洞察深刻、有温度"

# ── 评分等级（百分制）────────────────────────────────────────────
SCORE_GRADES = [
    (85, 100, "A", "卓越"),
    (70,  84, "B", "优秀"),
    (55,  69, "C", "合格"),
    ( 0,  54, "D", "不达标"),
]

# ── 面试模式 ──────────────────────────────────────────────────────
INTERVIEW_MODES = {
    "standard": {
        "name":        "标准专业",
        "description": "专业平衡，循序渐进，充分尊重候选人",
        "style_prompt": (
            "语气专业温和，问题循序渐进，给候选人充分表达空间。"
            "追问精准有力但不施压，从回答裂缝处切入。"
            "认可要具体真实，禁止'非常好！''您说得对！'等空洞夸赞。"
        ),
        "followup_aggression": "moderate",
    },
    "casual": {
        "name":        "轻松交流",
        "description": "友好开放，降低紧张感，适合创意/创业岗位",
        "style_prompt": (
            "语气轻松友好，像与老朋友深度聊天，偶尔可带点幽默。"
            "用开放引导代替封闭追问，让候选人自然展开。"
            "即使回答不够深度，也温和引导而非直接追问。"
        ),
        "followup_aggression": "low",
    },
    "stress": {
        "name":        "压力测试",
        "description": "挑战性追问，直接质疑，测试逆境思维清晰度",
        "style_prompt": (
            "采用压力面试风格，直接挑战候选人回答中的不足或假设。"
            "可提出反驳：'如果你的方法失败了怎么办？''这个数据你能保证准确吗？'"
            "目的是测试压力下的思维清晰度，保持专业尊重，而非让其难堪。"
        ),
        "followup_aggression": "high",
    },
    "behavioral": {
        "name":        "行为事件 BEI",
        "description": "严格STAR结构，聚焦过去行为预测未来",
        "style_prompt": (
            "严格BEI框架，所有问题聚焦具体行为事件。"
            "若候选人给出假设性或未来式回答，立即温和打断：'能告诉我您过去真实经历过的案例吗？'"
            "必须追问到具体人物、时间、量化数据，不接受泛化回答。"
        ),
        "followup_aggression": "moderate",
    },
}

# ── 五维权重参考（仅供LLM参考，LLM可自由根据岗位特性调整）────────
# 维度固定顺序：D1专业技能 D2问题解决 D3沟通协作 D4学习成长 D5价值观
DIMENSION_WEIGHT_REFERENCE = {
    "技术研发": [0.35, 0.27, 0.15, 0.14, 0.09],
    "管理领导": [0.22, 0.23, 0.30, 0.13, 0.12],
    "销售市场": [0.18, 0.20, 0.32, 0.15, 0.15],
    "产品设计": [0.25, 0.30, 0.22, 0.13, 0.10],
    "运营职能": [0.25, 0.22, 0.25, 0.18, 0.10],
    "校招初级": [0.22, 0.23, 0.18, 0.25, 0.12],
    "通用":     [0.28, 0.24, 0.22, 0.15, 0.11],
}