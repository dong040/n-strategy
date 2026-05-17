"""从飞书消息中提取 N 字战法规则

输入：群聊消息文本列表
输出：提取到的战法参数、案例、关键词
"""

import re
import logging
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRule:
    """从消息中提取的一条战法规则"""
    category: str           # 分类: entry / stop_loss / take_profit / filter / pattern
    keyword: str            # 触发关键词
    raw_text: str           # 原始消息片段
    confidence: float       # 置信度 0~1
    param_name: str = ""    # 对应参数名
    param_value: str = ""   # 参数值


@dataclass
class ExtractedCase:
    """从消息中提取的一个实战案例"""
    code: str = ""               # 股票代码
    name: str = ""               # 股票名称
    date: str = ""               # 提及日期
    direction: str = "long"      # long / short
    result: str = ""             # 结果描述
    raw_text: str = ""           # 原始消息
    tags: list[str] = field(default_factory=list)


# ====== 关键词词典 ======

# 买入/入场规则
ENTRY_KEYWORDS = {
    "突破前高": {
        "params": {"entry_mode": "breakout"},
        "keywords": ["突破前高", "过前高", "创新高", "突破高点", "放量突破", "带量突破"],
        "weight": 0.9,
    },
    "回调买入": {
        "params": {"entry_mode": "retrace_buy"},
        "keywords": ["回调买入", "回踩买入", "回调低吸", "回踩确认", "缩量回调买"],
        "weight": 0.85,
    },
    "分时确认": {
        "params": {"entry_confirm": "intraday"},
        "keywords": ["分时确认", "分时走好", "盘中确认", "尾盘确认"],
        "weight": 0.7,
    },
}

# 止损规则
STOP_LOSS_KEYWORDS = {
    "回调低点止损": {
        "params": {"stop_mode": "retrace_low"},
        "keywords": ["破回调低点", "跌破回调低点", "破低点走", "止损放在回调低点", "回调低点下方"],
        "weight": 0.9,
    },
    "固定比例止损": {
        "params": {"stop_mode": "fixed_pct"},
        "keywords": ["止损(\d+)[个点%％]", "亏(\d+)[个点%％]走", "止损(\d+)[个点%％]"],
        "weight": 0.8,
    },
    "均线止损": {
        "params": {"stop_mode": "ma"},
        "keywords": ["破(\d+)日线", "跌破(\d+)均线", "(\d+)日均线止损"],
        "weight": 0.75,
    },
    "ATR止损": {
        "params": {"stop_mode": "atr"},
        "keywords": ["ATR", "波动率止损", "(\d+)倍ATR"],
        "weight": 0.7,
    },
}

# 止盈规则
TAKE_PROFIT_KEYWORDS = {
    "等幅测距": {
        "params": {"take_profit_mode": "equal"},
        "keywords": ["等幅", "量度涨幅", "等长", "等距", "翻箱体", "一比一"],
        "weight": 0.85,
    },
    "前高止盈": {
        "params": {"take_profit_mode": "prev_high"},
        "keywords": ["前高止盈", "到前高减仓", "前高附近", "前期高点"],
        "weight": 0.8,
    },
    "移动止盈": {
        "params": {"take_profit_mode": "trailing"},
        "keywords": ["移动止盈", "移动止损", "跟踪止盈", "破均线止盈"],
        "weight": 0.75,
    },
}

# 筛选条件
FILTER_KEYWORDS = {
    "市值": {
        "params": {},
        "keywords": ["市值(\d+)亿", "市值小于(\d+)亿", "市值大于(\d+)亿", "盘子[大小]", "小盘"],
        "weight": 0.7,
    },
    "放量要求": {
        "params": {},
        "keywords": ["放量", "量比[大于>](\d+[\.\d]*)", "成交量放大", "换手[率]?(\d+[\.\d]*)%"],
        "weight": 0.8,
    },
    "涨幅筛选": {
        "params": {},
        "keywords": ["涨幅[大于>](\d+[\.\d]*)[%％]", "涨(\d+[\.\d]*)[%％]以上", "首板"],
        "weight": 0.75,
    },
}

# 形态/模式关键词
PATTERN_KEYWORDS = {
    "N字形态": {
        "params": {},
        "keywords": ["N字", "N型", "N形", "二次拉升", "二波", "第二波", "再起一波"],
        "weight": 0.9,
    },
    "旗形/楔形": {
        "params": {},
        "keywords": ["旗形", "楔形", "三角形整理", "平台整理", "箱体震荡", "横盘整理"],
        "weight": 0.7,
    },
}

ALL_RULE_GROUPS = {
    "entry": ENTRY_KEYWORDS,
    "stop_loss": STOP_LOSS_KEYWORDS,
    "take_profit": TAKE_PROFIT_KEYWORDS,
    "filter": FILTER_KEYWORDS,
    "pattern": PATTERN_KEYWORDS,
}


def extract_rules(messages: list[dict]) -> list[ExtractedRule]:
    """从消息列表中提取战法规则

    Args:
        messages: [{"content": "...", ...}, ...]

    Returns:
        提取到的规则列表
    """
    rules = []

    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue

        for category, group in ALL_RULE_GROUPS.items():
            for rule_name, rule_info in group.items():
                for kw in rule_info["keywords"]:
                    pattern = re.compile(kw, re.IGNORECASE)
                    matches = pattern.findall(content)
                    if matches:
                        rules.append(ExtractedRule(
                            category=category,
                            keyword=kw,
                            raw_text=content[:200],
                            confidence=rule_info["weight"],
                            param_name=rule_name,
                            param_value=str(matches[0]) if matches else "",
                        ))

    return rules


def extract_numeric_params(content: str) -> dict:
    """从消息中提取数值参数

    Returns:
        {"min_rise_1st": 0.05, "stop_loss_pct": 0.03, ...}
    """
    params = {}

    # 涨幅提取
    m = re.search(r"(?:涨幅|涨)[^\d]*(\d+[\.\d]*)[%％]", content)
    if m:
        params["min_rise_1st"] = float(m.group(1)) / 100

    # 回调比例提取
    for name, pattern in [
        ("retrace_max", r"(?:回调|回踩)[^\d]*(\d+[\.\d]*)[%％]"),
        ("retrace_max", r"(?:回调|回踩)[^\d]*(0?\.\d+)"),
        ("retrace_max", r"(\d+)分位"),
    ]:
        m = re.search(pattern, content)
        if m:
            val = float(m.group(1))
            if val >= 1:
                val = val / 100
            params[name] = val
            break

    # 止损百分比
    m = re.search(r"止损[^\d]*(\d+[\.\d]*)[%％]", content)
    if m:
        params["stop_loss_pct"] = float(m.group(1)) / 100

    # 量比
    m = re.search(r"量比[>大于]?\s*(\d+[\.\d]*)", content)
    if m:
        params["vol_ratio_breakout"] = float(m.group(1))

    # 换手率
    m = re.search(r"换手[率]?\s*[>大于]?\s*(\d+[\.\d]*)[%％]", content)
    if m:
        params["min_turnover"] = float(m.group(1))

    # 回调天数
    m = re.search(r"(?:回调|调整)[^\d]*(\d+)[个天]", content)
    if m:
        days = int(m.group(1))
        params["retrace_days_max"] = days

    return params


def extract_stock_cases(messages: list[dict]) -> list[ExtractedCase]:
    """从消息中提取实战案例（提到的股票代码+分析）"""
    cases = []
    stock_pattern = re.compile(r'(?:(?:60[0-4]\d{3})|(?:688\d{3})|(?:00[0-2]\d{3})|(?:30[0-4]\d{3})|(?:83[0-2]\d{3}))')

    for msg in messages:
        content = msg.get("content", "")
        codes = stock_pattern.findall(content)
        if not codes:
            continue

        for code in set(codes):
            case = ExtractedCase(
                code=code,
                date=msg.get("created_at", "")[:10],
                raw_text=content[:300],
                tags=_extract_tags(content),
            )
            cases.append(case)

    return cases


def _extract_tags(content: str) -> list[str]:
    """从消息中提取题材/概念标签"""
    tags = []
    tag_patterns = [
        r"(\S+概念)", r"(\S+板块)", r"(\S+赛道)",
        r"龙头", r"妖股", r"首板", r"连板", r"反包",
        r"机器人", r"AI", r"新能源", r"芯片", r"半导体",
        r"医药", r"消费", r"军工", r"光伏", r"锂电",
    ]
    for p in tag_patterns:
        m = re.findall(p, content)
        tags.extend(m)
    return list(set(tags))


def summarize_rules(rules: list[ExtractedRule]) -> dict:
    """汇总提取到的规则，按类别和置信度排序"""
    by_category = {}
    for r in rules:
        by_category.setdefault(r.category, []).append(r)

    summary = {}
    for cat, items in by_category.items():
        # 取最高置信度的规则
        items.sort(key=lambda x: x.confidence, reverse=True)
        keywords_counter = Counter(r.keyword for r in items)
        summary[cat] = {
            "count": len(items),
            "top_rules": items[:5],
            "top_keywords": keywords_counter.most_common(5),
        }
    return summary
