"""TradingAgents 多智能体二次打分模块

在 N 字战法扫描出 Top N 候选后，用 TradingAgents 的多智能体框架
对每只股票做深度分析，输出：
- confidence: 投资建议置信度 0~1
- action: 买入/持有/卖出
- bull_bear_divergence: 多空分歧度 0~1（分歧大→不确定性高）
- target_price: 目标价
- reasoning: 决策理由摘要

用法:
    from screener.tradingagents_scorer import score_with_tradingagents
    result = score_with_tradingagents('600036', '招商银行', '2026-05-31')
"""

import logging
import os
import sys
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

# TradingAgents 路径
_TA_PATH = os.path.expanduser("~/TradingAgents-CN")
if _TA_PATH not in sys.path:
    sys.path.insert(0, _TA_PATH)

# 修补 LangGraph 0.6.x + TradingAgents 的 forward reference bug
# AgentState.messages: ForwardRef('Annotated[list[AnyMessage], add_messages]') 无法解析
try:
    import tradingagents.agents.utils.agent_states as _agent_states
    from langchain_core.messages import AnyMessage
    from langgraph.graph import add_messages
    _agent_states.AnyMessage = AnyMessage
    _agent_states.add_messages = add_messages
except Exception:
    pass

_graph_instance = None


def _get_graph():
    """延迟初始化 TradingAgentsGraph（单例，避免重复加载）"""
    global _graph_instance
    if _graph_instance is not None:
        return _graph_instance

    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "deepseek"
    config["deep_think_llm"] = "deepseek-chat"
    config["quick_think_llm"] = "deepseek-chat"
    config["backend_url"] = "https://api.deepseek.com"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["online_tools"] = False
    config["online_news"] = False

    _graph_instance = TradingAgentsGraph(debug=False, config=config)
    logger.info("TradingAgentsGraph 初始化完成")
    return _graph_instance


def _compute_bull_bear_divergence(final_state: dict) -> float:
    """从辩论历史计算多空分歧度 0~1。

    判断方法: 辩论中多头/空头论据长度差异 → 分歧越小=意见越一致。
    0=完全一致, 1=极度分歧。
    """
    try:
        debate = final_state.get("investment_debate_state", {})
        history = debate.get("history", "")
        if isinstance(history, str) and len(history) > 100:
            # 计算 history 中 bull/bear 交替次数作为分歧代理
            import re
            bull_mentions = len(re.findall(r'(bull|多头|买入|看涨|看多)', history, re.I))
            bear_mentions = len(re.findall(r'(bear|空头|卖出|看跌|看空)', history, re.I))
            total = bull_mentions + bear_mentions
            if total > 0:
                # 分歧度 = 少数派占比 × 2 (0~1 scale)
                minority_ratio = min(bull_mentions, bear_mentions) / total
                return round(minority_ratio * 2, 2)
        return 0.5  # 默认中等分歧
    except Exception:
        return 0.5


def score_with_tradingagents(
    code: str,
    name: str = "",
    analysis_date: str = None,
    timeout: int = 180,
) -> dict:
    """用 TradingAgents 多智能体框架对单只股票做深度分析。

    Args:
        code: 6位股票代码
        name: 股票名称（可选）
        analysis_date: 分析日期 YYYY-MM-DD（默认今天）
        timeout: 超时秒数

    Returns:
        dict with keys:
            confidence, action, bull_bear_divergence,
            target_price, risk_score, reasoning, success, error
    """
    if analysis_date is None:
        from datetime import date
        analysis_date = date.today().strftime("%Y-%m-%d")

    result = {
        "confidence": 0.5,
        "action": "持有",
        "bull_bear_divergence": 0.5,
        "target_price": None,
        "risk_score": 0.5,
        "reasoning": "",
        "success": False,
        "error": "",
    }

    try:
        graph = _get_graph()
        start = time.time()

        # 用线程超时控制
        import threading

        output = {"final_state": None, "decision": None, "error": None}

        def _run():
            try:
                fs, dec = graph.propagate(code, analysis_date)
                output["final_state"] = fs
                output["decision"] = dec
            except Exception as e:
                output["error"] = str(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        elapsed = time.time() - start

        if thread.is_alive():
            result["error"] = f"超时 ({timeout}s)"
            logger.warning(f"TradingAgents {code} 超时 ({timeout}s)")
            return result

        if output["error"]:
            result["error"] = output["error"]
            logger.warning(f"TradingAgents {code} 错误: {output['error']}")
            return result

        decision = output["decision"]
        final_state = output["final_state"]

        if decision is None:
            result["error"] = "无决策输出"
            return result

        result["success"] = True
        result["confidence"] = float(decision.get("confidence", 0.5))
        result["action"] = str(decision.get("action", "持有"))
        result["target_price"] = decision.get("target_price")
        result["risk_score"] = float(decision.get("risk_score", 0.5))
        result["reasoning"] = str(decision.get("reasoning", ""))[:300]

        if final_state is not None:
            result["bull_bear_divergence"] = _compute_bull_bear_divergence(final_state)

        logger.info(
            f"TradingAgents {code}({name}): "
            f"action={result['action']} conf={result['confidence']:.2f} "
            f"divergence={result['bull_bear_divergence']:.2f} "
            f"({elapsed:.0f}s)"
        )

        return result

    except ImportError as e:
        result["error"] = f"TradingAgents 未安装: {e}"
        logger.error(result["error"])
        return result
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"TradingAgents {code} 异常: {e}")
        return result


def score_top_candidates(
    signals: list,
    top_n: int = 5,
    min_strength: int = 70,
) -> list:
    """对 Top N 个候选用 TradingAgents 二次打分，修改 signals 的 strength。

    规则:
    - buy + conf > 0.7 → strength + 10
    - buy + conf > 0.5 → strength + 5
    - sell → strength - 8
    - 分歧度惩罚: strength -= int(divergence * 12)

    Args:
        signals: NSignal 列表（按 strength 降序）
        top_n: 最多分析几只
        min_strength: 最低强度阈值

    Returns:
        修改后的 signals（原地修改）
    """
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")

    count = 0
    for sig in signals:
        if count >= top_n:
            break
        if sig.strength < min_strength:
            continue

        logger.info(f"TradingAgents 分析 #{count + 1}: {sig.code} {sig.name} (强度={sig.strength})")
        result = score_with_tradingagents(sig.code, sig.name, today)

        sig.tradingagents_confidence = result["confidence"]
        sig.tradingagents_action = result["action"]
        sig.tradingagents_divergence = result["bull_bear_divergence"]

        if result["success"]:
            # 行动方向调整
            action = result["action"]
            conf = result["confidence"]
            if action == "买入" and conf > 0.7:
                sig.strength += 10
            elif action == "买入" and conf > 0.5:
                sig.strength += 5
            elif action == "卖出":
                sig.strength -= 8

            # 分歧度惩罚
            divergence = result["bull_bear_divergence"]
            sig.strength -= int(divergence * 12)

        count += 1

    # 重新排序
    signals.sort(key=lambda s: s.strength, reverse=True)
    return signals
