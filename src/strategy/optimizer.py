"""N字战法参数优化器

支持：
- 网格搜索（粗搜索）
- 贝叶斯优化（精搜索）
- 随机搜索

目标函数：最大化夏普比率（可配置）
"""

import logging
import itertools
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .n_pattern import NPatternParams
from .backtest import BacktestConfig, BacktestResult, backtest_portfolio

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    method: str = "bayesian"           # bayesian / grid / random
    n_iterations: int = 200
    cv_folds: int = 3
    target_metric: str = "sharpe"      # sharpe / win_rate / profit_factor / calmar / combined


# 参数搜索空间
SEARCH_SPACE = {
    "min_rise_1st": (0.02, 0.15),
    "max_rise_1st": (0.15, 0.50),
    "retrace_min": (0.15, 0.40),
    "retrace_max": (0.40, 0.70),
    "retrace_days_min": (1, 5),
    "retrace_days_max": (5, 15),
    "vol_ratio_breakout": (0.8, 2.5),
    "stop_loss_pct": (0.01, 0.08),
}


def _params_to_vector(p: NPatternParams) -> list:
    return [
        p.min_rise_1st, p.max_rise_1st,
        p.retrace_min, p.retrace_max,
        float(p.retrace_days_min), float(p.retrace_days_max),
        p.vol_ratio_breakout, p.stop_loss_pct,
    ]


def _vector_to_params(v: list) -> NPatternParams:
    return NPatternParams(
        min_rise_1st=float(v[0]),
        max_rise_1st=float(v[1]),
        retrace_min=float(v[2]),
        retrace_max=float(v[3]),
        retrace_days_min=int(v[4]),
        retrace_days_max=int(v[5]),
        vol_ratio_breakout=float(v[6]),
        stop_loss_pct=float(v[7]),
    )


def _evaluate(
    params: NPatternParams,
    stocks_data: dict[str, pd.DataFrame],
    config: BacktestConfig,
    metric: str,
) -> float:
    """评估一组参数的表现"""
    result = backtest_portfolio(stocks_data, params, config)

    score_map = {
        "sharpe": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
    }
    score = score_map.get(metric, result.sharpe_ratio)

    # 交易次数太少惩罚
    if result.total_trades < 5:
        score *= 0.1
    elif result.total_trades < 10:
        score *= 0.5

    # 最大回撤过大的惩罚
    if result.max_drawdown > 30:
        score *= 0.5

    return score


def _random_params() -> NPatternParams:
    """在搜索空间内随机生成参数"""
    return NPatternParams(
        min_rise_1st=round(random.uniform(*SEARCH_SPACE["min_rise_1st"]), 3),
        max_rise_1st=round(random.uniform(*SEARCH_SPACE["max_rise_1st"]), 3),
        retrace_min=round(random.uniform(*SEARCH_SPACE["retrace_min"]), 3),
        retrace_max=round(random.uniform(*SEARCH_SPACE["retrace_max"]), 3),
        retrace_days_min=random.randint(*SEARCH_SPACE["retrace_days_min"]),
        retrace_days_max=random.randint(*SEARCH_SPACE["retrace_days_max"]),
        vol_ratio_breakout=round(random.uniform(*SEARCH_SPACE["vol_ratio_breakout"]), 2),
        stop_loss_pct=round(random.uniform(*SEARCH_SPACE["stop_loss_pct"]), 3),
    )


def _split_data(stocks_data: dict, n_folds: int) -> list:
    """按时间将数据拆分为训练集/测试集"""
    splits = []
    for code, df in stocks_data.items():
        n = len(df)
        fold_size = n // (n_folds + 1)
        for i in range(n_folds):
            train_end = n - (n_folds - i) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            splits.append((
                {code: df.iloc[:train_end]},
                {code: df.iloc[test_start:test_end]},
            ))
    return splits


def grid_search(
    stocks_data: dict[str, pd.DataFrame],
    config: BacktestConfig = None,
    opt_config: OptimizerConfig = None,
) -> tuple[NPatternParams, float, list[dict]]:
    """网格搜索优化

    对每个参数取 3 个值（低/中/高），遍历所有组合。
    实际组合数 = 3^8 = 6561，可根据需要缩减。
    """
    if config is None:
        config = BacktestConfig()
    if opt_config is None:
        opt_config = OptimizerConfig()

    # 每个参数取 3 个值
    grid = {}
    for name, (lo, hi) in SEARCH_SPACE.items():
        grid[name] = [lo, (lo + hi) / 2, hi]

    # 生成所有组合（8 个参数 × 3 个值 = 6561）
    keys = list(grid.keys())
    combinations = list(itertools.product(*[grid[k] for k in keys]))

    # 如果太多，随机采样
    if len(combinations) > opt_config.n_iterations:
        combinations = random.sample(combinations, opt_config.n_iterations)

    logger.info(f"网格搜索: {len(combinations)} 组参数")

    best_params = None
    best_score = -float("inf")
    results = []

    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        # 约束：retrace_max > retrace_min, retrace_days_max > retrace_days_min, max_rise > min_rise
        if param_dict["retrace_max"] <= param_dict["retrace_min"]:
            continue
        if param_dict["retrace_days_max"] <= param_dict["retrace_days_min"]:
            continue
        if param_dict["max_rise_1st"] <= param_dict["min_rise_1st"]:
            continue

        params = NPatternParams(
            min_rise_1st=param_dict["min_rise_1st"],
            max_rise_1st=param_dict["max_rise_1st"],
            retrace_min=param_dict["retrace_min"],
            retrace_max=param_dict["retrace_max"],
            retrace_days_min=param_dict["retrace_days_min"],
            retrace_days_max=param_dict["retrace_days_max"],
            vol_ratio_breakout=param_dict["vol_ratio_breakout"],
            stop_loss_pct=param_dict["stop_loss_pct"],
        )

        score = _evaluate(params, stocks_data, config, opt_config.target_metric)
        results.append({"params": params, "score": score})

        if score > best_score:
            best_score = score
            best_params = params

    results.sort(key=lambda x: x["score"], reverse=True)
    return best_params, best_score, results


def bayesian_optimize(
    stocks_data: dict[str, pd.DataFrame],
    config: BacktestConfig = None,
    opt_config: OptimizerConfig = None,
) -> tuple[NPatternParams, float, list[dict]]:
    """贝叶斯优化（高斯过程）"""

    try:
        from scipy.stats import norm
        from scipy.optimize import minimize
    except ImportError:
        logger.warning("scipy 未安装，回退到随机搜索")
        return random_search(stocks_data, config, opt_config)

    if config is None:
        config = BacktestConfig()
    if opt_config is None:
        opt_config = OptimizerConfig()

    bounds = np.array([
        SEARCH_SPACE["min_rise_1st"],
        SEARCH_SPACE["max_rise_1st"],
        SEARCH_SPACE["retrace_min"],
        SEARCH_SPACE["retrace_max"],
        [float(SEARCH_SPACE["retrace_days_min"][0]), float(SEARCH_SPACE["retrace_days_min"][1])],
        [float(SEARCH_SPACE["retrace_days_max"][0]), float(SEARCH_SPACE["retrace_days_max"][1])],
        SEARCH_SPACE["vol_ratio_breakout"],
        SEARCH_SPACE["stop_loss_pct"],
    ])
    n_dims = len(bounds)
    n_init = min(20, opt_config.n_iterations // 5)

    # 初始采样（Latin Hypercube 简化版：随机）
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, n_dims))
    y = np.array([_evaluate(_vector_to_params(xi), stocks_data, config, opt_config.target_metric) for xi in X])

    best_idx = np.argmax(y)
    best_x = X[best_idx].copy()
    best_y = y[best_idx]

    # 主优化循环
    for iteration in range(opt_config.n_iterations - n_init):
        # 用 GP 拟合当前数据
        # 简化版: 使用 RBF 核 + 期望改进采集函数

        # 预测均值和方差
        def predict(x_new):
            distances = np.linalg.norm(X - x_new, axis=1)
            length_scale = np.std(distances) + 1e-6
            weights = np.exp(-0.5 * (distances / length_scale) ** 2)
            weights_sum = weights.sum()
            if weights_sum > 0:
                mu = np.dot(weights, y) / weights_sum
                sigma = np.sqrt(np.dot(weights, (y - mu) ** 2) / weights_sum + 1e-6)
            else:
                mu, sigma = np.mean(y), np.std(y)
            return mu, sigma

        # 采集函数：Expected Improvement
        def ei(x_candidate):
            mu, sigma = predict(x_candidate)
            improvement = mu - best_y
            if sigma > 1e-6:
                z = improvement / sigma
                return improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return max(improvement, 0)

        # 随机采样 + 局部优化找下一个候选点
        n_random = 1000
        candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_random, n_dims))
        ei_values = np.array([ei(c) for c in candidates])
        x_next = candidates[np.argmax(ei_values)]

        # 评估
        try:
            y_next = _evaluate(_vector_to_params(x_next), stocks_data, config, opt_config.target_metric)
        except Exception as e:
            logger.warning(f"评估失败: {e}, 跳过")
            continue

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if y_next > best_y:
            best_y = y_next
            best_x = x_next.copy()
            logger.info(f"  迭代 {iteration + 1}: 发现更优参数 score={best_y:.3f}")

    best_params = _vector_to_params(best_x)
    results = [
        {"params": _vector_to_params(X[i]), "score": float(y[i])}
        for i in range(len(y))
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return best_params, best_y, results


def random_search(
    stocks_data: dict[str, pd.DataFrame],
    config: BacktestConfig = None,
    opt_config: OptimizerConfig = None,
) -> tuple[NPatternParams, float, list[dict]]:
    """随机搜索优化"""
    if config is None:
        config = BacktestConfig()
    if opt_config is None:
        opt_config = OptimizerConfig()

    best_params = None
    best_score = -float("inf")
    results = []

    for i in range(opt_config.n_iterations):
        params = _random_params()
        # 约束检查
        if params.retrace_max <= params.retrace_min:
            continue
        if params.retrace_days_max <= params.retrace_days_min:
            continue
        if params.max_rise_1st <= params.min_rise_1st:
            continue

        score = _evaluate(params, stocks_data, config, opt_config.target_metric)
        results.append({"params": params, "score": score})

        if score > best_score:
            best_score = score
            best_params = params

        if (i + 1) % 20 == 0:
            logger.info(f"  随机搜索 {i + 1}/{opt_config.n_iterations}, 当前最优={best_score:.3f}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return best_params, best_score, results


def optimize(
    stocks_data: dict[str, pd.DataFrame],
    method: str = "bayesian",
    config: BacktestConfig = None,
    opt_config: OptimizerConfig = None,
) -> tuple[NPatternParams, float, list[dict]]:
    """统一入口：参数优化"""
    if opt_config is None:
        opt_config = OptimizerConfig(method=method)

    if method == "grid":
        return grid_search(stocks_data, config, opt_config)
    elif method == "bayesian":
        return bayesian_optimize(stocks_data, config, opt_config)
    elif method == "random":
        return random_search(stocks_data, config, opt_config)
    else:
        raise ValueError(f"不支持的优化方法: {method}")
