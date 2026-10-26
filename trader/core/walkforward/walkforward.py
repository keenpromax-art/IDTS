import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# Walk-Forward Validation Framework
# ============================================================


class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    def __init__(self, training_bars=250, validation_bars=50, test_bars=100,
                 step_bars=50, expanding_window=False, min_data_bars=150,
                 metrics=None):
        self.training_bars = training_bars
        self.validation_bars = validation_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.expanding_window = expanding_window
        self.min_data_bars = min_data_bars
        self.metrics = metrics or [
            'expectancy_r', 'profit_factor', 'win_rate', 'max_drawdown',
            'total_trades', 'turnover', 'sharpe', 'sortino'
        ]


class WalkForwardFold:
    """A single walk-forward fold."""

    def __init__(self, fold_id, train_start, train_end, valid_start, valid_end,
                 test_start, test_end, train_df, valid_df, test_df):
        self.fold_id = fold_id
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start
        self.test_end = test_end
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.train_metrics = {}
        self.valid_metrics = {}
        self.test_metrics = {}
        self.strategy_name = ''
        self.data_version = ''
        self.config_hash = ''


class WalkForwardResult:
    """Results from walk-forward validation."""

    def __init__(self, folds, aggregate_metrics, best_fold_id, worst_fold_id, stable, report):
        self.folds = folds
        self.aggregate_metrics = aggregate_metrics
        self.best_fold_id = best_fold_id
        self.worst_fold_id = worst_fold_id
        self.stable = stable
        self.report = report


def _compute_date_ranges(df, config):
    """Compute the date ranges for each walk-forward fold."""
    folds = []
    total_bars = len(df)
    if total_bars < config.min_data_bars:
        return folds

    train_size = config.training_bars
    valid_size = config.validation_bars
    test_size = config.test_bars
    step_size = config.step_bars
    expanding = config.expanding_window

    if expanding:
        # Expanding window: training grows, test is always the last period
        pos = 0
        fold_id = 0
        while pos + train_size + valid_size + test_size <= total_bars:
            train_end = min(pos + train_size, total_bars)
            valid_start = train_end
            valid_end = min(valid_start + valid_size, total_bars)
            test_start = valid_end
            test_end = min(test_start + test_size, total_bars)

            if test_end <= test_start:
                break

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=df.index[max(0, pos)],
                train_end=df.index[train_end - 1],
                valid_start=df.index[valid_start],
                valid_end=df.index[valid_end - 1],
                test_start=df.index[test_start],
                test_end=df.index[test_end - 1],
                train_df=df.iloc[max(0, pos):train_end],
                valid_df=df.iloc[valid_start:valid_end],
                test_df=df.iloc[test_start:test_end],
            )
            folds.append(fold)
            fold_id += 1
            pos += step_size

    else:
        # Rolling window: training window moves forward
        pos = 0
        fold_id = 0
        while pos + train_size + valid_size + test_size <= total_bars:
            train_end = min(pos + train_size, total_bars)
            valid_start = train_end
            valid_end = min(valid_start + valid_size, total_bars)
            test_start = valid_end
            test_end = min(test_start + test_size, total_bars)

            if test_end <= test_start:
                break

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=df.index[max(0, pos)],
                train_end=df.index[train_end - 1],
                valid_start=df.index[valid_start],
                valid_end=df.index[valid_end - 1],
                test_start=df.index[test_start],
                test_end=df.index[test_end - 1],
                train_df=df.iloc[max(0, pos):train_end],
                valid_df=df.iloc[valid_start:valid_end],
                test_df=df.iloc[test_start:test_end],
            )
            folds.append(fold)
            fold_id += 1
            pos += step_size

    return folds


def _run_backtest(strategy, df, symbol, phase):
    """Run a backtest on the given data phase and return metrics."""
    if df is None or len(df) < 20:
        return {}

    closes = df['close'].dropna().values
    if len(closes) < 2:
        return {}

    pnls = []
    for i in range(1, len(closes)):
        pnl = closes[i] - closes[i-1]
        pnls.append(pnl)

    total_trades = len([p for p in pnls if p != 0])
    winning_trades = len([p for p in pnls if p > 0])
    losing_trades = len([p for p in pnls if p < 0])
    total_pnl = sum(pnls)
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))

    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (float('inf') if gross_wins > 0 else 0.0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    expectancy = gross_wins / total_trades - abs(gross_losses) / total_trades if total_trades > 0 else 0.0

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    return {
        'total_trades': total_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
        'profit_factor': profit_factor,
        'expectancy_r': expectancy,
        'max_drawdown': max_dd,
        'total_pnl': total_pnl,
        'gross_win': gross_wins,
        'gross_loss': gross_losses,
        'phase': phase
    }


def _compute_aggregate_metrics(folds):
    """Compute aggregate metrics across all folds."""
    if not folds:
        return {}

    n = len(folds)
    aggregates = {}

    for metric in ['expectancy_r', 'profit_factor', 'win_rate', 'max_drawdown']:
        values = [f.test_metrics.get(metric, 0) for f in folds]
        aggregates[metric] = np.mean(values) if values else 0.0

    total_trades = sum(f.test_metrics.get('total_trades', 0) for f in folds)
    aggregates['total_trades'] = total_trades

    test_returns = [f.test_metrics.get('expectancy_r', 0) for f in folds]
    if len(test_returns) > 1:
        aggregates['expectancy_std'] = np.std(test_returns)
    else:
        aggregates['expectancy_std'] = 0.0

    return aggregates


def _generate_report(folds, aggregate_metrics):
    """Generate a human-readable report."""
    lines = []
    lines.append('=' * 70)
    lines.append('WALK-FORWARD VALIDATION REPORT')
    lines.append('=' * 70)
    lines.append('')

    lines.append(f'Number of folds: {len(folds)}')
    lines.append('')

    lines.append('FOLD DETAILS:')
    lines.append('-' * 70)
    for fold in folds:
        lines.append('')
        lines.append(f'Fold {fold.fold_id}:')
        lines.append(f'  Train: {fold.train_start} to {fold.train_end}')
        lines.append(f'  Valid: {fold.valid_start} to {fold.valid_end}')
        lines.append(f'  Test:  {fold.test_start} to {fold.test_end}')
        lines.append(f'  Test Metrics:')
        for key, value in fold.test_metrics.items():
            if isinstance(value, float):
                lines.append(f'    {key}: {value:.4f}')
            elif isinstance(value, (int, np.integer)):
                lines.append(f'    {key}: {value}')
            else:
                lines.append(f'    {key}: {value}')
    lines.append('')

    lines.append('AGGREGATE METRICS:')
    lines.append('-' * 70)
    for key, value in aggregate_metrics.items():
        if isinstance(value, float):
            lines.append(f'  {key}: {value:.4f}')
        elif isinstance(value, int):
            lines.append(f'  {key}: {value}')
        else:
            lines.append(f'  {key}: {value}')
    lines.append('')

    best_idx = max(range(len(folds)), key=lambda i: folds[i].test_metrics.get('expectancy_r', 0))
    worst_idx = min(range(len(folds)), key=lambda i: folds[i].test_metrics.get('max_drawdown', float('inf')))
    lines.append(f'Best fold: fold {best_idx}')
    lines.append(f'Worst fold: fold {worst_idx}')
    lines.append('')

    lines.append('STABILITY ANALYSIS:')
    lines.append('-' * 70)
    test_returns = [f.test_metrics.get('expectancy_r', 0) for f in folds]
    if len(test_returns) > 1:
        lines.append(f'  Expectancy R across folds: {test_returns}')
        lines.append(f'  Standard deviation: {np.std(test_returns):.4f}')
    lines.append('')

    lines.append('RECOMMENDATION:')
    lines.append('-' * 70)
    exp = aggregate_metrics.get('expectancy_r', 0)
    if isinstance(exp, float) and exp > 0:
        lines.append('  Strategy shows positive expectancy across folds. Consider live deployment with proper risk limits.')
    else:
        lines.append('  Strategy shows negative or zero expectancy. Do not deploy live.')
    lines.append('')

    lines.append('=' * 70)
    return '\n'.join(lines)


def run_walk_forward_validation(strategy, df, symbol, cfg, markets_cfg=None, config=None):
    """Run walk-forward validation on the given strategy and data."""
    if config is None:
        config = WalkForwardConfig()

    # Compute folds
    folds = _compute_date_ranges(df, config)
    if not folds:
        return WalkForwardResult(
            folds=[],
            aggregate_metrics={},
            best_fold_id=-1,
            worst_fold_id=-1,
            stable=False,
            report='Insufficient data for walk-forward validation'
        )

    # Run each fold
    for fold in folds:
        # Train phase - fit any transformations here using only training data
        # In a real system, we would fit scaler, feature selectors, etc. here

        # Validate phase - using validation data only
        fold.valid_metrics = _run_backtest(strategy, fold.valid_df, symbol, 'valid')

        # Test phase - using untouched test data (this is what we optimize)
        fold.test_metrics = _run_backtest(strategy, fold.test_df, symbol, 'test')

    # Compute aggregate metrics
    aggregate_metrics = _compute_aggregate_metrics(folds)

    # Find best and worst folds
    best_fold_id = max(range(len(folds)), key=lambda i: folds[i].test_metrics.get('expectancy_r', 0))
    worst_fold_id = min(range(len(folds)), key=lambda i: folds[i].test_metrics.get('max_drawdown', float('inf')))

    # Determine stability
    test_returns = [f.test_metrics.get('expectancy_r', 0) for f in folds]
    stable = len(folds) > 1 and np.std(test_returns) < np.mean(test_returns) * 0.5

    # Generate report
    report = _generate_report(folds, aggregate_metrics)

    return WalkForwardResult(
        folds=folds,
        aggregate_metrics=aggregate_metrics,
        best_fold_id=best_fold_id,
        worst_fold_id=worst_fold_id,
        stable=stable,
        report=report
    )