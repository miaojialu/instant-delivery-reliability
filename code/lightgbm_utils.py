
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import optuna
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# 定义 train_lightgbm_optuna 函数
# train_lightgbm_optuna 函数
# train_lightgbm_optuna 函数
def train_lightgbm_optuna(X_train, y_train, X_test, y_test, n_trials, metrics_file=None, num_boost_round=2000):
    # 调整 Optuna 的日志级别，抑制每个 trial 的日志输出
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # 定义 Optuna 的目标函数
    def lgbm_objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        # 创建 LightGBM 数据集
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # 使用回调函数进行早停训练模型
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_test],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(False)
            ]
        )

        # 预测并计算 RMSE
        preds = gbm.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        return rmse

    # 使用 Optuna 优化超参数
    study = optuna.create_study(direction='minimize')
    study.optimize(lgbm_objective, n_trials=n_trials)

    print('lightGBM 最佳超参数: {}'.format(study.best_trial.params))

    # 使用最佳超参数重新训练模型
    best_params = study.best_trial.params
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse'
    best_params['verbosity'] = -1
    best_params['boosting_type'] = 'gbdt'

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 逐轮指标收集回调
    class _MetricsRecorder:
        def __init__(self):
            self.train_rmse = []
            self.test_rmse  = []

        def __call__(self, env):
            for name, metric, value, _ in env.evaluation_result_list:
                if name == 'train':
                    self.train_rmse.append(value)
                elif name == 'test':
                    self.test_rmse.append(value)

    recorder = _MetricsRecorder()

    gbm = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(False),
            recorder,
        ]
    )
    preds = gbm.predict(X_test)

    # 计算最终 RMSE 和 R²
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, preds)
    best_round = gbm.best_iteration if gbm.best_iteration > 0 else len(recorder.test_rmse)
    print(f'lightGBM R²: {r2:.4f}  RMSE: {rmse:.4f}  最佳轮次: {best_round}')

    # ── 学习曲线图 ──────────────────────────────────────────────────────────
    if recorder.train_rmse and recorder.test_rmse:
        rounds = np.arange(1, len(recorder.test_rmse) + 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, recorder.train_rmse,
                color='#4573b4', linewidth=1.5, label='Train RMSE')
        ax.plot(rounds, recorder.test_rmse,
                color='#d73221', linewidth=1.5, linestyle='--', label='Test RMSE')
        ax.axvline(x=best_round, color='gray', linestyle=':',
                   linewidth=1.5, label=f'Best round ({best_round})')
        ax.set_xlabel('Boosting Round', fontsize=14)
        ax.set_ylabel('RMSE', fontsize=14)
        ax.set_title('LightGBM Learning Curve', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        save_dir = os.path.dirname(metrics_file) if metrics_file else '.'
        curve_path = os.path.join(save_dir, 'LGBM_learning_curve.pdf')
        plt.savefig(curve_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f'学习曲线已保存至: {curve_path}')

        # 逐轮指标 CSV
        curve_df = pd.DataFrame({
            'round':      rounds,
            'train_rmse': recorder.train_rmse,
            'test_rmse':  recorder.test_rmse,
        })
        csv_path = os.path.join(save_dir, 'LGBM_training_metrics.csv')
        curve_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'逐轮训练指标已保存至: {csv_path}')

    # 保存最终评估指标到 txt
    if metrics_file:
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write(f'LightGBM R²:   {r2:.4f}\n')
                f.write(f'LightGBM RMSE: {rmse:.4f}\n')
                f.write(f'Best Round:    {best_round}\n')
                f.write(f'\nBest Hyperparameters:\n')
                for k, v in best_params.items():
                    f.write(f'  {k}: {v}\n')
            print(f'评估指标已保存至: {metrics_file}')
        except Exception as e:
            print(f'保存评估指标失败: {e}')

    return gbm, preds


# 定义 plot_feature_importance_donut 函数（从 CatBoost 复制）
def plot_feature_importance_donut(shap_values, feature_names, save_path=None, num_var=11):
    colors = ['#4573b4','#6491c1','#d2edf2','#fcb777','#fde699','#fef9b7',
              '#d73221', '#e35235', '#e48070', '#B5AED5', '#B1B2B6']
    
    # 提取 SHAP 值的数值部分
    shap_values_array = shap_values.values
    
    # 计算每个特征的平均绝对 SHAP 值
    feature_importance = np.abs(shap_values_array).mean(axis=0)
    feature_importance_percentage = feature_importance / feature_importance.sum() * 100
    
    # 对特征按贡献率排序，取前 num_var 特征
    sorted_idx = np.argsort(feature_importance_percentage)[::-1]
    top_idx = sorted_idx[:num_var]
    others_idx = sorted_idx[num_var:]

    # 重新整理特征名和重要性
    top_names = [feature_names[i] for i in top_idx]
    top_importance = feature_importance_percentage[top_idx]
    others_importance = feature_importance_percentage[others_idx].sum()

    # 合并前 num_var 和“Others”
    final_names = top_names + ['Others']
    final_importance = np.append(top_importance, others_importance)

    # 绘制环形图
    plt.figure(figsize=(10, 6))
    wedges, texts, autotexts = plt.pie(
        final_importance,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=1.15,
        colors=colors[:len(final_names)],
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3),
        textprops={'fontsize': 12}
    )

    # 添加图例
    plt.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10)
                 for i in range(len(final_names))],
        labels=final_names,
        loc='center right',
        bbox_to_anchor=(1.75, 0.5),
        edgecolor='black',
        fancybox=False,
        fontsize=18
    )

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
from alibi.explainers import ALE
from PyALE import ale  # 导入PyALE库的ale函数
# 定义 explain_shap_lightgbm 函数（包含 ALE 图）
# explain_shap_lightgbm 函数（添加 plt.close()）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os


def calculate_optimal_bins(series):
    """计算特征的最佳分箱数（基于自由度调整的Sturges规则）"""
    n = len(series.dropna())
    if n <= 1:
        return 10
    bins = int(np.ceil(np.log2(n)) + 1)  # Sturges规则
    return max(10, min(100, bins))  # 限制在10-100之间

def explain_shap_lightgbm(model, X_train, X_test, feature_names, plot_type="bar", save_path=None, depvar=None, color_values=None):
    # 确保输入为DataFrame
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # 确保所有列为数值类型（shap.partial_dependence_plot 要求 int/float/bool）
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 设置字体大小
    plt.rcParams.update({'font.size': 18})

    # SHAP特征重要性条形图
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar",color='#4573b4', show=False)
    plt.savefig(os.path.join(save_path, f'LGBM_{depvar}_contributor.pdf'), bbox_inches='tight')
    plt.close()

    # 特征重要性环形图
    def plot_feature_importance_donut(shap_values, feature_names, save_path, num_var=12):
        colors = ['#4573b4','#6491c1','#d2edf2','#fcb777','#fde699','#fef9b7',
              '#d73221', '#e35235', '#e48070', '#B5AED5', '#B1B2B6']
        shap_values_array = shap_values.values
        feature_importance = np.abs(shap_values_array).mean(axis=0)
        feature_importance_percentage = feature_importance / feature_importance.sum() * 100
        sorted_idx = np.argsort(feature_importance_percentage)[::-1]
        top_idx = sorted_idx[:num_var]
        others_idx = sorted_idx[num_var:]
        top_names = [feature_names[i] for i in top_idx]
        top_importance = feature_importance_percentage[top_idx]
        others_importance = feature_importance_percentage[others_idx].sum()
        final_names = top_names + ['Others']
        final_importance = np.append(top_importance, others_importance)
        # 画布加宽 + 图例独立成栏右移；百分比标签自动分层避让，避免相互遮挡/压住图例
        import math
        fig = plt.figure(figsize=(14.0, 6.6))
        ax = fig.add_axes([0.005, 0.020, 0.505, 0.960])
        axl = fig.add_axes([0.520, 0.000, 0.480, 1.000])
        axl.axis('off')
        wedges, _ = ax.pie(
            final_importance, labels=None, startangle=90, counterclock=True,
            colors=colors[:len(final_names)],
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3)
        )
        LIMIT = 1.60
        ax.set_xlim(-LIMIT, LIMIT); ax.set_ylim(-LIMIT, LIMIT)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def _ov(a, b):
            return not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0)

        placed = []
        for i, w in enumerate(wedges):
            mid = math.radians((w.theta1 + w.theta2) / 2.0)
            label = f'{final_importance[i]:.1f}%'
            for lvl in (1.14, 1.30, 1.46):
                t = ax.text(lvl * math.cos(mid), lvl * math.sin(mid), label,
                            ha='center', va='center', fontsize=12, zorder=5)
                bb = t.get_window_extent(renderer=renderer).expanded(1.10, 1.30)
                if all(not _ov(bb, p) for p in placed):
                    placed.append(bb)
                    if lvl > 1.14:
                        r0, r1 = 1.02, lvl - 0.05
                        ax.plot([r0 * math.cos(mid), r1 * math.cos(mid)],
                                [r0 * math.sin(mid), r1 * math.sin(mid)],
                                color='0.55', lw=0.9, zorder=2)
                    break
                t.remove()
            else:
                t = ax.text(LIMIT * 0.97 * math.cos(mid), LIMIT * 0.97 * math.sin(mid),
                            label, ha='center', va='center', fontsize=12, zorder=5)
                placed.append(t.get_window_extent(renderer=renderer))

        leg = axl.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10)
                     for i in range(len(final_names))],
            labels=final_names, loc='center left', fontsize=18,
            edgecolor='black', fancybox=False, framealpha=1.0,
            borderaxespad=0.0, handlelength=1.6, handletextpad=0.8, labelspacing=0.55
        )
        leg.get_frame().set_linewidth(1.0)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    plot_feature_importance_donut(
        shap_values, feature_names, 
        save_path=os.path.join(save_path, f'LGBM_{depvar}_feature_importance_donut.pdf'), 
        num_var=8
    )

    # SHAP beeswarm图
    from matplotlib.colors import LinearSegmentedColormap 
    colors_new = ['#4573b4',  '#fef4ae', '#d73221']
    cmap_new = LinearSegmentedColormap.from_list('blue_yellow_red', colors_new)

    # SHAP beeswarm图
    plt.figure(figsize=(12, 6))
    shap.plots.beeswarm(shap_values, max_display=11, show=False, color=cmap_new)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=18)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=18)

    # 修改colorbar样式
    colorbar = plt.gcf().axes[-1]
    colorbar.tick_params(labelsize=18)
    colorbar.set_ylabel(colorbar.get_ylabel(), fontsize=18)
    
    plt.savefig(os.path.join(save_path, f'LGBM_{depvar}_beeswarm.pdf'), bbox_inches='tight')    
    plt.close()

    # 创建保存目录
    os.makedirs(os.path.join(save_path, f'LGBM_dependence/{depvar}/'), exist_ok=True)
    os.makedirs(os.path.join(save_path, f'LGBM_ale/{depvar}/'), exist_ok=True)

    # 遍历每个特征绘制依赖图、部分依赖图和ALE图
    for feature in feature_names:
        # 验证特征是否存在
        if feature not in X_train.columns:
            print(f"Warning: Feature '{feature}' not found in X_train. Skipping...")
            continue

        # SHAP依赖图
        plt.figure(figsize=(9, 6))
        shap.plots.scatter(
            shap_values[:, feature], alpha=0.4, dot_size=10, show=False, hist=False
        )
        ax = plt.gca()
        scaled_ticks = ax.get_xticks()
        buffer = (scaled_ticks.max() - scaled_ticks.min()) * 0.05
        plt.xlim(scaled_ticks.min(), scaled_ticks.max() + buffer)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(plt.gca().get_xlabel(), fontsize=18)
        plt.ylabel(plt.gca().get_ylabel(), fontsize=18)
        plt.savefig(
            os.path.join(save_path, f'LGBM_dependence/{depvar}/LGBM_{depvar}_{feature}_dependence_plot.pdf'), 
            bbox_inches='tight'
        )
        plt.close()

        # 带颜色的SHAP依赖图
        if color_values is not None:
            plt.figure(figsize=(9, 6))
            shap.plots.scatter(
                shap_values[:, feature], alpha=0.4, dot_size=10, show=False,
                color=color_values, hist=False
            )
            ax = plt.gca()
            scaled_ticks = ax.get_xticks()
            buffer = (scaled_ticks.max() - scaled_ticks.min()) * 0.05
            plt.xlim(scaled_ticks.min(), scaled_ticks.max() + buffer)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel(plt.gca().get_xlabel(), fontsize=18)
            plt.ylabel(plt.gca().get_ylabel(), fontsize=18)
            colorbar = plt.gcf().axes[-1]
            colorbar.tick_params(labelsize=18)
            colorbar.set_ylabel(colorbar.get_ylabel(), fontsize=18)
            plt.savefig(
                os.path.join(save_path, f'LGBM_dependence/{depvar}/LGBM_{depvar}_{feature}_dependence_plot_color.pdf'), 
                bbox_inches='tight'
            )
            plt.close()

        # 部分依赖图
        plt.figure(figsize=(9, 6))
        shap.partial_dependence_plot(
            feature, model.predict, X_test, model_expected_value=True,
            feature_expected_value=True, show=False, ice=False
        )
        ax = plt.gca()
        scaled_ticks = ax.get_xticks()
        buffer = (scaled_ticks.max() - scaled_ticks.min()) * 0.05
        plt.xlim(scaled_ticks.min(), scaled_ticks.max() + buffer)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(plt.gca().get_xlabel(), fontsize=18)
        plt.ylabel(plt.gca().get_ylabel(), fontsize=18)
        plt.savefig(
            os.path.join(save_path, f'LGBM_dependence/{depvar}/LGBM_{depvar}_{feature}_partial_dependence_plot.pdf'), 
            bbox_inches='tight'
        )
        plt.close()

        # 混合SHAP和部分依赖图
        fig, ax1 = plt.subplots(figsize=(8, 6))
        shap.plots.scatter(
            shap_values[:, feature], alpha=1, dot_size=10, show=False,
            hist=False, ax=ax1, color='#38a1db'
        )
        ax1.set_xlabel(f'{feature}', fontsize=20)
        ax1.set_ylabel('SHAP value', fontsize=20)
        ax1.tick_params(axis='x', labelsize=20)
        ax1.tick_params(axis='y', labelsize=20)
        scaled_ticks = ax1.get_xticks()
        buffer = (scaled_ticks.max() - scaled_ticks.min()) * 0.05
        ax1.set_xlim(scaled_ticks.min(), scaled_ticks.max() + buffer)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

        ax2 = ax1.twinx()
        shap.partial_dependence_plot(
            feature, model.predict, X_test, show=False, ice=False, hist=False, ax=ax2
        )
        pdp_lines = ax2.get_lines()
        if pdp_lines:
            pdp_lines[-1].set_color('#f47983')
            pdp_lines[-1].set_linewidth(2)
        ax2.set_ylabel('Partial dependence', fontsize=20)
        ax2.spines["right"].set_visible(True)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_ticks_position("right")
        ax2.tick_params(axis='y', labelsize=20)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f'LGBM_dependence/{depvar}/LGBM_{depvar}_{feature}_mixed_partial_dependence_plot.pdf'), 
            bbox_inches='tight'
        )
        plt.close()

        # ALE图与SHAP散点使用双坐标轴
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 验证特征为数值型
        if not pd.api.types.is_numeric_dtype(X_train[feature]):
            print(f"Warning: Feature '{feature}' is not numeric. Converting to numeric...")
            X_train[feature] = pd.to_numeric(X_train[feature], errors='coerce')
            if X_train[feature].isnull().all():
                print(f"Warning: Feature '{feature}' conversion failed. Skipping...")
                plt.close()
                continue

        n_unique = X_train[feature].nunique()
        if n_unique == 1:
            print(f"Warning: Feature '{feature}' has only 1 unique value. Skipping...")
            plt.close()
            continue

        # 离散/类别特征（唯一值 ≤ 10）：手动累积ALE，避免PyALE分箱问题
        _DISCRETE_THRESHOLD = 10
        if n_unique <= _DISCRETE_THRESHOLD:
            vals = np.array(sorted(X_train[feature].dropna().unique()))
            k = len(vals)

            # 对每相邻区间 [vals[m-1], vals[m]] 计算局部平均效应
            deltas, delta_ses = [], []
            for m in range(1, k):
                mask = X_train[feature].isin([vals[m - 1], vals[m]])
                X_sub = X_train[mask].copy() if mask.sum() >= 2 else X_train.copy()
                X_low = X_sub.copy(); X_low[feature] = vals[m - 1]
                X_high = X_sub.copy(); X_high[feature] = vals[m]
                d = model.predict(X_high) - model.predict(X_low)
                deltas.append(d.mean())
                delta_ses.append(d.std() / np.sqrt(len(d)) if len(d) > 1 else 0.0)

            # 累积ALE并按样本比例中心化
            ale_vals = np.concatenate([[0.0], np.cumsum(deltas)])
            counts = np.array([(X_train[feature] == v).sum() for v in vals], dtype=float)
            ale_vals -= np.dot(counts / counts.sum(), ale_vals)

            # 误差传播95% CI（相邻区间独立）
            cum_var = np.concatenate([[0.0], np.cumsum(np.array(delta_ses) ** 2)])
            ci_half = 1.96 * np.sqrt(cum_var)
            ci_lower = ale_vals - ci_half
            ci_upper = ale_vals + ci_half

            # 左轴：SHAP散点（加抖动避免重叠）
            shap_vals = shap_values[:, feature].values
            rng = np.random.default_rng(0)
            jitter_scale = max((vals[-1] - vals[0]) * 0.05, 0.05)
            x_jitter = X_test[feature].values.astype(float) + rng.uniform(
                -jitter_scale, jitter_scale, size=len(X_test)
            )
            ax1.scatter(x_jitter, shap_vals, color='#f2724d', alpha=0.4, s=20, edgecolor='none')
            ax1.set_xlabel(feature, fontsize=18)
            ax1.set_ylabel('SHAP Value', fontsize=18)
            ax1.tick_params(axis='x', labelsize=16)
            ax1.tick_params(axis='y', labelsize=16)
            ax1.set_xticks(vals)
            ax1.set_xticklabels(
                [str(int(v)) if float(v) == int(float(v)) else str(v) for v in vals]
            )
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E0E0E0', alpha=0.7)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

            # 右轴：ALE柱状图（含误差线）
            ax2 = ax1.twinx()
            val_span = float(vals[-1] - vals[0]) if k > 1 else 1.0
            bar_width = min(0.35 * val_span / k, val_span / (k + 1))
            ax2.bar(vals, ale_vals, width=bar_width, color='#4573b4', alpha=0.75,
                    yerr=[ale_vals - ci_lower, ci_upper - ale_vals],
                    capsize=6, error_kw={'linewidth': 1.5})
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax2.set_ylabel('ALE Effect', fontsize=18)
            ax2.tick_params(axis='y', labelsize=16)

            shap_min, shap_max = shap_vals.min(), shap_vals.max()
            ale_ext = max(abs(ci_lower.min()), abs(ci_upper.max()))
            padding = max(abs(shap_max), abs(shap_min), ale_ext) * 0.1
            ax1.set_ylim(shap_min - padding, shap_max + padding)
            ax2.set_ylim(ci_lower.min() - padding, ci_upper.max() + padding)
            margin = val_span * 0.3
            ax1.set_xlim(vals[0] - margin, vals[-1] + margin)

            plt.title(f'ALE & SHAP Plot: {feature}', fontsize=20, pad=15)
            plt.savefig(
                os.path.join(save_path, f'LGBM_ale/{depvar}/LGBM_{depvar}_{feature}_ale_shap_plot.pdf'),
                bbox_inches='tight', dpi=300
            )
            plt.close()
            continue

        # 连续特征：使用PyALE分箱计算
        optimal_bins = calculate_optimal_bins(X_train[feature])
        optimal_bins = min(optimal_bins, n_unique - 1)

        # 计算ALE值
        # PyALE对离散特征忽略grid_size，直接用全部唯一值，存在off-by-one bug。
        # 解决方案：对离散特征，将训练数据中该列的最大值行移除，使PyALE内部数组缩小1个单位。
        try:
            ale_result = ale(
                X=X_train,
                model=model,
                feature=[feature],
                grid_size=optimal_bins,
                include_CI=True,
                C=0.95,
                plot=False
            )
        except IndexError as e:
            # PyALE离散特征off-by-one bug的回退方案：
            # 移除最大值所在行后重试，使唯一值数量减少从而规避越界
            try:
                max_val = X_train[feature].max()
                X_train_trimmed = X_train[X_train[feature] < max_val]
                if len(X_train_trimmed) < 100:
                    print(f"Error calculating ALE for '{feature}': {e}. Skipping...")
                    plt.close()
                    continue
                ale_result = ale(
                    X=X_train_trimmed,
                    model=model,
                    feature=[feature],
                    grid_size=optimal_bins,
                    include_CI=True,
                    C=0.95,
                    plot=False
                )
                print(f"ALE for '{feature}' succeeded after trimming max value (PyALE off-by-one workaround).")
            except Exception as e2:
                print(f"Error calculating ALE for '{feature}': {e2}. Skipping...")
                plt.close()
                continue
        except Exception as e:
            print(f"Error calculating ALE for '{feature}': {e}. Skipping...")
            plt.close()
            continue

        # 调试信息
        print(f"ale_result columns for {feature}: {ale_result.columns.tolist()}")
        print(f"ale_result index for {feature}: {ale_result.index.tolist()}")

        # 提取x_values、y_values和置信区间
        x_values = ale_result.index.values
        y_values = ale_result['eff'].values
        ci_lower = ale_result['lowerCI_95%'].values
        ci_upper = ale_result['upperCI_95%'].values

        # 计算特征值的5%和95%百分位
        feature_values = X_train[feature].dropna().values
        if len(feature_values) > 0:
            percentile_5 = np.percentile(feature_values, 5)
            percentile_95 = np.percentile(feature_values, 95)
        else:
            percentile_5 = x_values.min()
            percentile_95 = x_values.max()

        # 绘制ALE曲线在ax2（右侧坐标轴）
        ax2 = ax1.twinx()
        ax2.fill_between(x_values, ci_lower, ci_upper, color='#ADD8E6', alpha=0.3)
        for i in range(len(x_values) - 1):
            x_current = x_values[i]
            x_next = x_values[i + 1]
            if x_current <= percentile_5 or x_next >= percentile_95:
                linestyle = '--'
            else:
                linestyle = '-'
            ax2.plot([x_current, x_next], [y_values[i], y_values[i + 1]],
                     color='#4573b4', linewidth=2.5, linestyle=linestyle)
        ax2.set_ylabel('ALE Effect', fontsize=18)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.spines["right"].set_visible(True)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_ticks_position("right")

        # 绘制SHAP值散点在ax1（左侧坐标轴）
        shap_vals = shap_values[:, feature].values
        ax1.scatter(X_test[feature], shap_vals, color= '#f2724d', alpha=0.4, s=20, edgecolor='none')
        ax1.set_xlabel(feature, fontsize=18)
        ax1.set_ylabel('SHAP Value', fontsize=18)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E0E0E0', alpha=0.7)

        # 调整坐标轴范围以均衡分布
        shap_min, shap_max = np.min(shap_vals), np.max(shap_vals)
        ale_min, ale_max = np.min(y_values), np.max(y_values)
        padding = max(abs(shap_max), abs(shap_min), abs(ale_max), abs(ale_min)) * 0.1
        ax1.set_ylim(shap_min - padding, shap_max + padding)
        ax2.set_ylim(ale_min - padding, ale_max + padding)

        # 添加SHAP值为0的横线
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

        # 图表美化
        plt.title(f'ALE & SHAP Plot: {feature}', fontsize=20, pad=15)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        x_range = x_values.max() - x_values.min() if len(x_values) > 1 else 1
        plt.xlim(x_values.min() - x_range * 0.05, x_values.max() + x_range * 0.05)

        # 保存ALE图并关闭
        plt.savefig(
            os.path.join(save_path, f'LGBM_ale/{depvar}/LGBM_{depvar}_{feature}_ale_shap_plot.pdf'),
            bbox_inches='tight', dpi=300
        )
        plt.close()

    # SHAP交互效应矩阵：返回 interaction_df、shap_interaction 原始数组、采样子集 X_sample
    interaction_df, shap_interaction, X_sample = plot_shap_interaction_matrix(
        model, X_test, feature_names, save_path, depvar
    )

    # SHAP Interaction Value Summary Plot（蜂群矩阵）
    if shap_interaction is not None and X_sample is not None:
        plot_shap_interaction_summary_plot(
            shap_interaction, X_sample, feature_names, save_path, depvar, top_k=6
        )

    # 二维交互ALE图
    plot_2d_ale_interactions(model, X_train, feature_names, save_path, depvar,
                             interaction_df=interaction_df, top_k=5, grid_size=20)

    # SHAP 热力图摘要图
    plot_shap_heatmap_summary(shap_values, X_test, feature_names, save_path, depvar)

    return shap_values


def plot_shap_interaction_matrix(model, X_test, feature_names, save_path, depvar, max_samples=10000, top_k=5):
    """
    计算并可视化SHAP交互效应矩阵。
    生成三类图：
      1. 下三角交互效应热图（含对角主效应）
      2. Top交互对水平条形图
      3. Top K交互对散点图（交互SHAP值 vs 主特征，按伴随特征五分位数着色）

    参数：
      max_samples: 计算交互值时最多使用的样本数（无硬性上限，上限=len(X_test)）
      top_k: 绘制散点图的Top交互对数量
    """
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # 按 time_period 分层采样，保证高峰/非高峰比例；若列不存在则退化为随机采样
    if 'time_period' in X_test.columns:
        n_groups = X_test['time_period'].nunique()
        X_sample = X_test.groupby('time_period', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, max_samples // n_groups)), random_state=42)
        )
    else:
        X_sample = X_test.sample(min(max_samples, len(X_test)), random_state=42)

    print(f"正在计算SHAP交互值（样本数={len(X_sample)}，特征数={len(feature_names)}）...")
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(X_sample)
    # shap_interaction shape: (n_samples, n_features, n_features)

    # 计算平均绝对交互值矩阵
    mean_abs_interaction = np.abs(shap_interaction).mean(axis=0)
    n_features = len(feature_names)
    interaction_df = pd.DataFrame(mean_abs_interaction, index=feature_names, columns=feature_names)

    os.makedirs(save_path, exist_ok=True)

    # ── 图1：下三角交互效应热图 ──────────────────────────────────────────
    # 遮蔽上三角（矩阵对称，避免重复；对角线保留展示主效应）
    mask_upper = np.triu(np.ones((n_features, n_features), dtype=bool), k=1)

    cmap_inter = LinearSegmentedColormap.from_list('white_blue', ['#ffffff', '#4573b4'])

    fig_size = max(8, n_features * 0.85)
    _, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        interaction_df,
        mask=mask_upper,
        annot=True,
        fmt='.3f',
        cmap=cmap_inter,
        linewidths=0.8,
        linecolor='#f0f0f0',
        ax=ax,
        cbar_kws={'label': 'Mean |SHAP Interaction Value|', 'shrink': 0.8},
        annot_kws={'size': max(7, 11 - n_features // 3)}
    )

    # 用红色边框高亮对角线（主效应区域）
    for i in range(n_features):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='#d73221', lw=1.5))

    ax.set_title(f'SHAP Interaction Value Matrix: {depvar}', fontsize=16, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                       fontsize=max(9, 13 - n_features // 3))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=max(9, 13 - n_features // 3))

    # 添加图例说明对角线含义
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#4573b4', label='Pairwise Interaction'),
        Patch(facecolor='white', edgecolor='#d73221', linewidth=1.5, label='Main Effect (diagonal)')
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              bbox_to_anchor=(1.0, 1.12), fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f'LGBM_{depvar}_shap_interaction_matrix.pdf'),
        bbox_inches='tight', dpi=300
    )
    plt.close()

    # ── 图2：Top交互对水平条形图 ──────────────────────────────────────────
    # 提取上三角非对角线元素（即纯交互对，排除主效应）
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append({
                'pair': f'{feature_names[i]}\n× {feature_names[j]}',
                'value': mean_abs_interaction[i, j]
            })

    pairs_df = pd.DataFrame(pairs).sort_values('value', ascending=False).head(15).reset_index(drop=True)

    # 用红→蓝渐变区分强弱交互
    median_val = pairs_df['value'].median()
    bar_colors = ['#d73221' if v >= median_val else '#4573b4' for v in pairs_df['value']]

    _, ax = plt.subplots(figsize=(10, max(5, len(pairs_df) * 0.55)))
    bars = ax.barh(pairs_df['pair'], pairs_df['value'],
                   color=bar_colors, edgecolor='white', height=0.6)

    # 在条形末端标注数值
    for bar, val in zip(bars, pairs_df['value']):
        ax.text(bar.get_width() + pairs_df['value'].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=10)

    ax.set_xlabel('Mean |SHAP Interaction Value|', fontsize=14)
    ax.set_title(f'Top Feature Interaction Pairs: {depvar}', fontsize=16, pad=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.6, color='#cccccc')
    ax.spines[['top', 'right']].set_visible(False)

    # 图例说明颜色含义
    from matplotlib.patches import Patch
    legend_handles2 = [
        Patch(facecolor='#d73221', label=f'≥ median ({median_val:.4f})'),
        Patch(facecolor='#4573b4', label=f'< median')
    ]
    ax.legend(handles=legend_handles2, fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f'LGBM_{depvar}_shap_interaction_top_pairs.pdf'),
        bbox_inches='tight', dpi=300
    )
    plt.close()

    # ── 图3：Top K 交互依赖图（SHAP Interaction Value Dependence Plots）──────────
    # X轴：主特征值；Y轴：交互SHAP值；颜色：伴随特征连续取值（连续colormap）
    # 若颜色呈系统性分层（红色点偏上/蓝色点偏下）→ 强交互；颜色随机混杂 → 弱交互
    scatter_dir = os.path.join(save_path, f'LGBM_interaction_scatter/{depvar}/')
    os.makedirs(scatter_dir, exist_ok=True)

    cmap_dep = LinearSegmentedColormap.from_list(
        'blue_yellow_red', ['#4573b4', '#fef4ae', '#d73221']
    )

    # 提取top_k个交互强度最大的特征对（排除对角线主效应）
    raw_pairs = [
        (i, j, mean_abs_interaction[i, j])
        for i in range(n_features)
        for j in range(i + 1, n_features)
    ]
    raw_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = raw_pairs[:min(top_k, len(raw_pairs))]

    # 补充必须包含的交互对（若不在top_k中则追加）
    _required_pairs = [
        ('wave_duration_time_x', 'loadorder_number'),
        ('courier_work_hours_now', 'loadorder_number'),
        ('prev_standardized_delay', 'loadorder_number'),
        ('ride_distance_km', 'loadorder_number'),
    ]
    _fn_list = list(feature_names)
    _top_set = {(feature_names[i], feature_names[j]) for i, j, _ in top_pairs}
    for _fa, _fb in _required_pairs:
        if _fa in _fn_list and _fb in _fn_list:
            _ia, _ib = _fn_list.index(_fa), _fn_list.index(_fb)
            _i, _j = (_ia, _ib) if _ia < _ib else (_ib, _ia)
            if (feature_names[_i], feature_names[_j]) not in _top_set:
                top_pairs.append((_i, _j, mean_abs_interaction[_i, _j]))
                _top_set.add((feature_names[_i], feature_names[_j]))

    for feat_i_idx, feat_j_idx, inter_strength in top_pairs:
        feat_i = feature_names[feat_i_idx]
        feat_j = feature_names[feat_j_idx]

        inter_vals = shap_interaction[:, feat_i_idx, feat_j_idx]
        x_vals = X_sample[feat_i].values
        color_vals = X_sample[feat_j].values

        # 连续颜色映射（5%~95% 截断避免极端值影响色阶）
        vmin = np.percentile(color_vals, 5)
        vmax = np.percentile(color_vals, 95)

        _, ax = plt.subplots(figsize=(9, 6))

        sc = ax.scatter(
            x_vals, inter_vals,
            c=color_vals, cmap=cmap_dep,
            alpha=0.4, s=15, edgecolor='none',
            vmin=vmin, vmax=vmax
        )

        # 全样本分箱均值趋势线（整体趋势）
        sort_idx = np.argsort(x_vals)
        x_s, y_s = x_vals[sort_idx], inter_vals[sort_idx]
        n_bins = min(20, len(x_s))
        bin_edges = np.linspace(x_s.min(), x_s.max(), n_bins + 1)
        bin_x, bin_y = [], []
        for b in range(n_bins):
            in_bin = (x_s >= bin_edges[b]) & (x_s <= bin_edges[b + 1])
            if in_bin.sum() > 0:
                bin_x.append(x_s[in_bin].mean())
                bin_y.append(y_s[in_bin].mean())
        if len(bin_x) > 1:
            ax.plot(bin_x, bin_y, color='#333333', linewidth=2,
                    alpha=0.85, label='Binned mean', zorder=5)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(feat_j, fontsize=13)
        cbar.ax.tick_params(labelsize=11)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#E0E0E0', alpha=0.7)
        ax.set_xlabel(feat_i, fontsize=16)
        ax.set_ylabel(f'SHAP Interaction Value\n({feat_i} × {feat_j})', fontsize=14)
        ax.set_title(
            f'Interaction SHAP: {feat_i} × {feat_j}\n'
            f'(Mean |interaction| = {inter_strength:.4f})',
            fontsize=14, pad=10
        )
        ax.tick_params(labelsize=13)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(scatter_dir, f'LGBM_{depvar}_{feat_i}_x_{feat_j}_interaction.pdf'),
            bbox_inches='tight', dpi=300
        )
        plt.close()

    print(f"SHAP交互效应矩阵图已保存至: {save_path}")
    return interaction_df, shap_interaction, X_sample


    """
    使用 SHAP 解释 CatBoost 模型。

    参数：
    - model: 训练好的 CatBoost 模型
    - X_train: 训练集特征（用于构建 SHAP 解释器）
    - X_test: 测试集特征（用于计算 SHAP 值）
    - feature_names: 特征名称列表
    - plot_type: SHAP 图类型（"bar" 或 "summary"）
    """
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    # 如果 X_train 不是 DataFrame，将其转换为 DataFrame 并添加列名
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # 初始化 SHAP 解释器
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test)

    for i, feature in enumerate(feature_names):
        
        # 设置颜色
        # color_shap = 'tab:blue'
        # color_pd = 'tab:red'
        
        # 创建一个图形和主坐标轴
        fig, ax1 = plt.subplots(figsize=(9, 6))

        shap.plots.scatter(shap_values[:, feature], 
                           alpha=0.4,
                           dot_size=10,
                           show=False,
                           hist=False,
                           ax=ax1,
                           )
        
        ax1.set_xlabel(f'{feature}', fontsize=18)
        ax1.set_ylabel('SHAP Value', fontsize=18)
        
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)
        
        # 添加网格线
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        # 添加 SHAP=0 的横线
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # 5.2 创建第二个 y 轴用于绘制部分依赖图
        ax2 = ax1.twinx()

        # make a standard partial dependence plot with a single SHAP value overlaid
        shap.partial_dependence_plot(
            feature,
            model.predict,
            X_test,
            # model_expected_value=True,
            # feature_expected_value=True,
            show=False,
            ice=False,  
            hist=False,
            ax=ax2,
        )
        
        # 然后获取 ax2 上的所有线条
        pdp_lines = ax2.get_lines()

        # 如果只想改最后画上去的那一条线，可按索引 [-1]
        if pdp_lines:
            pdp_lines[-1].set_color('#f47983')     # 修改颜色
            pdp_lines[-1].set_linewidth(2)     # 修改线宽
                
        # 5) 设置右侧 y 轴标签
        ax2.set_ylabel('Partial Dependence', fontsize=18)
        
        ax2.spines["right"].set_visible(True)  
        
        # 6) 将第二个 y 轴刻度与标签放在右边，并设置字体大小
        ax2.yaxis.set_label_position("right")  
        ax2.yaxis.set_ticks_position("right")
        ax2.tick_params(axis='y', labelsize=18)
        
        
        # 获取当前的坐标轴
        current_axes = plt.gca()
        # 获取当前x轴的刻度位置（标准化后的值）
        scaled_ticks = current_axes.get_xticks()
        # 找到当前特征在特征列表中的索引
        feature_index = feature_names.index(feature)
        
        min_val = data_scaler.data_min_[feature_index]
        max_val = data_scaler.data_max_[feature_index]
        
        # 将标准化后的刻度位置反标准化为原始数据值
        original_ticks = scaled_ticks * (max_val - min_val) + min_val

        # 设置 x 轴范围，基于原始数据值并添加一定的缓冲区
        buffer = (scaled_ticks.max() - scaled_ticks.min()) * 0.05  # 5% 缓冲
        # plt.xlim(0, scaled_ticks.max() + buffer)
        
        # 设置新的刻度标签为原始数据值，并保留两位小数
        ax2.set_xticks(scaled_ticks)
        ax2.set_xticklabels([f"{tick:.0f}" for tick in original_ticks])
        plt.xlim(0, scaled_ticks.max() + buffer)
        # plt.xlim(0, 1)  # 将 x 轴限制在 0 到 1 之间
        
        # 调整布局并显示图形
        plt.tight_layout()
        plt.show()

    return shap_values


# ─────────────────────────────────────────────────────────────────────────────
# 二维交互累计局部效应图 (2D ALE)
# ─────────────────────────────────────────────────────────────────────────────
def plot_2d_ale_interactions(model, X_train, feature_names, save_path, depvar,
                             interaction_df=None, top_k=5, grid_size=20):
    """
    对交互强度最高的 top_k 个特征对绘制二维 ALE 热力图。

    参数：
      interaction_df : plot_shap_interaction_matrix 返回的 DataFrame（均值绝对交互值矩阵）。
                       若为 None，则取所有特征对的前 top_k 个。
      top_k          : 绘制的特征对数量。
      grid_size      : 每个维度的分箱数。
    """
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    ale2d_dir = os.path.join(save_path, f'LGBM_ale2d/{depvar}/')
    os.makedirs(ale2d_dir, exist_ok=True)

    # 确定要绘制的特征对
    n = len(feature_names)
    if interaction_df is not None:
        pairs_strength = []
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = feature_names[i], feature_names[j]
                pairs_strength.append((fi, fj, interaction_df.loc[fi, fj]))
        pairs_strength.sort(key=lambda x: x[2], reverse=True)
        top_pairs = [(fi, fj) for fi, fj, _ in pairs_strength[:top_k]]
    else:
        top_pairs = [
            (feature_names[i], feature_names[j])
            for i in range(n) for j in range(i + 1, n)
        ][:top_k]

    # 补充必须包含的交互对（若不在top_k中则追加）
    _required_pairs = [
        ('wave_duration_time_x', 'loadorder_number'),
        ('courier_work_hours_now', 'loadorder_number'),
        ('prev_standardized_delay', 'loadorder_number'),
        ('ride_distance_km', 'loadorder_number'),
    ]
    _fn_set = set(feature_names)
    _top_normalized = {tuple(sorted(p)) for p in top_pairs}
    for _fa, _fb in _required_pairs:
        if _fa in _fn_set and _fb in _fn_set and tuple(sorted((_fa, _fb))) not in _top_normalized:
            top_pairs.append((_fa, _fb))
            _top_normalized.add(tuple(sorted((_fa, _fb))))

    cmap_ale2d = LinearSegmentedColormap.from_list(
        'blue_white_red', ['#4573b4', '#ffffff', '#d73221']
    )

    for feat_i, feat_j in top_pairs:
        if feat_i not in X_train.columns or feat_j not in X_train.columns:
            print(f"Warning: '{feat_i}' 或 '{feat_j}' 不在 X_train 中，跳过...")
            continue
        for col in [feat_i, feat_j]:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')

        try:
            ale_result = ale(
                X=X_train,
                model=model,
                feature=[feat_i, feat_j],
                grid_size=grid_size,
                plot=False
            )
        except Exception as e:
            print(f"2D ALE 计算失败 ({feat_i} × {feat_j}): {e}，跳过...")
            continue

        # ale_result 为 DataFrame，index=feat_i bins，columns=feat_j bins
        fig, ax = plt.subplots(figsize=(9, 7))
        vmax = np.abs(ale_result.values).max()

        sns.heatmap(
            ale_result,
            cmap=cmap_ale2d,
            center=0,
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            cbar_kws={'label': 'ALE Effect', 'shrink': 0.85},
            linewidths=0,
        )

        # 格式化刻度：只显示首/中/末
        def sparse_labels(labels, n_show=6):
            step = max(1, len(labels) // n_show)
            return [
                f'{float(l.get_text()):.2g}' if idx % step == 0 else ''
                for idx, l in enumerate(labels)
            ]

        ax.set_xticklabels(sparse_labels(ax.get_xticklabels()), rotation=45,
                           ha='right', fontsize=11)
        ax.set_yticklabels(sparse_labels(ax.get_yticklabels()), rotation=0,
                           fontsize=11)
        ax.set_xlabel(feat_j, fontsize=14)
        ax.set_ylabel(feat_i, fontsize=14)
        ax.set_title(f'2D ALE: {feat_i} × {feat_j}\n({depvar})',
                     fontsize=15, pad=12)

        plt.tight_layout()
        safe_i = feat_i.replace('/', '_')
        safe_j = feat_j.replace('/', '_')
        plt.savefig(
            os.path.join(ale2d_dir, f'LGBM_{depvar}_{safe_i}_x_{safe_j}_ale2d.pdf'),
            bbox_inches='tight', dpi=300
        )
        plt.close()

    print(f"二维 ALE 图已保存至: {ale2d_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Interaction Value Summary Plot（蜂群矩阵）
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_interaction_summary_plot(shap_interaction, X_sample, feature_names,
                                       save_path, depvar, top_k=6):
    """
    SHAP Interaction Value Summary Plot（蜂群矩阵）。

    以 top_k × top_k 的子图矩阵呈现交互值全景：
      - 对角线 (i==i)：纯独立效应（已剥离交互），颜色 = 该特征自身值
      - 非对角线 (i,j)：特征 i 与特征 j 的交互 SHAP 值；
        若颜色（特征 j 值）在纵向呈现系统性分层 → 强交互
      - 颜色：蓝(低)→黄→红(高)，与标准 SHAP 配色一致

    参数：
      shap_interaction : ndarray, shape (n_samples, n_features, n_features)
      X_sample         : DataFrame，与 shap_interaction 对齐的样本
      top_k            : 按总交互强度选取的 Top-K 特征数量
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.ticker as ticker

    n = len(feature_names)

    # ── 选取 Top-K 特征（按非对角交互强度总和排序）──────────────────────────
    mean_abs = np.abs(shap_interaction).mean(axis=0)
    inter_only = mean_abs.copy()
    np.fill_diagonal(inter_only, 0)
    total_inter = inter_only.sum(axis=1)
    top_idx = np.argsort(total_inter)[::-1][:min(top_k, n)]
    k = len(top_idx)
    top_features = [feature_names[i] for i in top_idx]

    cmap = LinearSegmentedColormap.from_list(
        'blue_yellow_red', ['#4573b4', '#fef4ae', '#d73221']
    )

    cell_size = 2.6
    fig, axes = plt.subplots(k, k, figsize=(k * cell_size + 1.2, k * cell_size))
    if k == 1:
        axes = np.array([[axes]])

    # ── 绘制每个子图 ──────────────────────────────────────────────────────────
    for r, i in enumerate(top_idx):
        for c, j in enumerate(top_idx):
            ax = axes[r, c]

            y_vals = shap_interaction[:, i, j]       # 交互 SHAP 值
            x_vals = X_sample[feature_names[i]].values  # 主特征（行特征）值
            color_vals = X_sample[feature_names[j]].values  # 伴随特征（列特征）值

            vmin = np.percentile(color_vals, 5)
            vmax = np.percentile(color_vals, 95)

            ax.scatter(
                x_vals, y_vals,
                c=color_vals, cmap=cmap,
                alpha=0.25, s=4,
                edgecolors='none',
                vmin=vmin, vmax=vmax,
                rasterized=True
            )
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.6)

            # 对角线用红色边框标出（纯独立主效应）
            edge_color = '#d73221' if i == j else '#cccccc'
            edge_lw = 1.5 if i == j else 0.5
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(edge_lw)

            # 刻度精简
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.tick_params(labelsize=6)

            # 行列标签（仅边缘显示）
            if r == 0:
                ax.set_title(top_features[c], fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(top_features[r], fontsize=8, labelpad=3)
            else:
                ax.set_yticklabels([])
            if r == k - 1:
                ax.set_xlabel(top_features[c], fontsize=7)
            else:
                ax.set_xticklabels([])

    # ── 共享颜色条 ─────────────────────────────────────────────────────────────
    fig.subplots_adjust(right=0.87, hspace=0.08, wspace=0.08)
    cbar_ax = fig.add_axes([0.89, 0.12, 0.025, 0.74])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Feature Value (low → high)', fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f'SHAP Interaction Value Summary Plot: {depvar}\n'
        f'(diagonal = pure main effect  |  off-diagonal = pairwise interaction)',
        fontsize=12, y=1.01
    )

    plt.savefig(
        os.path.join(save_path, f'LGBM_{depvar}_shap_interaction_summary_plot.pdf'),
        bbox_inches='tight', dpi=300
    )
    plt.close()
    print(f"SHAP交互蜂群矩阵已保存至: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 特征交互聚类热力图
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_interaction_clustermap(interaction_df, feature_names, save_path, depvar):
    """
    对 SHAP 交互效应矩阵进行层次聚类并绘制聚类热力图，
    同时突出显示对角线（主效应）与非对角交互强度。
    """
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # 非对角交互矩阵（主效应置0，仅保留交互项）
    inter_only = interaction_df.copy()
    np.fill_diagonal(inter_only.values, 0)

    cmap_clus = LinearSegmentedColormap.from_list(
        'white_blue_dark', ['#f7fbff', '#6491c1', '#08306b']
    )

    n = len(feature_names)
    fig_size = max(8, n * 0.9)

    g = sns.clustermap(
        inter_only,
        cmap=cmap_clus,
        method='average',
        metric='euclidean',
        figsize=(fig_size, fig_size),
        annot=(n <= 15),
        fmt='.3f',
        annot_kws={'size': max(7, 11 - n // 3)},
        linewidths=0.5,
        linecolor='#e8e8e8',
        cbar_pos=(0.02, 0.85, 0.03, 0.12),
        dendrogram_ratio=0.12,
    )

    g.ax_heatmap.set_title(
        f'SHAP Interaction Clustermap: {depvar}',
        fontsize=15, pad=14
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha='right',
        fontsize=max(9, 13 - n // 3)
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0,
        fontsize=max(9, 13 - n // 3)
    )
    g.cax.set_ylabel('Mean |SHAP Interaction|', fontsize=10)

    # clustermap 创建独立 Figure，必须用 g.savefig() 而非 plt.savefig()
    g.savefig(
        os.path.join(save_path, f'LGBM_{depvar}_shap_interaction_clustermap.pdf'),
        bbox_inches='tight', dpi=300
    )
    plt.close(g.figure)
    print(f"聚类交互热力图已保存至: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP 热力图摘要图
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_heatmap_summary(shap_values, X_test, feature_names, save_path, depvar,
                              max_display=15, max_samples=500):
    """
    绘制 SHAP 热力图摘要图（shap.plots.heatmap）：
      - 行为样本，列为特征，颜色为 SHAP 值
      - 上方显示平均绝对 SHAP 值条形图，右侧显示预测值曲线
    参数：
      max_display : 显示的最大特征数（按平均|SHAP|降序排列）
      max_samples : 传入热力图的最大样本数，防止大数据集 MemoryError
    """
    from matplotlib.colors import LinearSegmentedColormap

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # 子采样以防止内存溢出
    n = len(shap_values)
    if n > max_samples:
        idx = np.random.default_rng(42).choice(n, max_samples, replace=False)
        idx.sort()
        shap_values_plot = shap_values[idx]
    else:
        shap_values_plot = shap_values

    cmap_hm = LinearSegmentedColormap.from_list(
        'blue_yellow_red', ['#4573b4', '#fef4ae', '#d73221']
    )

    fig_w = max(12, max_display * 0.8)
    plt.figure(figsize=(fig_w, 7))

    shap.plots.heatmap(
        shap_values_plot,
        max_display=max_display,
        show=False,
        cmap=cmap_hm,
    )

    # 统一字体大小
    for ax in plt.gcf().axes:
        ax.tick_params(labelsize=13)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=15)

    plt.suptitle(f'SHAP Heatmap Summary: {depvar}', fontsize=16, y=1.01)
    plt.savefig(
        os.path.join(save_path, f'LGBM_{depvar}_shap_heatmap_summary.pdf'),
        bbox_inches='tight', dpi=300
    )
    plt.close()
    print(f"SHAP 热力图摘要图已保存至: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP 瀑布图：极端订单深度剖析
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_waterfall_extreme(
    model,
    shap_values,
    X_test,
    save_path,
    depvar,
    top_n=3,
    extra_indices=None,
    extra_order_ids=None,
    order_id_col=None,
):
    """
    对极端订单绘制 SHAP 瀑布图，深度剖析单笔订单的特征贡献。

    自动选取：
      - 严重超时组：预测值最高的 top_n 笔订单
      - 超前完成组：预测值最低的 top_n 笔订单

    手动追加：
      - extra_indices   : X_test 中的行索引列表（整数位置索引）
      - extra_order_ids : 订单 ID 值列表（需同时指定 order_id_col）
      - order_id_col    : X_test 中存放订单 ID 的列名

    参数
    ----
    top_n          : 每组自动选取的订单数，默认 3
    extra_indices  : list[int]，额外指定的行索引，可为 None
    extra_order_ids: list，额外指定的订单 ID 值，可为 None
    order_id_col   : str，订单 ID 所在列名，与 extra_order_ids 配合使用
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test 必须为 DataFrame")

    preds = model.predict(X_test)
    n = len(X_test)

    # ── 1. 自动选取极端组 ──────────────────────────────────────────────────
    sorted_asc  = np.argsort(preds)          # 升序：最低在前
    sorted_desc = np.argsort(preds)[::-1]    # 降序：最高在前

    severe_idx = list(sorted_desc[:min(top_n, n)])   # 严重超时
    early_idx  = list(sorted_asc[:min(top_n, n)])    # 超前完成

    # ── 2. 手动追加额外订单 ────────────────────────────────────────────────
    extra_idx = []

    if extra_indices is not None:
        for idx in extra_indices:
            if 0 <= idx < n:
                extra_idx.append(int(idx))
            else:
                print(f"Warning: 行索引 {idx} 超出范围（0~{n-1}），已跳过。")

    if extra_order_ids is not None:
        if order_id_col is None or order_id_col not in X_test.columns:
            print(f"Warning: order_id_col='{order_id_col}' 未在 X_test 中找到，跳过 ID 查找。")
        else:
            for oid in extra_order_ids:
                matched = X_test.index[X_test[order_id_col] == oid].tolist()
                if matched:
                    # 转换为整数位置索引
                    pos = X_test.index.get_loc(matched[0])
                    extra_idx.append(int(pos) if isinstance(pos, (int, np.integer)) else int(pos.start))
                else:
                    print(f"Warning: 订单 ID '{oid}' 未在 X_test 中找到，已跳过。")

    # ── 3. 按组汇总，去重，保留首次出现的标签 ─────────────────────────────
    # label: 0=严重超时, 1=超前完成, 2=手动指定
    label_map = {}
    for idx in severe_idx:
        label_map.setdefault(idx, '严重超时')
    for idx in early_idx:
        label_map.setdefault(idx, '超前完成')
    for idx in extra_idx:
        label_map.setdefault(idx, '手动指定')

    # ── 4. 按组分目录保存 ─────────────────────────────────────────────────
    waterfall_dir = os.path.join(save_path, f'LGBM_waterfall/{depvar}/')
    os.makedirs(waterfall_dir, exist_ok=True)

    # 全英文标签，避免 Times New Roman 等西文字体无法显示中文
    GROUP_EN  = {'严重超时': 'Severely_Delayed',
                 '超前完成': 'Early_Completed',
                 '手动指定': 'Manual'}
    GROUP_TITLE = {'严重超时': 'Severely Delayed',
                   '超前完成': 'Early Completed',
                   '手动指定': 'Manual'}

    group_order = ['严重超时', '超前完成', '手动指定']
    groups = {g: [] for g in group_order}
    for idx, label in label_map.items():
        groups[label].append(idx)

    # 临时切换为支持英文的字体，画完后恢复
    _prev_family = plt.rcParams.get('font.family', 'sans-serif')
    plt.rcParams['font.family'] = 'Times New Roman'

    for group_name in group_order:
        idxs = groups[group_name]
        if not idxs:
            continue
        reverse = (group_name == '严重超时')
        idxs.sort(key=lambda i: preds[i], reverse=reverse)

        for rank, i in enumerate(idxs, start=1):
            pred_val = preds[i]

            shap.plots.waterfall(shap_values[i], show=False)
            fig = plt.gcf()

            title = (f'[{GROUP_TITLE[group_name]}] Order #{rank}'
                     f'  |  Pred: {pred_val:.4f}')
            fig.suptitle(title, fontsize=12, fontfamily='Times New Roman',
                         y=1.02, ha='center')

            fname = (f'LGBM_{depvar}_{GROUP_EN[group_name]}'
                     f'_rank{rank:02d}_pred{pred_val:.4f}.pdf')
            plt.savefig(os.path.join(waterfall_dir, fname),
                        bbox_inches='tight', dpi=300)
            plt.close()

    plt.rcParams['font.family'] = _prev_family

    total = len(label_map)
    print(f"SHAP 瀑布图已保存至: {waterfall_dir}  （共 {total} 张）")