import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from xgboost import XGBRegressor
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


warnings.filterwarnings("ignore")


DESKTOP = Path.home() / "Desktop"
DEFAULT_DATA_DIR = DESKTOP / "kaggle"
TARGET = "return_pct"
ID_COL = "id"
DROP_IDENTITY_COLS = [TARGET, ID_COL, "ticker"]


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true, y_pred):
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def safe_ratio(df, num, den, name):
    if num not in df.columns or den not in df.columns:
        return
    denominator = df[den].replace(0, np.nan).astype(float)
    df[name] = (df[num].astype(float) / denominator).replace([np.inf, -np.inf], np.nan)


def add_features(train, test, include_year=True):
    train_part = train.copy()
    test_part = test.copy()
    train_part["_is_train"] = 1
    test_part["_is_train"] = 0
    test_part[TARGET] = np.nan
    df = pd.concat([train_part, test_part], ignore_index=True, sort=False)

    for col in ["period_start", "period_end"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    if not include_year and "start_year" in df.columns:
        df.drop(columns=["start_year"], inplace=True)

    base_num = [
        col
        for col in df.columns
        if col not in [ID_COL, "ticker", TARGET, "_is_train"] and pd.api.types.is_numeric_dtype(df[col])
    ]
    ratio_cols = [col for col in base_num if col not in ["start_year", "sector_code"]]

    df["missing_count"] = df[base_num].isna().sum(axis=1).astype("float32")
    df["missing_frac"] = df["missing_count"] / max(1, len(base_num))
    df["zero_count"] = (df[base_num].fillna(np.nan) == 0).sum(axis=1).astype("float32")
    df["negative_count"] = (df[base_num].fillna(np.nan) < 0).sum(axis=1).astype("float32")

    for col in ratio_cols:
        values = df[col].astype(float)
        df[f"{col}_logabs"] = np.sign(values) * np.log1p(np.abs(values))
        lo, hi = values.quantile(0.005), values.quantile(0.995)
        df[f"{col}_winsor"] = values.clip(lo, hi)

    safe_ratio(df, "net_income_ttm", "revenue_ttm", "net_income_to_revenue")
    safe_ratio(df, "income_before_tax", "revenue_ttm", "pretax_to_revenue")
    safe_ratio(df, "stockholders_equity", "total_assets", "equity_to_assets")
    safe_ratio(df, "long_term_debt", "total_assets", "debt_to_assets")
    safe_ratio(df, "goodwill", "total_assets", "goodwill_to_assets")
    safe_ratio(df, "inventory", "current_assets", "inventory_to_current_assets")
    safe_ratio(df, "current_assets", "total_assets", "current_assets_to_assets")
    safe_ratio(df, "current_liabilities", "total_assets", "current_liabilities_to_assets")
    safe_ratio(df, "dividends_ttm", "net_income_ttm", "dividends_to_income")
    safe_ratio(df, "dividends_paid_ttm", "net_income_ttm", "dividends_paid_to_income")
    safe_ratio(df, "net_income_ttm", "shares_diluted", "income_per_diluted_share")
    safe_ratio(df, "revenue_ttm", "shares_diluted", "sales_per_diluted_share")
    safe_ratio(df, "eps_basic", "eps_diluted", "eps_basic_to_diluted")

    for col in [
        "pe_ttm",
        "price_to_book",
        "price_to_sales",
        "growth_pe_ratio",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
    ]:
        if col in df.columns:
            df[f"{col}_inv"] = 1.0 / df[col].replace(0, np.nan).astype(float)
            df[f"{col}_positive"] = (df[col].astype(float) > 0).astype("float32")

    if {"price_to_sales", "revenue_ttm"} <= set(df.columns):
        df["market_cap_sales"] = df["price_to_sales"] * df["revenue_ttm"]
    if {"price_to_book", "stockholders_equity"} <= set(df.columns):
        df["market_cap_book"] = df["price_to_book"] * df["stockholders_equity"]
    if {"pe_ttm", "net_income_ttm"} <= set(df.columns):
        df["market_cap_pe_income"] = df["pe_ttm"] * df["net_income_ttm"]
    for col in ["market_cap_sales", "market_cap_book", "market_cap_pe_income"]:
        if col in df.columns:
            df[f"{col}_logabs"] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    if {"eps_basic", "pe_ttm"} <= set(df.columns):
        df["price_est_basic"] = df["eps_basic"] * df["pe_ttm"]
    if {"eps_diluted", "pe_ttm"} <= set(df.columns):
        df["price_est_diluted"] = df["eps_diluted"] * df["pe_ttm"]

    safe_ratio(df, "market_cap_sales", "market_cap_book", "mcap_sales_to_book")
    safe_ratio(df, "market_cap_pe_income", "market_cap_sales", "mcap_pe_to_sales")
    safe_ratio(df, "market_cap_pe_income", "market_cap_book", "mcap_pe_to_book")

    stat_source = [
        col
        for col in [
            "pe_ttm",
            "price_to_book",
            "price_to_sales",
            "growth_pe_ratio",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "roa",
            "roe",
            "rote",
            "revenue_growth_yoy",
            "revenue_growth_3y",
            "revenue_ttm",
            "net_income_ttm",
            "income_before_tax",
            "total_assets",
            "stockholders_equity",
            "current_assets",
            "current_liabilities",
            "long_term_debt",
            "shares_diluted",
            "shares_outstanding",
            "market_cap_sales_logabs",
            "market_cap_book_logabs",
            "market_cap_pe_income_logabs",
        ]
        if col in df.columns
    ]

    if "sector_code" in df.columns:
        for col in stat_source:
            group = df.groupby("sector_code", dropna=False)[col]
            median = group.transform("median")
            q75 = group.transform(lambda s: s.quantile(0.75))
            q25 = group.transform(lambda s: s.quantile(0.25))
            iqr = (q75 - q25).replace(0, np.nan)
            df[f"{col}_sector_diff"] = df[col] - median
            df[f"{col}_sector_z"] = (df[col] - median) / iqr
        df["sector_cat"] = df["sector_code"].fillna(-1).astype(int).astype(str)

    # Ticker identity itself is excluded. These are only within-company feature summaries.
    for col in stat_source[:18]:
        group = df.groupby("ticker", dropna=False)[col]
        mean = group.transform("mean")
        median = group.transform("median")
        df[f"{col}_ticker_mean"] = mean
        df[f"{col}_ticker_std"] = group.transform("std")
        df[f"{col}_ticker_diff"] = df[col] - mean
        df[f"{col}_ticker_med_diff"] = df[col] - median
        df[f"{col}_ticker_rank"] = group.rank(pct=True, method="average")

    train_features = df[df["_is_train"].eq(1)].drop(columns=["_is_train"])
    test_features = df[df["_is_train"].eq(0)].drop(columns=["_is_train", TARGET])
    features = [col for col in train_features.columns if col not in DROP_IDENTITY_COLS]
    cat_features = [features.index("sector_cat")] if "sector_cat" in features else []
    numeric_features = [col for col in features if col != "sector_cat"]
    return train_features, test_features, features, cat_features, numeric_features


class TargetTransform:
    def __init__(self, kind="identity", lo=None, hi=None, scale=50):
        self.kind = kind
        self.lo = lo
        self.hi = hi
        self.scale = scale

    def transform(self, y):
        values = np.asarray(y, dtype=float)
        if self.lo is not None or self.hi is not None:
            values = np.clip(
                values,
                self.lo if self.lo is not None else -np.inf,
                self.hi if self.hi is not None else np.inf,
            )
        if self.kind == "identity":
            return values
        if self.kind == "asinh":
            return np.arcsinh(values / self.scale)
        if self.kind == "loggross":
            return np.log(np.maximum(0.005, 1 + values / 100.0))
        raise ValueError(f"Unknown target transform: {self.kind}")

    def inverse(self, values):
        values = np.asarray(values, dtype=float)
        if self.kind == "identity":
            return values
        if self.kind == "asinh":
            return self.scale * np.sinh(values)
        if self.kind == "loggross":
            return 100 * (np.exp(values) - 1)
        raise ValueError(f"Unknown target transform: {self.kind}")


@dataclass
class Spec:
    name: str
    family: str
    include_year: bool
    target: TargetTransform
    params: dict
    fixed_iterations: bool = False
    weight_y_clip: tuple | None = None


def build_specs(full=False):
    cat_iters = 1200 if full else 700
    return [
        Spec(
            "xgb_raw_a_yr",
            "xgb",
            True,
            TargetTransform(),
            dict(
                n_estimators=1800,
                learning_rate=0.025,
                max_depth=4,
                min_child_weight=12,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=20,
                reg_alpha=0.05,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=2042,
                n_jobs=-1,
                eval_metric="rmse",
                early_stopping_rounds=120,
            ),
        ),
        Spec(
            "lgb_raw_a_yr",
            "lgb",
            True,
            TargetTransform(),
            dict(
                objective="regression",
                n_estimators=2500,
                learning_rate=0.025,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.05,
                reg_lambda=10,
                random_state=2040,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        Spec(
            "cat_raw_d7_yr",
            "cat",
            True,
            TargetTransform(),
            dict(iterations=1800, learning_rate=0.025, depth=7, l2_leaf_reg=16, random_seed=2028),
        ),
        Spec(
            "cat_clip500_d6_yr",
            "cat",
            True,
            TargetTransform("identity", -98, 500),
            dict(iterations=cat_iters, learning_rate=0.035, depth=6, l2_leaf_reg=8, random_seed=2029),
            fixed_iterations=True,
        ),
        Spec(
            "et_clip_yr",
            "et",
            True,
            TargetTransform("identity", -98, 600),
            dict(n_estimators=700, max_features=0.45, min_samples_leaf=12, random_state=2044, n_jobs=-1),
            fixed_iterations=True,
        ),
        Spec(
            "rf_clip_noyr",
            "rf",
            False,
            TargetTransform("identity", -98, 600),
            dict(n_estimators=500, max_features=0.45, min_samples_leaf=18, random_state=2045, n_jobs=-1),
            fixed_iterations=True,
        ),
        Spec(
            "hgb_clip_noyr",
            "hgb",
            False,
            TargetTransform("identity", -98, 600),
            dict(
                max_iter=800,
                learning_rate=0.035,
                max_leaf_nodes=31,
                l2_regularization=10,
                min_samples_leaf=35,
                random_state=2046,
                early_stopping=True,
            ),
            fixed_iterations=True,
        ),
        Spec(
            "cat_raw_d7_noyr",
            "cat",
            False,
            TargetTransform(),
            dict(iterations=1800, learning_rate=0.025, depth=7, l2_leaf_reg=16, random_seed=2028),
        ),
        Spec(
            "cat_raw_d6_noyr",
            "cat",
            False,
            TargetTransform(),
            dict(iterations=2200, learning_rate=0.03, depth=6, l2_leaf_reg=8, random_seed=2027),
        ),
        Spec(
            "cat_raw_d5_noyr",
            "cat",
            False,
            TargetTransform(),
            dict(iterations=2000, learning_rate=0.035, depth=5, l2_leaf_reg=12, random_seed=2026),
        ),
        Spec(
            "xgb_raw_a_noyr",
            "xgb",
            False,
            TargetTransform(),
            dict(
                n_estimators=1800,
                learning_rate=0.025,
                max_depth=4,
                min_child_weight=12,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=20,
                reg_alpha=0.05,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=2042,
                n_jobs=-1,
                eval_metric="rmse",
                early_stopping_rounds=120,
            ),
        ),
    ]


def prepare_feature_sets(train, test):
    feature_sets = {}
    for include_year in [True, False]:
        train_x, test_x, features, cat_features, numeric_features = add_features(train, test, include_year=include_year)
        feature_sets[include_year] = {
            "train_x": train_x,
            "test_x": test_x,
            "features": features,
            "cat_features": cat_features,
            "numeric_features": numeric_features,
        }
    return feature_sets


def make_numeric_frame(df, columns):
    return df[columns].replace([np.inf, -np.inf], np.nan)


def fit_predict_spec(spec, feature_pack, y_train, train_mask, valid_mask=None, final_fit=False, best_iterations=None):
    train_x = feature_pack["train_x"]
    test_x = feature_pack["test_x"]
    features = feature_pack["features"]
    cat_features = feature_pack["cat_features"]
    numeric_features = feature_pack["numeric_features"]

    if final_fit:
        train_idx = np.ones(len(train_x), dtype=bool)
        predict_frame = test_x
    else:
        train_idx = train_mask
        predict_frame = train_x.loc[valid_mask]

    y_fit = spec.target.transform(y_train[train_idx])
    model = None
    best_iteration = None

    if spec.family == "cat":
        params = dict(spec.params)
        if final_fit and best_iterations and spec.name in best_iterations:
            params["iterations"] = max(30, int(best_iterations[spec.name]) + 30)
        model = CatBoostRegressor(
            **params,
            loss_function=params.pop("loss_function", "RMSE") if "loss_function" in params else "RMSE",
            eval_metric="RMSE",
            random_strength=params.pop("random_strength", 0.7) if "random_strength" in params else 0.7,
            bagging_temperature=params.pop("bagging_temperature", 0.5) if "bagging_temperature" in params else 0.5,
            od_type="Iter",
            od_wait=150,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        train_pool = Pool(train_x.loc[train_idx, features], y_fit, cat_features=cat_features)
        if final_fit or spec.fixed_iterations:
            model.fit(train_pool, verbose=False)
        else:
            eval_pool = Pool(
                train_x.loc[valid_mask, features],
                spec.target.transform(y_train[valid_mask]),
                cat_features=cat_features,
            )
            model.fit(train_pool, eval_set=eval_pool, use_best_model=True, verbose=False)
            best_iteration = model.get_best_iteration()
        pred = model.predict(predict_frame[features])

    elif spec.family == "lgb":
        params = dict(spec.params)
        if final_fit and best_iterations and spec.name in best_iterations:
            params["n_estimators"] = max(30, int(best_iterations[spec.name]) + 30)
        model = lgb.LGBMRegressor(**params)
        x_fit = make_numeric_frame(train_x.loc[train_idx], numeric_features)
        if final_fit:
            model.fit(x_fit, y_fit)
        else:
            x_val = make_numeric_frame(train_x.loc[valid_mask], numeric_features)
            y_val = spec.target.transform(y_train[valid_mask])
            model.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
            )
            best_iteration = getattr(model, "best_iteration_", None)
        pred = model.predict(make_numeric_frame(predict_frame, numeric_features))

    elif spec.family == "xgb":
        params = dict(spec.params)
        if final_fit and best_iterations and spec.name in best_iterations:
            params.pop("early_stopping_rounds", None)
            params["n_estimators"] = max(30, int(best_iterations[spec.name]) + 30)
        model = XGBRegressor(**params)
        x_fit = make_numeric_frame(train_x.loc[train_idx], numeric_features)
        if final_fit:
            model.fit(x_fit, y_fit, verbose=False)
        else:
            x_val = make_numeric_frame(train_x.loc[valid_mask], numeric_features)
            y_val = spec.target.transform(y_train[valid_mask])
            model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
            best_iteration = getattr(model, "best_iteration", None)
        pred = model.predict(make_numeric_frame(predict_frame, numeric_features))

    elif spec.family == "et":
        model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(**spec.params))
        x_fit = make_numeric_frame(train_x.loc[train_idx], numeric_features)
        model.fit(x_fit, y_fit)
        pred = model.predict(make_numeric_frame(predict_frame, numeric_features))

    elif spec.family == "rf":
        model = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(**spec.params))
        x_fit = make_numeric_frame(train_x.loc[train_idx], numeric_features)
        model.fit(x_fit, y_fit)
        pred = model.predict(make_numeric_frame(predict_frame, numeric_features))

    elif spec.family == "hgb":
        model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(**spec.params))
        x_fit = make_numeric_frame(train_x.loc[train_idx], numeric_features)
        model.fit(x_fit, y_fit)
        pred = model.predict(make_numeric_frame(predict_frame, numeric_features))

    else:
        raise ValueError(f"Unknown model family: {spec.family}")

    pred = spec.target.inverse(pred)
    pred = np.clip(pred, -99.5, 1500)
    return pred, best_iteration


def greedy_blend(y_true, pred_frame, max_steps=24):
    names = list(pred_frame.columns)
    pred_matrix = pred_frame.values
    start_idx = int(np.argmin([rmse(y_true, pred_matrix[:, i]) for i in range(pred_matrix.shape[1])]))
    blend = pred_matrix[:, start_idx].copy()
    weights = np.zeros(pred_matrix.shape[1], dtype=float)
    weights[start_idx] = 1.0
    history = [
        {
            "step": 0,
            "added_model": names[start_idx],
            "added_weight": 1.0,
            "rmse": rmse(y_true, blend),
            "mae": mae(y_true, blend),
            "spearman": spearman(y_true, blend),
            "mean_pred": float(blend.mean()),
        }
    ]
    for step in range(1, max_steps + 1):
        current = rmse(y_true, blend)
        best = None
        for j in range(pred_matrix.shape[1]):
            for w in np.linspace(0.02, 0.60, 30):
                candidate = (1 - w) * blend + w * pred_matrix[:, j]
                score = rmse(y_true, candidate)
                if best is None or score < best[0]:
                    best = (score, j, w, candidate)
        if best is None or best[0] >= current - 1e-5:
            break
        score, model_idx, added_weight, blend = best
        weights = (1 - added_weight) * weights
        weights[model_idx] += added_weight
        history.append(
            {
                "step": step,
                "added_model": names[model_idx],
                "added_weight": float(added_weight),
                "rmse": score,
                "mae": mae(y_true, blend),
                "spearman": spearman(y_true, blend),
                "mean_pred": float(blend.mean()),
            }
        )
    weight_frame = pd.DataFrame({"model": names, "weight": weights}).query("weight > 1e-8")
    weight_frame = weight_frame.sort_values("weight", ascending=False).reset_index(drop=True)
    return weight_frame, pd.DataFrame(history), blend


def optimize_blend(y_true, pred_frame, start_weights):
    names = list(pred_frame.columns)
    pred_matrix = pred_frame.values.astype(float)
    start = np.zeros(len(names), dtype=float)
    for _, row in start_weights.iterrows():
        if row["model"] in names:
            start[names.index(row["model"])] = float(row["weight"])
    if start.sum() <= 0:
        start[:] = 1.0 / len(start)
    else:
        start /= start.sum()

    def objective(weights):
        pred = pred_matrix.dot(weights)
        return float(np.mean((y_true - pred) ** 2))

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    weights = np.asarray(result.x if result.success else start, dtype=float).clip(0)
    weights = weights / weights.sum()
    opt_blend = pred_matrix.dot(weights)

    affine_x = np.column_stack([opt_blend, np.ones(len(opt_blend))])
    affine_coef = np.linalg.lstsq(affine_x, y_true, rcond=None)[0]
    affine_blend = affine_x.dot(affine_coef)

    weight_frame = pd.DataFrame({"model": names, "weight": weights})
    weight_frame = weight_frame.query("weight > 1e-8").sort_values("weight", ascending=False).reset_index(drop=True)
    comparison = pd.DataFrame(
        [
            {
                "blend": "optimized",
                "rmse": rmse(y_true, opt_blend),
                "mae": mae(y_true, opt_blend),
                "spearman": spearman(y_true, opt_blend),
                "mean_pred": float(opt_blend.mean()),
                "std_pred": float(opt_blend.std()),
            },
            {
                "blend": "optimized_affine",
                "rmse": rmse(y_true, affine_blend),
                "mae": mae(y_true, affine_blend),
                "spearman": spearman(y_true, affine_blend),
                "mean_pred": float(affine_blend.mean()),
                "std_pred": float(affine_blend.std()),
                "affine_scale": float(affine_coef[0]),
                "affine_intercept": float(affine_coef[1]),
            },
        ]
    )
    return weight_frame, comparison, opt_blend, affine_blend, affine_coef


def run_solution(data_dir, output_dir, max_steps=24):
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    y = train[TARGET].values.astype(float)

    specs = build_specs(full=False)
    feature_sets = prepare_feature_sets(train, test)
    train_mask = train["start_year"].values < 2022
    valid_mask = train["start_year"].values == 2022
    y_valid = y[valid_mask]

    records = []
    valid_predictions = {}
    best_iterations = {}

    print(f"Loaded train={train.shape}, test={test.shape}")
    print(f"Validation: train years < 2022 ({train_mask.sum()} rows), valid year 2022 ({valid_mask.sum()} rows)")

    for spec in specs:
        spec_start = time.time()
        print(f"Training validation model: {spec.name}", flush=True)
        pred, best_iteration = fit_predict_spec(
            spec,
            feature_sets[spec.include_year],
            y,
            train_mask=train_mask,
            valid_mask=valid_mask,
            final_fit=False,
        )
        valid_predictions[spec.name] = pred
        if best_iteration is not None and best_iteration == best_iteration:
            best_iterations[spec.name] = int(best_iteration)
        records.append(
            {
                "model": spec.name,
                "family": spec.family,
                "include_year": spec.include_year,
                "rmse": rmse(y_valid, pred),
                "mae": mae(y_valid, pred),
                "spearman": spearman(y_valid, pred),
                "mean_pred": float(np.mean(pred)),
                "std_pred": float(np.std(pred)),
                "best_iteration": best_iteration,
                "seconds": round(time.time() - spec_start, 2),
            }
        )

    metrics = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
    valid_pred_frame = pd.DataFrame(valid_predictions, index=train.loc[valid_mask, ID_COL].values)
    weights, blend_history, valid_blend = greedy_blend(y_valid, valid_pred_frame, max_steps=max_steps)
    opt_weights, blend_comparison, opt_valid_blend, affine_valid_blend, affine_coef = optimize_blend(
        y_valid, valid_pred_frame, weights
    )

    metrics.to_csv(output_dir / "cv_summary.csv", index=False)
    valid_pred_frame.assign(
        actual_return_pct=y_valid,
        greedy_blend=valid_blend,
        optimized_blend=opt_valid_blend,
        optimized_affine_blend=affine_valid_blend,
    ).to_csv(output_dir / "valid_predictions.csv")
    weights.to_csv(output_dir / "blend_weights.csv", index=False)
    opt_weights.to_csv(output_dir / "blend_weights_optimized.csv", index=False)
    blend_history.to_csv(output_dir / "blend_history.csv", index=False)
    blend_comparison.to_csv(output_dir / "blend_comparison.csv", index=False)
    (output_dir / "best_iterations.json").write_text(json.dumps(best_iterations, indent=2), encoding="utf-8")

    print("Validation summary:")
    print(metrics.head(20).to_string(index=False))
    print("Blend weights:")
    print(weights.to_string(index=False))
    print("Blend validation:", blend_history.tail(1).to_dict("records")[0])
    print("Optimized blend weights:")
    print(opt_weights.to_string(index=False))
    print("Optimized blend comparison:")
    print(blend_comparison.to_string(index=False))

    final_specs = build_specs(full=True)
    spec_by_name = {spec.name: spec for spec in final_specs}
    final_predictions = {}
    final_feature_sets = prepare_feature_sets(train, test)

    needed_models = sorted(set(weights["model"]).union(set(opt_weights["model"])))
    for name in needed_models:
        spec = spec_by_name[name]
        print(f"Training final model: {name}", flush=True)
        pred, _ = fit_predict_spec(
            spec,
            final_feature_sets[spec.include_year],
            y,
            train_mask=None,
            valid_mask=None,
            final_fit=True,
            best_iterations=best_iterations,
        )
        final_predictions[name] = pred

    final_pred_frame = pd.DataFrame(final_predictions)
    greedy_final_blend = np.zeros(len(test), dtype=float)
    for _, row in weights.iterrows():
        greedy_final_blend += float(row["weight"]) * final_pred_frame[row["model"]].values
    greedy_final_blend = np.clip(greedy_final_blend, -99.5, 1500)

    optimized_final_blend = np.zeros(len(test), dtype=float)
    for _, row in opt_weights.iterrows():
        optimized_final_blend += float(row["weight"]) * final_pred_frame[row["model"]].values
    optimized_final_blend = np.clip(optimized_final_blend, -99.5, 1500)
    affine_final_blend = np.clip(affine_coef[0] * optimized_final_blend + affine_coef[1], -99.5, 1500)

    submission = sample.copy()
    submission[TARGET] = affine_final_blend
    submission.to_csv(output_dir / "submission.csv", index=False)
    final_pred_frame.assign(
        greedy_blend=greedy_final_blend,
        optimized_blend=optimized_final_blend,
        optimized_affine_blend=affine_final_blend,
        id=test[ID_COL].values,
    ).to_csv(output_dir / "test_model_predictions.csv", index=False)

    greedy_submission = sample.copy()
    greedy_submission[TARGET] = greedy_final_blend
    greedy_submission.to_csv(output_dir / "submission_greedy_backup.csv", index=False)

    optimized_submission = sample.copy()
    optimized_submission[TARGET] = optimized_final_blend
    optimized_submission.to_csv(output_dir / "submission_optimized_blend.csv", index=False)

    # A lower-variance backup can be useful if the hidden year is more like a noisy market baseline.
    safe_submission = sample.copy()
    safe_submission[TARGET] = np.clip(0.85 * affine_final_blend + 0.15 * np.median(train[TARGET]), -99.5, 1000)
    safe_submission.to_csv(output_dir / "submission_conservative_backup.csv", index=False)

    run_info = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "validation_year": 2022,
        "best_single_model": metrics.iloc[0].to_dict(),
        "greedy_blend_validation": blend_history.tail(1).to_dict("records")[0],
        "optimized_blend_validation": blend_comparison.to_dict("records"),
        "submission_rows": int(len(submission)),
        "submission_mean": float(submission[TARGET].mean()),
        "submission_std": float(submission[TARGET].std()),
        "submission_min": float(submission[TARGET].min()),
        "submission_max": float(submission[TARGET].max()),
        "seconds_total": round(time.time() - start, 2),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print("Done.")
    print(json.dumps(run_info, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DESKTOP)
    parser.add_argument("--max-blend-steps", type=int, default=24)
    args = parser.parse_args()
    run_solution(args.data_dir, args.output_dir, max_steps=args.max_blend_steps)


if __name__ == "__main__":
    main()
