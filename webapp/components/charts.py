"""Pure echarts option builders. Every option obeys the dataviz rules: emphasis form
(subject in s1 blue, context in de-emphasis gray), solid hairline grid, connectNulls
false (gaps are data), no dual axes, status colors only where value truly means bad."""
import pandas as pd

from webapp.theme import TOKENS


def _base(mode):
    t = TOKENS[mode]
    return t, {
        "backgroundColor": "transparent",
        "grid": {"left": 48, "right": 90, "top": 24, "bottom": 32},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
    }


def fund_vs_peers_option(mode, quarters, fund_vals, median_vals, fund_label):
    t, opt = _base(mode)
    opt["xAxis"] = {"type": "category", "data": quarters,
                    "axisLine": {"lineStyle": {"color": t["grid"]}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "value",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"], "formatter": "{value}"}}
    opt["series"] = [
        {"name": fund_label, "type": "line", "data": fund_vals, "connectNulls": False,
         "lineStyle": {"width": 2, "color": t["s1"]}, "itemStyle": {"color": t["s1"]},
         "symbolSize": 5, "endLabel": {"show": True, "color": t["ink"]}},
        {"name": "Peer median", "type": "line", "data": median_vals, "connectNulls": False,
         "lineStyle": {"width": 2, "color": t["demph"]}, "itemStyle": {"color": t["demph"]},
         "symbolSize": 4, "endLabel": {"show": True, "color": t["ink2"]}},
    ]
    return opt


def diverging_delta_option(mode, quarters, deltas):
    t, opt = _base(mode)
    opt["grid"]["top"] = 8
    opt["xAxis"] = {"type": "category", "data": quarters, "axisLabel": {"show": False},
                    "axisLine": {"lineStyle": {"color": t["grid"]}}}
    opt["yAxis"] = {"type": "value",
                    "splitLine": {"show": False},
                    "axisLabel": {"color": t["ink2"]}}
    opt["series"] = [{
        "type": "bar", "barMaxWidth": 14,
        "data": [{"value": d,
                  "itemStyle": {"color": t["div_pos"] if (d or 0) >= 0 else t["div_neg"]}}
                 for d in deltas],
        "markLine": {"silent": True, "symbol": "none",
                     "lineStyle": {"color": t["muted"], "type": "solid"},
                     "data": [{"yAxis": 0}], "label": {"show": False}},
    }]
    return opt


def auc_by_quarter_option(mode, df: pd.DataFrame):
    t, opt = _base(mode)
    df = df[df["source"] == "retrained"].sort_values("quarter") if "source" in df else df
    opt["xAxis"] = {"type": "category", "data": list(df["quarter"]),
                    "axisLine": {"lineStyle": {"color": t["grid"]}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "value", "min": 0.3, "max": 0.8,
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    model_pts = [{"value": float(a),
                  "itemStyle": {"color": t["critical"] if a < 0.5 else t["s1"]}}
                 for a in df["auc"]]
    opt["series"] = [
        {"name": "Model", "type": "line", "data": model_pts, "connectNulls": False,
         "lineStyle": {"width": 2, "color": t["s1"]}, "symbolSize": 7,
         "endLabel": {"show": True, "color": t["ink"]},
         "markLine": {"silent": True, "symbol": "none",
                      "lineStyle": {"color": t["ink2"], "type": "solid", "width": 1},
                      "data": [{"yAxis": 0.5,
                                "label": {"formatter": "coin flip (0.5)",
                                          "color": t["ink2"]}}]}},
        {"name": "Mean-reversion rule", "type": "line",
         "data": [float(v) if pd.notna(v) else None for v in df["persistence_auc"]],
         "connectNulls": False, "lineStyle": {"width": 2, "color": t["demph"]},
         "itemStyle": {"color": t["demph"]}, "symbolSize": 4,
         "endLabel": {"show": True, "color": t["ink2"]}},
    ]
    return opt


def dumbbell_option(mode, rows):
    """rows = [(label, before, after)] -> horizontal dumbbell, one hue two shades."""
    t, opt = _base(mode)
    labels = [r[0] for r in rows]
    opt["tooltip"] = {"trigger": "item"}
    opt["xAxis"] = {"type": "value", "min": 0.4, "max": 0.8,
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "category", "data": labels, "axisLabel": {"color": t["ink"]}}
    opt["series"] = [
        {"name": "before", "type": "scatter", "symbolSize": 10,
         "itemStyle": {"color": t["seq"][2]},
         "data": [[r[1], i] for i, r in enumerate(rows)]},
        {"name": "after", "type": "scatter", "symbolSize": 10,
         "itemStyle": {"color": t["seq"][5]},
         "data": [[r[2], i] for i, r in enumerate(rows)]},
        {"name": "gap", "type": "lines", "coordinateSystem": "cartesian2d",
         "lineStyle": {"color": t["seq"][3], "width": 2},
         "data": [{"coords": [[r[1], i], [r[2], i]]} for i, r in enumerate(rows)]},
    ]
    return opt


def histogram_option(mode, values, bins=20):
    t, opt = _base(mode)
    counts = [0] * bins
    for v in values:
        counts[max(0, min(bins - 1, int(float(v) * bins)))] += 1
    opt["tooltip"] = {"trigger": "item"}
    opt["xAxis"] = {"type": "category",
                    "data": [f"{i / bins:.0%}" for i in range(bins)],
                    "axisLabel": {"color": t["ink2"], "interval": 4},
                    "axisLine": {"lineStyle": {"color": t["grid"]}}}
    opt["yAxis"] = {"type": "value",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["series"] = [{"type": "bar", "data": counts, "barCategoryGap": "10%",
                      "itemStyle": {"color": t["seq"][4]}}]
    return opt


def calibration_option(mode, df: pd.DataFrame, base_rate: float):
    t, opt = _base(mode)
    opt["tooltip"] = {"trigger": "item"}
    opt["xAxis"] = {"type": "value", "min": 0, "max": 1, "name": "predicted",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "value", "min": 0, "max": 1, "name": "actual lag rate",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["series"] = [
        {"name": "bins", "type": "scatter",
         "symbolSize": [max(6, min(20, int(n ** 0.5))) for n in df["n"]],
         "itemStyle": {"color": t["s1"]},
         "data": df[["predicted_mean", "actual_lag_rate"]].values.tolist()},
        {"name": "perfect", "type": "line", "data": [[0, 0], [1, 1]], "symbol": "none",
         "lineStyle": {"color": t["demph"], "type": "solid", "width": 1},
         "markLine": {"silent": True, "symbol": "none",
                      "lineStyle": {"color": t["muted"], "type": "solid"},
                      "data": [{"yAxis": base_rate,
                                "label": {"formatter": f"base rate {base_rate:.0%}",
                                          "color": t["ink2"]}}]}},
    ]
    return opt
