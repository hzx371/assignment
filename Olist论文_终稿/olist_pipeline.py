# -*- coding: utf-8 -*-
"""
olist_pipeline.py —— Olist 电商用户价值与留存分析：一键复现管线
=================================================================
输入 : data/raw/ 下 9 个 Olist 官方 CSV（Kaggle 官方下载）
输出 : outputs/tables/metrics.json     全部指标
       outputs/figures/fig28_*.png ... fig33_*.png   业务模型图
运行 :  python olist_pipeline.py --raw "data/raw" --out "outputs"
说明 :  全部数值由原始数据计算，随机种子固定为 42，结果可复现。
"""
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200

CUTOFF = pd.Timestamp("2018-08-31")     # 分析截止日（2018-09/10 交付信息不完整，剔除）
WIN_START = pd.Timestamp("2017-01-01")  # 主分析窗口起点
SEED = 42
np.random.seed(SEED)

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
    "trans": "product_category_name_translation.csv",
}


# ---------------------------------------------------------------- 01 读取
def load_raw(raw_dir):
    rd = lambda k, **kw: pd.read_csv(os.path.join(raw_dir, FILES[k]), **kw)
    df = {
        "orders": rd("orders", parse_dates=[
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date"]),
        "items": rd("items"), "customers": rd("customers"),
        "payments": rd("payments"),
        "reviews": rd("reviews", parse_dates=["review_creation_date"]),
        "products": rd("products"), "sellers": rd("sellers"),
        "trans": rd("trans"),
        "geo": rd("geo", usecols=["geolocation_zip_code_prefix",
                                  "geolocation_lat", "geolocation_lng"]),
    }
    return df


# ---------------------------------------------------------------- 02 清洗
def clean(df):
    orders, products, reviews, customers = (df["orders"], df["products"],
                                            df["reviews"], df["customers"])
    # 类型统一：邮编前缀保持字符串，避免前导零丢失
    customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].astype(str)
    # 缺失：品类名缺失填充 unknown（缺失 610 个）
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    # 重复：评价表同一 order_id 保留最新一条
    reviews = (reviews.sort_values(["order_id", "review_creation_date"])
               .drop_duplicates("order_id", keep="last"))
    # 逻辑异常：送达早于下单
    mask_bad = (orders["order_delivered_customer_date"] < orders["order_purchase_timestamp"])
    orders.loc[mask_bad, "order_delivered_customer_date"] = pd.NaT
    df.update(orders=orders, products=products, reviews=reviews, customers=customers)
    return df


# ---------------------------------------------------------------- 03 建宽表
def build_fact(df):
    items, products, trans = df["items"], df["products"], df["trans"]
    it = items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    it = it.merge(trans, on="product_category_name", how="left")
    it["product_category_name_english"] = it["product_category_name_english"].fillna("unknown")
    item_ord = it.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        gmv_items=("price", "sum"), freight=("freight_value", "sum"))
    pay_ord = df["payments"].groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0]))
    od = (df["orders"]
          .merge(df["customers"], on="customer_id", how="left")
          .merge(item_ord, on="order_id", how="left")
          .merge(pay_ord, on="order_id", how="left")
          .merge(df["reviews"][["order_id", "review_score"]], on="order_id", how="left"))
    od["item_count"] = od["item_count"].fillna(1)
    od["order_total_value"] = od["gmv_items"].fillna(0) + od["freight"].fillna(0)
    od["delivery_days"] = (od["order_delivered_customer_date"]
                           - od["order_purchase_timestamp"]).dt.days
    od["delay_days"] = (od["order_delivered_customer_date"]
                        - od["order_estimated_delivery_date"]).dt.days
    od["is_late"] = (od["delay_days"] > 0).astype("Int64")
    od["purchase_ym"] = od["order_purchase_timestamp"].dt.to_period("M")
    return od, it


# ---------------------------------------------------------------- 04 RFM
SEG_MAP = {(True, True, True): "重要价值客户", (False, True, True): "重要保持客户",
           (True, False, True): "重要发展客户", (False, False, True): "重要挽留客户",
           (True, True, False): "一般价值客户", (False, True, False): "一般保持客户",
           (True, False, False): "一般发展客户", (False, False, False): "一般挽留客户"}


def rfm(w, cutoff=CUTOFF):
    c = w.groupby("customer_unique_id").agg(
        last=("order_purchase_timestamp", "max"), F=("order_id", "nunique"),
        M=("order_total_value", "sum"), first=("order_purchase_timestamp", "min"))
    c["R"] = (cutoff - c["last"]).dt.days
    r_med, m_med = c["R"].median(), c["M"].median()
    c["R_high"], c["F_high"], c["M_high"] = c["R"] >= r_med, c["F"] >= 2, c["M"] >= m_med
    c["seg"] = [SEG_MAP[(a, b, d)] for a, b, d in zip(c.R_high, c.F_high, c.M_high)]
    return c, r_med, m_med


def rfm_summary(c):
    s = c.groupby("seg").agg(customers=("M", "size"), revenue=("M", "sum"))
    s["cus_share"] = 100 * s.customers / s.customers.sum()
    s["rev_share"] = 100 * s.revenue / s.revenue.sum()
    s["arpu"] = s.revenue / s.customers
    return s.sort_values("revenue", ascending=False)


# ---------------------------------------------------------------- 05 同期群 / 二单间隔
def cohort_matrix(w, c, end=pd.Period("2018-08", "M"), kmax=12):
    """返回 (留存矩阵, 各同期群样本量, 样本量加权的留存率)。"""
    c = c.copy(); c["cohort"] = c["first"].dt.to_period("M")
    o = w.merge(c[["cohort"]], left_on="customer_unique_id", right_index=True, how="left")
    o["k"] = (o["purchase_ym"].astype("int64") - o["cohort"].astype("int64"))
    size = c.groupby("cohort").size()
    act = o[o.k >= 0].groupby(["cohort", "k"]).customer_unique_id.nunique()
    mat = pd.DataFrame(index=size.index, columns=range(kmax + 1), dtype=float)
    for coh in size.index:
        maxk = end.ordinal - coh.ordinal
        for k in range(min(kmax, maxk) + 1):
            mat.loc[coh, k] = 100 * act.get((coh, k), 0) / size[coh]
    weighted = {}
    for k in range(kmax + 1):
        num = den = 0
        for coh in size.index:
            if end.ordinal - coh.ordinal >= k and not np.isnan(mat.loc[coh, k]):
                num += act.get((coh, k), 0); den += size[coh]
        weighted[k] = round(100 * num / den, 2) if den else None
    return mat, size, weighted


def repurchase_gap(w):
    """首购至第二单的间隔天数（仅复购客户）。"""
    o = w.sort_values(["customer_unique_id", "order_purchase_timestamp"]).copy()
    o["rk"] = o.groupby("customer_unique_id").cumcount()
    f1 = o[o.rk == 0][["customer_unique_id", "order_purchase_timestamp"]].rename(
        columns={"order_purchase_timestamp": "t1"})
    s2 = o[o.rk == 1][["customer_unique_id", "order_purchase_timestamp"]].rename(
        columns={"order_purchase_timestamp": "t2"})
    g = s2.merge(f1, on="customer_unique_id", how="inner")
    gap = (g["t2"] - g["t1"]).dt.total_seconds() / 86400
    return gap, o


# ---------------------------------------------------------------- 06 帕累托
def pareto(c):
    s = c["M"].sort_values(ascending=False).reset_index(drop=True)
    cum = s.cumsum() / s.sum()
    return s, cum


# ---------------------------------------------------------------- 07 预测
def forecast_arima(y, steps_out=3):
    """SARIMA(1,1,1)(0,0,0)12 即 ARIMA(1,1,1)：季节项因样本不足不保留。"""
    import statsmodels.api as sm
    kw = dict(order=(1, 1, 1), seasonal_order=(0, 0, 0, 12),
              enforce_stationarity=False, enforce_invertibility=False)
    ytr, yte = y[:-6], y[-6:]
    r = sm.tsa.statespace.SARIMAX(ytr, **kw).fit(disp=False)
    f = r.get_forecast(6)
    pred = np.asarray(f.predicted_mean, float)
    ci = np.asarray(f.conf_int(alpha=.05), float).reshape(-1, 2)
    mape = float(np.mean(np.abs((yte - pred) / yte)) * 100)
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    full = sm.tsa.statespace.SARIMAX(y, **kw).fit(disp=False)
    f3 = full.get_forecast(steps_out)
    return dict(pred=pred, ci=ci, mape=mape, rmse=rmse, aic=float(r.aic),
                full=full, point=np.asarray(f3.predicted_mean, float),
                ci3=np.asarray(f3.conf_int(alpha=.05), float).reshape(-1, 2))


# ---------------------------------------------------------------- 08 图
def fig_rfm_scatter(c, path):
    col = {"重要价值客户": "#c0392b", "重要保持客户": "#e67e22", "重要发展客户": "#2980b9",
           "重要挽留客户": "#16a085", "一般价值客户": "#8e44ad", "一般保持客户": "#f1c40f",
           "一般发展客户": "#7f8c8d", "一般挽留客户": "#bdc3c7"}
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for seg, sub in c.groupby("seg"):
        sub = sub.sample(min(len(sub), 1500), random_state=SEED)
        ax.scatter(sub["R"], sub["M"], s=6, alpha=.55, c=col[seg], label=seg, linewidths=0)
    ax.set_yscale("log"); ax.set_xlabel("R：最近一次购买距截止日天数（天）")
    ax.set_ylabel("M：客户累计消费金额（雷亚尔，对数轴）")
    # 图题在 Word 中以题注给出，图内不重复
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False)
    ax.grid(alpha=.25, ls=":")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_rfm_revenue(s, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(s))
    ax.barh(y, s["rev_share"], color="#2980b9", label="收入占比")
    ax.barh(y, s["cus_share"], height=.38, color="#e67e22", label="客户数占比")
    ax.set_yticks(y); ax.set_yticklabels(s.index, fontsize=9); ax.invert_yaxis()
    for i, (r, cc) in enumerate(zip(s["rev_share"], s["cus_share"])):
        ax.text(r + .6, i - .18, f"{r:.2f}%", va="center", fontsize=7.5, color="#1a5276")
        ax.text(cc + .6, i + .18, f"{cc:.2f}%", va="center", fontsize=7.5, color="#a04000")
    ax.set_xlabel("占比（%）"); ax.legend(fontsize=8, frameon=False)

    ax.grid(axis="x", alpha=.25, ls=":")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_cohort(mat, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    data = mat.iloc[:, 1:].values.astype(float)          # 第 0 列恒为 100%，不参与配色
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0,
                   vmax=float(np.nanpercentile(data, 99)))
    ax.set_yticks(range(len(mat))); ax.set_yticklabels([str(i) for i in mat.index], fontsize=7)
    ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(range(1, data.shape[1] + 1), fontsize=8)
    ax.set_xlabel("首购后第 k 个月"); ax.set_ylabel("同期群（首次购买月份）")
    thr = float(np.nanpercentile(data, 70))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > thr else "#333")
    fig.colorbar(im, ax=ax, label="留存率（%）", fraction=.025)

    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_pareto(s, cum, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(1, len(s) + 1) / len(s) * 100
    ax.plot(x, cum.values * 100, color="#1f618d", lw=1.8)
    ax.plot([0, 100], [0, 100], ls="--", c="grey", lw=1, label="绝对平均线")
    for p, c_ in ((70, "#c0392b"), (90, "#7d3c98")):
        ax.axhline(p, ls=":", c=c_, lw=1)
        ax.text(2, p + 1.2, f"累计收入 {p}%", color=c_, fontsize=8)
    ax.set_xlabel("客户累计占比（%，按消费金额降序）"); ax.set_ylabel("累计收入占比（%）")

    ax.grid(alpha=.25, ls=":"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_forecast(ym, y, res, path):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    n = len(y); axis = np.arange(n)
    ax.plot(axis, y, "-o", ms=3.5, c="#1f618d", label="实际订单量")
    ti = axis[-6:]
    ax.plot(ti, res["pred"], "--s", ms=3.5, c="#c0392b", label="回测预测（6 个月）")
    ax.fill_between(ti, res["ci"][:, 0], res["ci"][:, 1], color="#c0392b", alpha=.15,
                    label="95% 置信区间")
    fi = np.arange(n, n + len(res["point"]))
    ax.plot(fi, res["point"], "--^", ms=4, c="#16a085", label="未来预测")
    ax.fill_between(fi, res["ci3"][:, 0], res["ci3"][:, 1], color="#16a085", alpha=.18)
    labels = list(ym) + ["2018-09", "2018-10", "2018-11"]
    ticks = list(range(0, n, 2)) + list(fi)
    ax.set_xticks(ticks); ax.set_xticklabels([labels[i] for i in ticks], rotation=45,
                                             fontsize=7, ha="right")
    ax.set_ylabel("月度订单量（单）")

    ax.grid(alpha=.25, ls=":"); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_delay_dist(d, path):
    """评价得分为 1—5 的定序变量，用 100% 堆叠条形图比箱线图更能体现分布差异。"""
    colors = ["#922b21", "#cd6155", "#f0b27a", "#7fb3d5", "#1f618d"]
    rows, n = [], []
    for flag in (0, 1):
        sub = d.loc[d.is_late == flag, "review_score"].dropna()
        n.append(len(sub))
        vc = sub.value_counts(normalize=True) * 100
        rows.append([float(vc.get(s, 0)) for s in (1, 2, 3, 4, 5)])
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    left = np.zeros(2)
    for j, s in enumerate((1, 2, 3, 4, 5)):
        ax.barh([0, 1], rows[:, j], left=left, color=colors[j], label=f"{s} 分", height=.5)
        for i in range(2):
            if rows[i, j] > 4:
                ax.text(left[i] + rows[i, j] / 2, i, f"{rows[i, j]:.1f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if j in (0, 4) else "#222")
        left += rows[:, j]
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"准时交付\n(n={n[0]:,})", f"延迟交付\n(n={n[1]:,})"], fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("评价得分构成（%）")
    ax.legend(fontsize=7, ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, 1.28))

    ax.grid(axis="x", alpha=.2, ls=":")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main(raw_dir, out_dir):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    df = clean(load_raw(raw_dir))
    od, it = build_fact(df)

    w = od[(od.order_purchase_timestamp >= WIN_START) &
           (od.order_purchase_timestamp <= CUTOFF)].copy()
    c, r_med, m_med = rfm(w)
    s = rfm_summary(c)
    mat, size, wret = cohort_matrix(w, c)
    gap, _ = repurchase_gap(w)
    sr, cum = pareto(c)

    m = {
        "raw_rows": {k: len(df[k]) for k in df},
        "window": {"orders": len(w), "customers": int(w.customer_unique_id.nunique()),
                   "gmv": round(float(w.order_total_value.sum()), 2),
                   "aov": round(float(w.order_total_value.mean()), 2)},
        "full": {"orders": len(od), "customers": int(od.customer_unique_id.nunique()),
                 "gmv": round(float(od.order_total_value.sum()), 2),
                 "states": int(df["customers"].customer_state.nunique())},
        "rfm_threshold": {"r_median": float(r_med), "m_median": round(float(m_med), 2),
                          "customers": int(len(c))},
        "rfm_segments": s.round(4).reset_index().to_dict("records"),
        "one_vs_repeat": {
            "one_share": round(100 * float((c.F == 1).mean()), 2),
            "one_rev_share": round(100 * float(c.loc[c.F == 1, "M"].sum() / c.M.sum()), 2),
            "rep_share": round(100 * float((c.F >= 2).mean()), 2),
            "rep_rev_share": round(100 * float(c.loc[c.F >= 2, "M"].sum() / c.M.sum()), 2),
            "rep_arpu": round(float(c.loc[c.F >= 2, "M"].mean()), 2),
            "one_arpu": round(float(c.loc[c.F == 1, "M"].mean()), 2),
        },
        "cohort_retention": {f"m{k}": wret[k] for k in (1, 3, 6, 12)},
        "gap": {"n": int(gap.notna().sum()), "median": round(float(gap.median()), 1),
                "mean": round(float(gap.mean()), 1),
                "w30": round(100 * float((gap <= 30).mean()), 2),
                "w60": round(100 * float((gap <= 60).mean()), 2),
                "w90": round(100 * float((gap <= 90).mean()), 2),
                "w180": round(100 * float((gap <= 180).mean()), 2)},
        "pareto": {f"share_{p}": round(100 * float(((cum <= p / 100).sum() + 1) / len(cum)), 2)
                   for p in (70, 80, 90)},
        "category": {"n": int(it.product_category_name_english.nunique()),
                     "top10_share": round(100 * float(
                         it.groupby("product_category_name_english").price.sum()
                         .nlargest(10).sum() / it.price.sum()), 2)},
    }

    ym = [str(p) for p in pd.period_range("2017-01", "2018-08", freq="M")]
    y = w.groupby("purchase_ym").order_id.nunique().reindex(
        pd.period_range("2017-01", "2018-08", freq="M")).values.astype(float)
    res = forecast_arima(y)
    m["forecast"] = {"order": [1, 1, 1], "seasonal": [0, 0, 0, 12],
                     "mape": round(res["mape"], 2), "rmse": round(res["rmse"], 1),
                     "aic": round(res["aic"], 2),
                     "point": [int(round(v)) for v in res["point"]],
                     "low": [int(round(v)) for v in res["ci3"][:, 0]],
                     "high": [int(round(v)) for v in res["ci3"][:, 1]]}
    m["delivery"] = {"late_rate": round(100 * float(
        w.loc[(w.order_status == "delivered") & w.delivery_days.notna() & (w.delivery_days >= 0),
              "is_late"].mean()), 2)}
    dl0 = w[(w.order_status == "delivered") & w.delivery_days.notna() & (w.delivery_days >= 0)]
    for flag, key in ((0, "on_time"), (1, "late")):
        sub = dl0.loc[dl0.is_late == flag, "review_score"].dropna()
        m["delivery"][f"{key}_n"] = int(len(sub))
        m["delivery"][f"{key}_mean"] = round(float(sub.mean()), 3)
        m["delivery"][f"{key}_median"] = float(sub.median())
        m["delivery"][f"{key}_low"] = round(100 * float((sub <= 2).mean()), 2)
        m["delivery"][f"{key}_five"] = round(100 * float((sub == 5).mean()), 2)
    m["delivery"]["mean_days"] = round(float(dl0.delivery_days.mean()), 2)
    m["delivery"]["median_days"] = float(dl0.delivery_days.median())
    m["delivery"]["p90_days"] = float(dl0.delivery_days.quantile(.9))
    m["delivery"]["late_delay_mean"] = round(float(dl0.loc[dl0.is_late == 1, "delay_days"].mean()), 2)

    fig_rfm_scatter(c, os.path.join(out_dir, "figures", "fig28_rfm_scatter.png"))
    fig_rfm_revenue(s, os.path.join(out_dir, "figures", "fig29_rfm_segment_revenue.png"))
    fig_cohort(mat, os.path.join(out_dir, "figures", "fig30_cohort_heatmap.png"))
    fig_pareto(sr, cum, os.path.join(out_dir, "figures", "fig31_pareto_customer.png"))
    fig_forecast(ym, y, res, os.path.join(out_dir, "figures", "fig32_sarima_forecast.png"))
    dl = w[(w.order_status == "delivered") & w.delivery_days.notna() & (w.delivery_days >= 0)]
    fig_delay_dist(dl, os.path.join(out_dir, "figures", "fig33_delay_score_dist.png"))

    with open(os.path.join(out_dir, "tables", "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=1)
    print(json.dumps(m, ensure_ascii=False, indent=1)[:2000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    main(a.raw, a.out)
