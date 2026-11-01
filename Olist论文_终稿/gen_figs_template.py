# -*- coding: utf-8 -*-
"""
gen_figs_template.py —— 用 Olist 真实数据复现《Excel 数据可视化》27 种图表模板
输出：outputs/figures/fig01_*.png … fig27_*.png（200 dpi，无图内标题，题注在 Word 中）
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Wedge, Circle
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olist_pipeline import load_raw, clean, build_fact, rfm, WIN_START, CUTOFF

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#7f8c8d"
plt.rcParams["axes.linewidth"] = .8
plt.rcParams["xtick.color"] = "#444"
plt.rcParams["ytick.color"] = "#444"

RAW = r"C:\Users\Administrator\Desktop\archive(1)"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, DBLUE, ORANGE, RED, GREEN, GREY = ("#2980b9", "#1a5276", "#e67e22",
                                         "#c0392b", "#27ae60", "#95a5a6")
FS = 8.5          # 图内字号（按 6.4in 宽设计）

# ------------------------------------------------------------------ 数据
df = clean(load_raw(RAW))
od, it = build_fact(df)
w = od[(od.order_purchase_timestamp >= WIN_START) &
       (od.order_purchase_timestamp <= CUTOFF)].copy()
cust, r_med, m_med = rfm(w)

mon = (w.groupby("purchase_ym")
       .agg(orders=("order_id", "nunique"), gmv=("order_total_value", "sum")))
mon.index = [str(p) for p in mon.index]
m_ord, m_gmv = mon["orders"].values, mon["gmv"].values
ym = list(mon.index)

dl = w[(w.order_status == "delivered") & w.delivery_days.notna() & (w.delivery_days >= 0)]
st = (w.groupby("customer_state")
      .agg(orders=("order_id", "nunique"), days=("delivery_days", "mean"))
      .sort_values("orders", ascending=False))
st10 = st.head(10)

cat = (it.groupby("product_category_name_english").price.sum()
       .sort_values(ascending=False))
cat = cat[cat.index != "unknown"]
cat8 = cat.head(8)

pay = w.payment_type.value_counts(normalize=True) * 100
late_m = dl.groupby(dl.order_purchase_timestamp.dt.to_period("M")).is_late.mean() * 100
late_m.index = [str(p) for p in late_m.index]

y17 = pd.Series(m_ord, index=pd.PeriodIndex(ym, freq="M"))["2017"].values
y18 = pd.Series(m_ord, index=pd.PeriodIndex(ym, freq="M"))["2018"].values
cat_rev17 = (w[w.order_purchase_timestamp.dt.year == 2017]
             .merge(it[["order_id", "price"]].groupby("order_id").price.sum(),
                    on="order_id", how="left").groupby("customer_state").order_id.nunique())
seg = (cust.groupby("seg").agg(n=("M", "size"), rev=("M", "sum")))
seg["rev_share"] = 100 * seg.rev / seg.rev.sum()
seg = seg.sort_values("rev_share", ascending=False)


def save(fig, name):
    fig.tight_layout(pad=.4)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def grad_bar(ax, x, h, width=.62, cmap="Blues", lo=.35, hi=1.0, zorder=2):
    g = np.linspace(0, 1, 256).reshape(-1, 1)
    for xi, hi_ in zip(x, h):
        rect = Rectangle((xi - width / 2, 0), width, hi_, facecolor="none",
                         edgecolor="none")
        ax.add_patch(rect)
        im = ax.imshow(g, extent=[xi - width / 2, xi + width / 2, 0, hi_],
                       aspect="auto", cmap=cmap, vmin=lo, vmax=hi, zorder=zorder)
        im.set_clip_path(rect)


def newax(h=3.4, wid=6.4):
    return plt.subplots(figsize=(wid, h))


# ------------------------------------------------------------ 01 渐变柱形图
def fig01():
    fig, ax = newax()
    x = np.arange(len(m_gmv[:12]))
    grad_bar(ax, x, m_gmv[:12] / 1e4, cmap="Blues")
    ax.set_xticks(x); ax.set_xticklabels(ym[:12], rotation=45, fontsize=FS - .5)
    ax.set_ylabel("GMV（万雷亚尔）", fontsize=FS)
    for xi, v in zip(x, m_gmv[:12] / 1e4):
        ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=FS - 1.5, color=DBLUE)
    ax.set_ylim(0, (m_gmv[:12] / 1e4).max() * 1.18)
    ax.set_title("2017 年月度 GMV（渐变填充）", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig01_gradient_bar.png")


# ------------------------------------------------------------ 02 带均值柱形
def fig02():
    fig, ax = newax()
    x = np.arange(len(st10))
    ax.bar(x, st10.orders.values, .6, color=BLUE, edgecolor="white")
    m = st10.orders.mean()
    ax.axhline(m, ls="--", color=RED, lw=1.4)
    ax.text(len(st10) - .5, m * 1.04, f"平均值 {m:,.0f} 单", color=RED, fontsize=FS,
            ha="right")
    ax.set_xticks(x); ax.set_xticklabels(st10.index, fontsize=FS)
    ax.set_ylabel("订单量（单）", fontsize=FS)
    ax.set_title("TOP10 州订单量与平均值", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig02_mean_bar.png")


# ------------------------------------------------------------ 03 渐变圆角柱形
def fig03():
    fig, ax = newax(3.4)
    v = cat8.values / 1e4
    y = np.arange(len(v))[::-1]
    cmap = plt.get_cmap("Greens")
    for i, (yi, vi) in enumerate(zip(y, v)):
        col = cmap(.35 + .55 * i / len(v))
        # 用圆端粗线模拟“圆角条形”，避免数据坐标比例不一致导致圆角失真
        ax.plot([0, vi], [yi, yi], lw=15, solid_capstyle="round", color=col, zorder=2)
        ax.annotate(f"{vi:.0f}", (vi, yi), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=FS - 1, color="#1e8449")
    ax.set_yticks(y); ax.set_yticklabels([c[:22] for c in cat8.index], fontsize=FS - .5)
    ax.set_xlabel("品类收入（万雷亚尔）", fontsize=FS)
    ax.set_xlim(-2, v.max() * 1.18); ax.set_ylim(-.7, len(v) - .3)
    ax.set_title("TOP8 品类收入（渐变圆角条形）", fontsize=FS + 1, color="#333")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "fig03_rounded_gradient_bar.png")


# ------------------------------------------------------------ 04 标注柱形图
def fig04():
    fig, ax = newax()
    lab = ["信用卡", "boleto", "代金券", "借记卡"]
    v = pay.reindex(["credit_card", "boleto", "voucher", "debit_card"]).values
    x = np.arange(len(v))
    bars = ax.bar(x, v, .55, color=[BLUE, ORANGE, GREEN, GREY], edgecolor="white")
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 1.4, f"{vi:.2f}%", ha="center", fontsize=FS, color="#333")
        ax.text(xi, vi / 2, f"{vi:.1f}", ha="center", va="center", fontsize=FS - .5,
                color="white")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=FS)
    ax.set_ylabel("占比（%）", fontsize=FS); ax.set_ylim(0, 92)
    ax.set_title("支付方式结构（柱顶数值标注）", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig04_labeled_bar.png")


# ------------------------------------------------------------ 05 层叠柱形图
def fig05():
    q = (w.assign(q=w.order_purchase_timestamp.dt.to_period("Q"))
         .groupby("q").agg(goods=("gmv_items", "sum"), freight=("freight", "sum")))
    x = np.arange(len(q))
    ax = newax()[1]
    ax.bar(x, q.goods / 1e4, .55, label="商品金额", color=BLUE)
    ax.bar(x, q.freight / 1e4, .55, bottom=q.goods / 1e4, label="运费", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in q.index], fontsize=FS - .5)
    ax.set_ylabel("金额（万雷亚尔）", fontsize=FS)
    ax.legend(fontsize=FS - 1, frameon=False)
    ax.set_title("季度 GMV 构成（商品金额 + 运费层叠）", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(plt.figure(fig.number) if False else ax.figure, "fig05_stacked_bar.png")


# ------------------------------------------------------------ 06 蝴蝶图
def fig06():
    fig, ax = newax(3.6)
    k = 8
    a = y17[:k]; b = y18[:k]
    y = np.arange(k)
    ax.barh(y, -a, .55, color="#5dade2", label="2017 年")
    ax.barh(y, b, .55, color="#e59866", label="2018 年")
    for yi, va, vb in zip(y, a, b):
        ax.text(-va - 120, yi, f"{va:,.0f}", va="center", ha="right", fontsize=FS - 1.5)
        ax.text(vb + 120, yi, f"{vb:,.0f}", va="center", fontsize=FS - 1.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{i} 月" for i in range(1, k + 1)], fontsize=FS)
    ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel("月度订单量（单，左 2017 / 右 2018）", fontsize=FS)
    ax.set_xlim(-max(a) * 1.5, max(b) * 1.35)
    ax.set_xticks([])
    ax.legend(fontsize=FS - 1, frameon=False, loc="lower right")
    ax.set_title("2017 与 2018 同月订单量（蝴蝶图）", fontsize=FS + 1, color="#333")
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "fig06_butterfly.png")


# ------------------------------------------------------------ 07 数值百分比
def fig07():
    fig, ax = newax()
    v = m_ord[12:] / 1e3
    x = np.arange(len(v))
    ax.bar(x, v, .55, color=DBLUE)
    yoy = [100 * (m_ord[12 + i] / m_ord[i] - 1) for i in range(len(v))]
    for xi, vi, g in zip(x, v, yoy):
        ax.text(xi, vi + .08, f"{vi:.1f}千\n{g:+.1f}%", ha="center", fontsize=FS - 2,
                color="#333", linespacing=1.1)
    ax.set_xticks(x); ax.set_xticklabels(ym[12:], rotation=45, fontsize=FS - .5)
    ax.set_ylabel("订单量（千单）", fontsize=FS); ax.set_ylim(0, v.max() * 1.32)
    ax.set_title("2018 年月度订单量与同比（双标注）", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig07_value_pct_bar.png")


# ------------------------------------------------------------ 08 对比柱形图
def fig08():
    fig, ax = newax()
    x = np.arange(8); wid = .36
    ax.bar(x - wid / 2, y17[:8] / 1e3, wid, label="2017", color="#aed6f1")
    ax.bar(x + wid / 2, y18[:8] / 1e3, wid, label="2018", color="#2874a6")
    ax.set_xticks(x); ax.set_xticklabels([f"{i} 月" for i in range(1, 9)], fontsize=FS)
    ax.set_ylabel("订单量（千单）", fontsize=FS)
    ax.legend(fontsize=FS - 1, frameon=False, ncol=2)
    ax.set_title("2017 与 2018 年 1—8 月订单量对比", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig08_grouped_bar.png")


# ------------------------------------------------------------ 09 甘特图
def fig09():
    fig, ax = newax(3.4, 6.0)
    d = st10[["days"]].sort_values("days")
    base = pd.Timestamp("2017-01-01")
    y = np.arange(len(d))
    for yi, (nm, row) in zip(y, d.iterrows()):
        ax.barh(yi, row.days, left=base, height=.5, color=BLUE, alpha=.85)
        ax.text(base + pd.Timedelta(days=row.days) + pd.Timedelta(days=1.2), yi,
                f"{row.days:.1f} 天", va="center", fontsize=FS - 1.5)
    ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=FS)
    ax.set_xlim(base, base + pd.Timedelta(days=30))
    ax.set_xlabel("平均配送时长（条长 = 天数）", fontsize=FS)
    fig.autofmt_xdate(rotation=30)
    ax.set_title("TOP10 州平均配送时长（甘特图）", fontsize=FS + 1, color="#333")
    ax.grid(axis="x", ls=":", alpha=.4)
    save(fig, "fig09_gantt.png")


# ------------------------------------------------------------ 10 平滑折线图
def fig10():
    fig, ax = newax()
    x = np.arange(len(m_ord))
    xs = np.linspace(0, len(m_ord) - 1, 300)
    ax.plot(xs, make_interp_spline(x, m_ord)(xs), color=BLUE, lw=1.8, label="平滑趋势")
    ax.plot(x, m_ord, "o", ms=3.2, color=DBLUE, label="实际值")
    ax.set_xticks(x[::2]); ax.set_xticklabels(ym[::2], rotation=45, fontsize=FS - .5)
    ax.set_ylabel("订单量（单）", fontsize=FS)
    ax.legend(fontsize=FS - 1, frameon=False, loc="upper left")
    ax.set_title("月度订单量平滑折线（样条插值）", fontsize=FS + 1, color="#333")
    ax.grid(ls=":", alpha=.4)
    save(fig, "fig10_smooth_line.png")


# ------------------------------------------------------------ 11 菱形走势
def fig11():
    fig, ax = newax()
    v = late_m.reindex(ym).fillna(0).values
    v = 100 - v
    x = np.arange(len(v))
    ax.plot(x, v, "-", color=GREEN, lw=1.4)
    ax.plot(x, v, "D", ms=4.2, mfc="white", mec=GREEN, mew=1.3)
    ax.set_xticks(x[::2]); ax.set_xticklabels(ym[::2], rotation=45, fontsize=FS - .5)
    ax.set_ylabel("准时交付率（%）", fontsize=FS)
    ax.set_ylim(v.min() - 3, 101)
    ax.set_title("月度准时交付率（菱形标记走势）", fontsize=FS + 1, color="#333")
    ax.grid(ls=":", alpha=.4)
    save(fig, "fig11_diamond_line.png")


# ------------------------------------------------------------ 12 对比折线图
def fig12():
    fig, ax = newax()
    x = np.arange(8)
    ax.plot(x, y17[:8] / 1e3, "-o", ms=4, color="#f39c12", label="2017 年")
    ax.plot(x, y18[:8] / 1e3, "-s", ms=4, color="#1f618d", label="2018 年")
    ax.set_xticks(x); ax.set_xticklabels([f"{i} 月" for i in range(1, 9)], fontsize=FS)
    ax.set_ylabel("订单量（千单）", fontsize=FS)
    ax.legend(fontsize=FS - 1, frameon=False)
    ax.set_title("两年同期订单量对比折线", fontsize=FS + 1, color="#333")
    ax.grid(ls=":", alpha=.4)
    save(fig, "fig12_compare_line.png")


def _donut_value(pct, label):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.pie([pct, 100 - pct], colors=[BLUE, "#eaecee"], startangle=90,
           counterclock=False, wedgeprops=dict(width=.30, edgecolor="white"))
    ax.text(0, .08, f"{pct:.2f}%", ha="center", va="center", fontsize=15,
            color=DBLUE, fontweight="bold")
    ax.text(0, -.22, label, ha="center", va="center", fontsize=FS)
    ax.set(aspect="equal")
    return fig


# ------------------------------------------------------------ 13 单值圆环
def fig13():
    save(_donut_value(96.89, "一次性买家占比"), "fig13_donut.png")


# ------------------------------------------------------------ 14/15 水球图
def _ball(pct, wave, label, name):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=DBLUE, lw=2.2, zorder=3)
    circ = Circle((0, 0), 1, facecolor="none", edgecolor="none")
    ax.add_patch(circ)
    x = np.linspace(-1, 1, 400)
    top = np.full_like(x, -1 + 2 * pct / 100)
    if wave:
        top = top + 0.05 * np.sin(4 * np.pi * x)
    poly = ax.fill_between(x, -1.05, top, color=BLUE, alpha=.88, zorder=2)
    poly.set_clip_path(circ)
    ln, = ax.plot(x, top, color="#154360", lw=1.4, zorder=2.5)
    ln.set_clip_path(circ)
    ax.text(0, .12, f"{pct:.2f}%", ha="center", va="center", fontsize=16,
            color="white", fontweight="bold", zorder=4)
    ax.text(0, -.26, label, ha="center", va="center", fontsize=FS - 1.5,
            color="white", zorder=4)
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    save(fig, name)


def fig14():
    pct = 100 * st.orders.head(5).sum() / st.orders.sum()
    _ball(round(pct, 2), False, "前五州订单占比", "fig14_water_ball.png")


def fig15():
    pct = 100 * cat.head(10).sum() / cat.sum()
    _ball(round(pct, 2), True, "前十品类收入占比", "fig15_wave_ball.png")


# ------------------------------------------------------------ 16 玉玦图
def fig16():
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_subplot(111, polar=True)
    v = cat8.values / cat8.values.sum() * 100
    n = len(v)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + .12
    rax = 40 + v                                   # 内圈留白形成“玦”
    for t, r_, c in zip(theta, rax, plt.get_cmap("Blues")(np.linspace(.4, .95, n))):
        ax.bar(t, r_ - 38, bottom=38, width=2 * np.pi / n * .82, color=c,
               edgecolor="white", linewidth=1.2)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 62); ax.set_yticks([]); ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title("TOP8 品类收入占比（玉玦图）", fontsize=FS + 1, color="#333", pad=6)
    save(fig, "fig16_jade_ring.png")


# ------------------------------------------------------------ 17 跑道图
def fig17():
    fig, ax = newax(3.4, 6.0)
    v = st10.orders.values
    y = np.arange(len(v))[::-1]
    cmap = plt.get_cmap("Blues")
    for i, (yi, vi) in enumerate(zip(y, v)):
        ax.plot([0, vi], [yi, yi], lw=15, solid_capstyle="round",
                color=cmap(.35 + .55 * i / len(v)), zorder=2)
        ax.annotate(f"{vi:,}", (vi, yi), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=FS - 1)
    ax.set_yticks(y); ax.set_yticklabels(st10.index, fontsize=FS)
    ax.set_xlabel("订单量（单）", fontsize=FS); ax.set_xlim(-600, v.max() * 1.18)
    ax.set_title("TOP10 州订单量（跑道图）", fontsize=FS + 1, color="#333")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "fig17_track.png")


# ------------------------------------------------------------ 18 南丁格尔
def fig18():
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_subplot(111, polar=True)
    v = cat8.values / 1e4
    n = len(v)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.bar(th, v, width=2 * np.pi / n * .9,
           color=plt.get_cmap("Oranges")(np.linspace(.45, .95, n)),
           edgecolor="white")
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title("TOP8 品类收入（南丁格尔玫瑰图）", fontsize=FS + 1, color="#333", pad=6)
    save(fig, "fig18_nightingale.png")


# ------------------------------------------------------------ 19 仪表盘
def fig19():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.set_aspect("equal"); ax.axis("off")
    bands = [(0, 60, "#f5b7b1"), (60, 85, "#fad7a0"), (85, 100, "#abebc6")]
    for a0, a1, col in bands:
        ax.add_patch(Wedge((0, 0), 1, 180 - a1 * 1.8, 180 - a0 * 1.8, width=.34,
                           facecolor=col, edgecolor="white"))
    pct = 93.21
    ang = np.deg2rad(180 - pct * 1.8)
    ax.plot([0, .78 * np.cos(ang)], [0, .78 * np.sin(ang)], color=RED, lw=2.6,
            solid_capstyle="round")
    ax.add_patch(Circle((0, 0), .055, color=DBLUE))
    ax.text(0, -.30, f"{pct:.2f}%", ha="center", fontsize=17, color=DBLUE,
            fontweight="bold")
    ax.text(0, -.52, "准时交付率", ha="center", fontsize=FS)
    ax.text(-1.05, -.02, "0", fontsize=FS - 1); ax.text(1.02, -.02, "100", fontsize=FS - 1)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-.62, 1.12)
    ax.set_title("准时交付率仪表盘", fontsize=FS + 1, color="#333")
    save(fig, "fig19_gauge.png")


# ------------------------------------------------------------ 20 柱形折线
def fig20():
    fig, ax = newax()
    x = np.arange(len(m_ord))
    ax.bar(x, m_ord, .6, color="#aed6f1", label="订单量")
    ax.set_ylabel("订单量（单）", fontsize=FS)
    ax2 = ax.twinx()
    yoy = [np.nan] + [100 * (m_ord[i] / m_ord[i - 1] - 1) for i in range(1, len(m_ord))]
    ax2.plot(x, yoy, "-o", ms=3, color=RED, lw=1.3, label="环比")
    ax2.set_ylabel("环比（%）", fontsize=FS, color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax.set_xticks(x[::2]); ax.set_xticklabels(ym[::2], rotation=45, fontsize=FS - .5)
    ax.set_title("月度订单量（柱）与环比（折线）", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig20_bar_line.png")


# ------------------------------------------------------------ 21 目标柱形
def fig21():
    fig, ax = newax()
    c6 = cat.head(6) / 1e4
    target = c6 * 1.10
    x = np.arange(len(c6)); wid = .36
    ax.bar(x - wid / 2, c6.values, wid, label="实际", color=DBLUE)
    ax.bar(x + wid / 2, target.values, wid, label="目标（+10%）", color="#f5cba7",
           edgecolor=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([c[:10] for c in c6.index], rotation=20, fontsize=FS - 1)
    ax.set_ylabel("收入（万雷亚尔）", fontsize=FS)
    ax.legend(fontsize=FS - 1, frameon=False)
    ax.set_title("TOP6 品类实际收入与目标对比", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig21_target_bar.png")


# ------------------------------------------------------------ 22 子弹图
def fig22():
    fig, ax = newax(3.6, 6.0)
    v = st10.orders.values
    y = np.arange(len(v))[::-1]
    q1, q2, q3 = np.percentile(v, [50, 75, 100])
    for yi in y:
        ax.barh(yi, q1, .52, color="#ecf0f1")
        ax.barh(yi, q2 - q1, .52, left=q1, color="#d5dbdb")
        ax.barh(yi, q3 - q2, .52, left=q2, color="#bfc9ca")
    for yi, vi in zip(y, v):
        ax.barh(yi, vi, .26, color=DBLUE)
        ax.plot([vi, vi], [yi - .17, yi + .17], color=RED, lw=2)
    ax.set_yticks(y); ax.set_yticklabels(st10.index, fontsize=FS)
    ax.set_xlabel("订单量（单）", fontsize=FS)
    ax.set_title("TOP10 州订单量（子弹图：及格/良好/优秀）", fontsize=FS + 1, color="#333")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "fig22_bullet.png")


# ------------------------------------------------------------ 23 柱形圆
def fig23():
    fig, ax = newax()
    v = cat.head(7) / 1e4
    x = np.arange(len(v))
    ax.bar(x, v.values, .55, color="#7fb3d5", label="实际收入")
    ax.plot(x, v.values * 1.18, "o", ms=5, color=RED, label="目标上限")
    for xi, vi in zip(x, v.values):
        ax.plot([xi, xi], [vi, vi * 1.18], ls=":", color=RED, lw=.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[:9] for c in v.index], rotation=25, fontsize=FS - 1)
    ax.set_ylabel("收入（万雷亚尔）", fontsize=FS)
    ax.legend(fontsize=FS - 1, frameon=False)
    ax.set_title("TOP7 品类收入与上限（柱形+圆点）", fontsize=FS + 1, color="#333")
    ax.grid(axis="y", ls=":", alpha=.4)
    save(fig, "fig23_bar_dot.png")


# ------------------------------------------------------------ 24 簇状柱形折线
def fig24():
    q = (w.assign(q=w.order_purchase_timestamp.dt.to_period("Q"))
         .agg(q=("order_id", "nunique"), cus=("customer_unique_id", "nunique")))
    q = (w.assign(q=w.order_purchase_timestamp.dt.to_period("Q"))
         .groupby("q").agg(orders=("order_id", "nunique"),
                           cus=("customer_unique_id", "nunique")))
    fig, ax = newax()
    x = np.arange(len(q)); wid = .36
    ax.bar(x - wid / 2, q.orders / 1e3, wid, label="订单数（千）", color=DBLUE)
    ax.bar(x + wid / 2, q.cus / 1e3, wid, label="客户数（千）", color="#aed6f1")
    ax.set_ylabel("数量（千）", fontsize=FS)
    ax2 = ax.twinx()
    ax2.plot(x, q.orders / q.cus, "-o", ms=4, color=RED, lw=1.4, label="单客订单数")
    ax2.set_ylabel("单客订单数", fontsize=FS, color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in q.index], fontsize=FS - .5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=FS - 1, frameon=False, loc="upper left")
    ax.set_title("季度订单/客户数与单客订单数", fontsize=FS + 1, color="#333")
    save(fig, "fig24_cluster_line.png")


# ------------------------------------------------------------ 25 复合柱形
def fig25():
    fig, ax = newax()
    x = np.arange(len(m_ord))
    ax.bar(x, m_ord / 1e3, .62, color="#a9cce3", label="月度订单量（千）")
    ax.set_ylabel("月度订单量（千单）", fontsize=FS)
    q = (pd.Series(m_gmv, index=pd.PeriodIndex(ym, freq="M"))
         .groupby(pd.PeriodIndex(ym, freq="M").asfreq("Q")).sum() / 1e4)
    qx = [i * 3 + 1.5 for i in range(len(q))]
    ax2 = ax.twinx()
    ax2.plot(qx, q.values, "-o", ms=4.5, color=ORANGE, lw=1.8, label="季度 GMV（万）")
    ax2.set_ylabel("季度 GMV（万雷亚尔）", fontsize=FS, color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.set_xticks(x[::2]); ax.set_xticklabels(ym[::2], rotation=45, fontsize=FS - .5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=FS - 1, frameon=False, loc="upper left")
    ax.set_title("月度订单量（柱）与季度 GMV（折线）", fontsize=FS + 1, color="#333")
    save(fig, "fig25_composite_bar.png")


# ------------------------------------------------------------ 26 滑珠图
def fig26():
    fig, ax = newax(3.2, 6.0)
    v = seg.rev_share.sort_values()
    y = np.arange(len(v))
    ax.hlines(y, 0, v.values, color="#d5dbdb", lw=2.4)
    ax.plot(v.values, y, "o", ms=8, color=BLUE, zorder=3)
    for yi, vi in zip(y, v.values):
        ax.text(vi + 1.2, yi, f"{vi:.2f}%", va="center", fontsize=FS - 1)
    ax.set_yticks(y); ax.set_yticklabels(v.index, fontsize=FS)
    ax.set_xlabel("收入占比（%）", fontsize=FS); ax.set_xlim(0, v.max() * 1.2)
    ax.set_title("RFM 八分类收入占比（滑珠图）", fontsize=FS + 1, color="#333")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "fig26_slide_bead.png")


# ------------------------------------------------------------ 27 对比滑珠
def fig27():
    fig, ax = newax(3.2, 6.0)
    c17 = (w[w.order_purchase_timestamp.dt.year == 2017]
           .merge(it[["order_id", "price"]], on="order_id", how="left")
           .merge(it[["order_id", "product_category_name_english"]]
                  .drop_duplicates("order_id"), on="order_id", how="left")
           .groupby("product_category_name_english").price.sum())
    c18 = (w[w.order_purchase_timestamp.dt.year == 2018]
           .merge(it[["order_id", "price"]], on="order_id", how="left")
           .merge(it[["order_id", "product_category_name_english"]]
                  .drop_duplicates("order_id"), on="order_id", how="left")
           .groupby("product_category_name_english").price.sum())
    both = pd.concat([c17.rename("y17"), c18.rename("y18")], axis=1).dropna()
    both = both.sort_values("y18", ascending=False).head(8).iloc[::-1] / 1e4
    y = np.arange(len(both))
    for yi, (nm, row) in zip(y, both.iterrows()):
        ax.hlines(yi, row.y17, row.y18, color="#bfc9ca", lw=2.2, zorder=1)
        ax.text(max(row.y17, row.y18) + 6, yi, f"{(row.y18 / row.y17 - 1) * 100:+.0f}%",
                va="center", fontsize=FS - 2, color="#666")
    ax.plot(both.y17, y, "o", ms=7, color="#f39c12", label="2017", zorder=3)
    ax.plot(both.y18, y, "o", ms=7, color=DBLUE, label="2018", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([c[:14] for c in both.index], fontsize=FS - .5)
    ax.set_xlabel("品类收入（万雷亚尔）", fontsize=FS)
    ax.legend(fontsize=FS - 1, frameon=False, loc="lower right")
    ax.set_title("TOP8 品类收入两年对比（对比滑珠）", fontsize=FS + 1, color="#333")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "fig27_slide_bead_cmp.png")


if __name__ == "__main__":
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09,
               fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18,
               fig19, fig20, fig21, fig22, fig23, fig24, fig25, fig26, fig27]:
        try:
            fn()
        except Exception as e:
            print("FAIL", fn.__name__, type(e).__name__, e)
