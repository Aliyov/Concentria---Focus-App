from datetime import datetime, date, timedelta
import io
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dateutil import parser

st.set_page_config(page_title="Concentria Dashboard", layout="wide", initial_sidebar_state="auto")

plt.style.use("dark_background")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    html, body, .block-container { background: #0e1117; }
    .stApp { background: #0e1117; color: #d6e6ea; }
    .dashboard-card {
        background: #0b0f13;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
        border: 1px solid rgba(64, 224, 208, 0.06);
    }
    .metric-large { font-family: monospace; font-size: 22px; color: #e7fff8 }
    .metric-small { font-family: monospace; font-size: 13px; color: #9fbfc0 }
    .section-title { color: #bfeee6; font-weight: 600; margin-bottom: 6px; }
    .muted { color: #7f8b8d; font-size: 13px; }

    /* KPI improvements */
    .kpi-container {
        display:flex;
        gap:14px;
        justify-content:center;
        align-items:stretch;
        flex-wrap:wrap;
    }
    .kpi-card {
        background: linear-gradient(180deg, rgba(10,14,16,0.8), rgba(7,11,13,0.6));
        border-radius: 10px;
        padding: 12px 16px;
        min-width: 160px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.03);
        display:flex;
        flex-direction:column;
        align-items:center;
    }
    .kpi-number {
        font-family: "Segoe UI", Roboto, monospace;
        font-size: 26px;
        color: #e7fff8;
        font-weight: 700;
        letter-spacing: 0.6px;
    }
    .kpi-label {
        font-size: 12px;
        color: #9fbfc0;
        margin-top:6px;
    }
    .kpi-note {
        font-size: 11px;
        color: #7f8b8d;
        margin-top:4px;
    }
    @media (max-width:800px) {
        .kpi-card { min-width: 140px; padding:10px; }
        .kpi-number { font-size:22px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data(path="items.csv"):
    expected_cols = ["date", "clock", "title", "duration", "hardness", "note"]

    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_cols)

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=expected_cols)

    if "date" not in df.columns:
        df["date"] = pd.NA

    def parse_date(v):
        try:
            dt = parser.parse(str(v), dayfirst=True)
            return pd.to_datetime(dt.date())
        except Exception:
            return pd.NaT

    df["date_parsed"] = df["date"].apply(parse_date)

    df = df.dropna(subset=["date_parsed"]).copy()

    if df.empty:
        for col in ("date_time", "duration", "hardness", "hour", "date"):
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return pd.DataFrame(columns=expected_cols) 

    def parse_datetime(row):
        try:
            clock_val = row.get("clock", "")
            if pd.isna(clock_val) or str(clock_val).strip() == "":
                return pd.Timestamp(row["date_parsed"])
            return pd.to_datetime(f"{row['date_parsed'].date().isoformat()} {clock_val}", dayfirst=True)
        except Exception:
            return pd.Timestamp(row["date_parsed"])

    df["date_time"] = df.apply(parse_datetime, axis=1)

    df["duration"] = pd.to_numeric(df.get("duration", 0), errors="coerce").fillna(0).astype(int)
    df["hardness"] = pd.to_numeric(df.get("hardness", 0), errors="coerce").fillna(0).astype(int)

    df["date"] = df["date_parsed"].dt.date
    df["hour"] = df["date_time"].dt.hour.fillna(0).astype(int)

    if "title" not in df.columns:
        df["title"] = "untitled"

    return df

df = load_data("items.csv")

if df.empty or "date_parsed" not in df.columns or df["date_parsed"].isna().all():
    st.title("Concentria Dashboard")
    st.error("❌ `items.csv` not found or contains no valid `date` values. Please place a CSV with a 'date' column next to this script.")
    st.stop()

st.sidebar.header("Controls")
min_date = df["date_parsed"].min().date()
max_date = df["date_parsed"].max().date()
date_range = st.sidebar.date_input("Data range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

titles = sorted(df["title"].fillna("untitled").unique().tolist())
selected_titles = st.sidebar.multiselect("Titles (filter)", options=titles, default=titles)

min_duration = st.sidebar.number_input("Min session (min)", min_value=0, value=0, step=5)
hardness_min, hardness_max = st.sidebar.slider("Hardness range", 0, 10, (0, 10))

default_month = max_date.month
default_year = max_date.year
col_m, col_y = st.sidebar.columns([1, 1])
sel_month = col_m.selectbox(
    "Month",
    options=list(range(1, 13)),
    index=default_month - 1,
    format_func=lambda m: datetime(2000, m, 1).strftime("%B"),
)
sel_year = col_y.number_input("Year", min_value=min_date.year, max_value=max_date.year, value=default_year, step=1)

start_date, end_date = date_range
mask = (
    (df["date_parsed"].dt.date >= start_date)
    & (df["date_parsed"].dt.date <= end_date)
    & (df["title"].isin(selected_titles))
    & (df["duration"] >= int(min_duration))
    & (df["hardness"] >= int(hardness_min))
    & (df["hardness"] <= int(hardness_max))
)
fdf = df[mask].copy()

if fdf.empty:
    st.title("Concentria Dashboard")
    st.warning("No sessions match filters. Adjust filters to see data.")
    st.stop()

def monthly_totals(df_, year, month):
    first = date(year, month, 1)
    if month == 12:
        next_first = date(year + 1, 1, 1)
    else:
        next_first = date(year, month + 1, 1)
    last = next_first - timedelta(days=1)
    all_days = pd.date_range(first, last, freq="D").date
    grouped = df_.groupby(df_["date_parsed"].dt.date)["duration"].sum()
    series = pd.Series({d: int(grouped.get(d, 0)) for d in all_days})
    series.index.name = "day"
    return series

def compute_streak(dates):
    if not dates:
        return 0, 0
    norm = sorted({pd.to_datetime(d).normalize() for d in dates})
    longest = 0
    cur_len = 0
    prev = None
    for d in norm:
        if prev is None or (d - prev).days == 1:
            cur_len += 1
        else:
            cur_len = 1
        prev = d
        if cur_len > longest:
            longest = cur_len
    latest = norm[-1]
    cur = latest
    current = 0
    while cur in norm:
        current += 1
        cur = cur - pd.Timedelta(days=1)
    return current, longest

month_series = monthly_totals(fdf, sel_year, sel_month)

if "selected_day" not in st.session_state:
    available_days = [d for d in month_series.index if month_series.loc[d] > 0]
    if available_days:
        st.session_state.selected_day = available_days[-1]
    else:
        st.session_state.selected_day = month_series.index[-1]

st.markdown("<div style='text-align:center;'><h1 style='color:#bfeee6;margin:0'>Concentria</h1>"
            "<div class='muted'></div></div>", unsafe_allow_html=True)
st.markdown("---")

total_minutes_month = int(month_series.sum())
focus_days_month = int((month_series > 0).sum())
avg_per_focus_day = int(month_series[month_series > 0].mean()) if focus_days_month > 0 else 0
avg_session = int(fdf["duration"].mean()) if not fdf.empty else 0

st.markdown("<div class='dashboard-card' style='margin-bottom:12px'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Key metrics</div>", unsafe_allow_html=True)

kpi_html = f"""
<div class="kpi-container">
  <div class="kpi-card">
    <div class="kpi-number">{total_minutes_month}</div>
    <div class="kpi-label">Total minutes</div>
    <div class="kpi-note">This month</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-number">{focus_days_month}</div>
    <div class="kpi-label">Active days</div>
    <div class="kpi-note">Days with sessions</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-number">{avg_per_focus_day}</div>
    <div class="kpi-label">Avg / focus-day</div>
    <div class="kpi-note">Mean minutes on active days</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-number">{avg_session}</div>
    <div class="kpi-label">Avg session</div>
    <div class="kpi-note">Mean session length</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
days = [d.day for d in month_series.index]
vals = month_series.values
bars = ax.bar(days, vals, color="#22c1a8", edgecolor="#0b3a33", linewidth=0.6)
ax.set_facecolor("#0b0f13")
ax.set_xlabel("Day of month", color="#9fbfc0")
ax.set_ylabel("Minutes focused", color="#9fbfc0")
ax.set_title(f"Monthly Focus — {datetime(sel_year, sel_month, 1).strftime('%B %Y')}", color="#bfeee6")
ax.tick_params(colors="#9fbfc0")
sel_day_num = pd.to_datetime(st.session_state.selected_day).day
if 1 <= sel_day_num <= len(days):
    if sel_day_num in days:
        bars[days.index(sel_day_num)].set_color("#9be7d6")
        bars[days.index(sel_day_num)].set_edgecolor("#ffffff")
        bars[days.index(sel_day_num)].set_linewidth(1.2)
if vals.max() > 0:
    top_idx = int(np.argmax(vals))
    top_day = days[top_idx]
    top_val = vals[top_idx]
    ax.annotate(f"Best: {top_day} ({top_val}m)", xy=(top_day, top_val), xytext=(0, 8),
                textcoords="offset points", ha="center", color="#e7fff8", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="#072a24", ec="none", alpha=0.7))
ymax = vals.max() if vals.size > 0 else 0
top_space = max(6, int(ymax * 0.12))
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + top_space / 3, f"{int(h)}", ha="center", va="bottom", color='#e7fff8', fontsize=8)
ax.set_ylim(0, max(ymax + top_space, 10))
ax.grid(axis="y", color="#072a24", linestyle="--", linewidth=0.6, alpha=0.8)
plt.tight_layout()
st.pyplot(fig)

st.markdown("<div class='dashboard-card' style='margin-top:12px'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Day Inspector</div>", unsafe_allow_html=True)
dp_min = month_series.index[0]
dp_max = month_series.index[-1]
selected = st.date_input("Selected day", value=pd.to_datetime(st.session_state.selected_day).date(), min_value=dp_min, max_value=dp_max)
st.session_state.selected_day = selected
day_mask = fdf["date_parsed"].dt.date == pd.to_datetime(selected).date()
day_sessions = fdf[day_mask].sort_values("date_time")
if day_sessions.empty:
    st.info("No sessions recorded for this day.")
else:
    st.markdown(f"<div class='metric-large'>{int(day_sessions['duration'].sum())} min</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-small'>{len(day_sessions)} sessions · avg {int(day_sessions['duration'].mean())} min</div>", unsafe_allow_html=True)
    display_df = day_sessions[["clock", "title", "duration", "hardness", "note"]].rename(columns={"clock": "time", "duration": "min"})
    st.table(display_df.reset_index(drop=True).style.set_table_styles([{'selector':'','props':[('background','#0b0f13')]}]))
    c1, c2 = st.columns(2)
    csv_bytes = day_sessions.to_csv(index=False).encode("utf-8")
    c1.download_button("Download CSV", data=csv_bytes, file_name=f"sessions-{selected.isoformat()}.csv", mime="text/csv")
    try:
        buf = io.BytesIO()
        day_sessions.to_excel(buf, index=False, sheet_name="sessions")
        buf.seek(0)
        c2.download_button("Download XLSX", data=buf, file_name=f"sessions-{selected.isoformat()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        c2.write("XLSX unavailable")

st.markdown("<hr style='border:0.5px solid rgba(255,255,255,0.04)'/>", unsafe_allow_html=True)

all_dates = sorted({d for d in fdf["date_parsed"].dt.date.unique()})
current_streak, longest_streak = compute_streak(all_dates)
st.markdown(
    f"<div style='margin-top:6px'><span class='metric-small'>Current streak</span><div class='metric-large'>{current_streak} days</div>"
    f"<div class='metric-small'>Longest streak {longest_streak} days</div></div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)  # close Day Inspector card

st.markdown("<div class='dashboard-card' style='margin-top:12px'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Additional charts</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    cum = pd.Series(vals, index=month_series.index).cumsum()
    fig_cum, axc = plt.subplots(figsize=(4.5, 3), dpi=100)
    axc.plot([d.day for d in cum.index], cum.values, marker='o', linewidth=1.6)
    axc.set_facecolor("#0b0f13")
    axc.set_xlabel("Day", color="#9fbfc0")
    axc.set_ylabel("Cumulative minutes", color="#9fbfc0")
    axc.set_title("Cumulative minutes (month)", color="#bfeee6")
    axc.tick_params(colors="#9fbfc0")
    axc.grid(axis="y", color="#072a24", linestyle="--", linewidth=0.6, alpha=0.8)
    plt.tight_layout()
    st.pyplot(fig_cum)

with c2:
    weekday_totals = fdf.groupby(fdf["date_parsed"].dt.weekday)["duration"].sum().reindex(range(7), fill_value=0)
    wd_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_wd, axw = plt.subplots(figsize=(4.5, 3), dpi=100)
    bars_w = axw.bar(range(7), weekday_totals.values, color="#2fbf9a", edgecolor="#08332b")
    axw.set_xticks(range(7))
    axw.set_xticklabels(wd_names, color="#9fbfc0")
    axw.set_ylabel("Minutes", color="#9fbfc0")
    axw.set_title("Minutes by weekday", color="#bfeee6")
    axw.tick_params(colors="#9fbfc0")
    for i, b in enumerate(bars_w):
        h = b.get_height()
        axw.text(b.get_x() + b.get_width() / 2, h + max(3, int(0.06 * weekday_totals.max())), f"{int(h)}", ha='center', va='bottom', fontsize=8, color='#e7fff8')
    axw.grid(axis="y", color="#072a24", linestyle="--", linewidth=0.6, alpha=0.8)
    plt.tight_layout()
    st.pyplot(fig_wd)

with c3:
    top_titles = fdf.groupby("title")["duration"].sum().sort_values(ascending=False)
    if top_titles.empty:
        st.info("No titles to show")
    else:
        top_n = 6
        top = top_titles.head(top_n)
        other = top_titles.iloc[top_n:].sum()
        labels = top.index.tolist()
        sizes = top.values.tolist()
        if other > 0:
            labels.append("Other")
            sizes.append(int(other))
        fig_p, axp = plt.subplots(figsize=(4.5, 3), dpi=100)
        wedges, texts, autotexts = axp.pie(sizes, autopct=lambda p: f"{p:.0f}%" if p >= 3 else "", startangle=90)
        axp.set_title("Top titles share", color="#bfeee6")
        for t in texts:
            t.set_color("#9fbfc0")
        for a in autotexts:
            a.set_color("#e7fff8")
        leg = axp.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
        for t in leg.get_texts():
            t.set_color("#9fbfc0")
        plt.tight_layout()
        st.pyplot(fig_p)

st.markdown("</div>", unsafe_allow_html=True)

insightful_texts = []
prev_month_year = sel_year if sel_month > 1 else sel_year - 1
prev_month = sel_month - 1 if sel_month > 1 else 12
try:
    prev_series = monthly_totals(fdf, prev_month_year, prev_month)
    prev_total = int(prev_series.sum())
except Exception:
    prev_total = 0
if prev_total > 0:
    pct = 100.0 * (total_minutes_month - prev_total) / prev_total
    sign = "+" if pct >= 0 else ""
    insightful_texts.append(f"{sign}{pct:.1f}% vs previous month ({prev_total} min)")
else:
    insightful_texts.append("No data for previous month to compare")

if 'weekday_totals' in locals() and not weekday_totals.empty:
    wd_idx = int(weekday_totals.idxmax())
    wd_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wd_idx]
    insightful_texts.append(f"Best weekday: {wd_name} ({int(weekday_totals.max())} min)")

st.markdown("<div class='dashboard-card' style='margin-top:12px'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Quick Insights</div>", unsafe_allow_html=True)
for t in insightful_texts:
    st.markdown(f"<div class='metric-small'>{t}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Top Focused Titles (this range)</div>", unsafe_allow_html=True)
top_titles_tbl = fdf.groupby("title")["duration"].sum().sort_values(ascending=False).head(12)
if not top_titles_tbl.empty:
    tbl = top_titles_tbl.reset_index().rename(columns={"duration": "minutes"})
    st.table(tbl)
else:
    st.info("No titles found in this range.")

st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
csv_bytes = fdf.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_sessions.csv", mime="text/csv")
try:
    buf2 = io.BytesIO()
    fdf.to_excel(buf2, index=False, sheet_name="sessions")
    buf2.seek(0)
    st.download_button("Download filtered XLSX", data=buf2, file_name="filtered_sessions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception:
    st.write("XLSX export requires openpyxl (optional).")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;margin-top:18px;color:#7f8b8d'>Ali Aliyev 2025</div>", unsafe_allow_html=True)
