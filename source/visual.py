import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")  

def run_dashboard(csv_path: str = "items.csv"):
    df = pd.read_csv(csv_path)

    df["date_parsed"] = pd.to_datetime(df["date"], format="%d-%m-%y", dayfirst=True, errors="coerce")

    df = df.dropna(subset=["date_parsed"])
    if df.empty:
        raise SystemExit("No valid dates in CSV.")
    df["day"] = df["date_parsed"].dt.normalize()
    latest_day = df["day"].max()

    today_df = df[df["day"] == latest_day].copy()
    if today_df.empty:
        raise SystemExit("No rows for the latest day. Check your date format or data.")

    week_start = latest_day - pd.Timedelta(days=6)     # last 7 calendar days
    trend_start = latest_day - pd.Timedelta(days=13)   # last 14 calendar days

    week_df = df[(df["day"] >= week_start) & (df["day"] <= latest_day)].copy()

    for col in ("duration", "hardness"):
        today_df[col] = pd.to_numeric(today_df[col], errors="coerce")
        week_df[col]  = pd.to_numeric(week_df[col], errors="coerce")

    grouped_today = (
        today_df.groupby("title", as_index=False)
        .agg(total_duration=("duration", "sum"), avg_hardness=("hardness", "mean"))
        .sort_values("total_duration", ascending=False)
    )
    grouped_week = (
        week_df.groupby("title", as_index=False)
        .agg(total_duration=("duration", "sum"), avg_hardness=("hardness", "mean"))
        .sort_values("total_duration", ascending=False)
    )

    total_day_duration  = grouped_today["total_duration"].sum()
    total_week_duration = grouped_week["total_duration"].sum()

    all_titles = grouped_week["title"].tolist() if len(grouped_week) else grouped_today["title"].tolist()
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, max(3, len(all_titles))))
    title_to_color = {t: palette[i % len(palette)] for i, t in enumerate(all_titles)}
    def colors_for(grouped):
        return [title_to_color.get(t, palette[0]) for t in grouped["title"]]

    trend_df = df[(df["day"] >= trend_start) & (df["day"] <= latest_day)].copy()
    daily_totals = trend_df.groupby("day", as_index=True)["duration"].sum()
    trend_index = pd.date_range(trend_start, latest_day, freq="D")
    daily_totals = daily_totals.reindex(trend_index, fill_value=0)

    stack_df = week_df.pivot_table(index="day", columns="title", values="duration", aggfunc="sum").fillna(0)
    stack_index = pd.date_range(week_start, latest_day, freq="D")
    stack_df = stack_df.reindex(stack_index, fill_value=0)
    stack_titles = sorted(stack_df.columns.tolist(), key=lambda t: all_titles.index(t) if t in all_titles else 999)
    stack_df = stack_df[stack_titles]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), facecolor="#121212")
    (ax_bar_today, ax_pie_today), (ax_bar_week, ax_pie_week), (ax_line_trend, ax_stack_week) = axes

    pretty_date = latest_day.strftime("%d-%m-%y")
    pretty_week = f"{week_start.strftime('%d-%m-%y')} → {pretty_date}"
    trend_start_str = trend_start.strftime('%d-%m-%y')

    fig.suptitle(
        f"Today: {pretty_date} {total_day_duration:.0f} min    |    "
        f"7 days: {pretty_week} {total_week_duration:.0f} min",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    bars = ax_bar_today.bar(
        grouped_today["title"], grouped_today["total_duration"],
        color=colors_for(grouped_today), edgecolor="white", linewidth=1
    )
    for b in bars:
        h = b.get_height()
        ax_bar_today.annotate(f"{h:.0f}", xy=(b.get_x()+b.get_width()/2, h),
                              xytext=(0,4), textcoords="offset points",
                              ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
    ax_bar_today.set_title("", fontsize=13, fontweight="bold", color="white")
    ax_bar_today.set_xlabel("Title", fontsize=11, color="lightgray")
    ax_bar_today.set_ylabel("Duration (minutes)", fontsize=11, color="lightgray")
    if len(grouped_today):
        ax_bar_today.set_ylim(0, grouped_today["total_duration"].max() * 1.2)
    ax_bar_today.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
    ax_bar_today.tick_params(colors="lightgray")

    sizes = grouped_today["total_duration"].values
    labels = grouped_today["title"].values
    total = sizes.sum() if len(sizes) else 0
    autopct = (lambda p: f"{p:.1f}%\n({p*total/100:.0f}m)") if total > 0 else None
    wedges, texts, autotexts = ax_pie_today.pie(
        sizes, labels=None, autopct=autopct, startangle=90,
        colors=colors_for(grouped_today), pctdistance=0.75,
        textprops=dict(color="white", fontsize=9),
        wedgeprops=dict(edgecolor="white", linewidth=1)
    )
    ax_pie_today.axis("equal")
    ax_pie_today.set_title("", fontsize=13, fontweight="bold", color="white")
    ax_pie_today.legend(
        wedges, labels, title="Title", loc="center left", bbox_to_anchor=(1.0, 0.5),
        facecolor="#121212", edgecolor="white", labelcolor="white"
    )

    bars_w = ax_bar_week.bar(
        grouped_week["title"], grouped_week["total_duration"],
        color=colors_for(grouped_week), edgecolor="white", linewidth=1
    )
    for b in bars_w:
        h = b.get_height()
        ax_bar_week.annotate(f"{h:.0f}", xy=(b.get_x()+b.get_width()/2, h),
                             xytext=(0,4), textcoords="offset points",
                             ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
    ax_bar_week.set_title("Last 7 Days", fontsize=13, fontweight="bold", color="white")
    ax_bar_week.set_xlabel("Title", fontsize=11, color="lightgray")
    ax_bar_week.set_ylabel("Duration", fontsize=11, color="lightgray")
    if len(grouped_week):
        ax_bar_week.set_ylim(0, grouped_week["total_duration"].max() * 1.2)
    ax_bar_week.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
    ax_bar_week.tick_params(colors="lightgray")

    sizes_w = grouped_week["total_duration"].values
    labels_w = grouped_week["title"].values
    total_w = sizes_w.sum() if len(sizes_w) else 0
    autopct_w = (lambda p: f"{p:.1f}%\n({p*total_w/100:.0f}m)") if total_w > 0 else None
    wedges_w, texts_w, autotexts_w = ax_pie_week.pie(
        sizes_w, labels=None, autopct=autopct_w, startangle=90,
        colors=colors_for(grouped_week), pctdistance=0.75,
        textprops=dict(color="white", fontsize=9),
        wedgeprops=dict(edgecolor="white", linewidth=1)
    )
    ax_pie_week.axis("equal")
    ax_pie_week.set_title("Last 7 Days", fontsize=13, fontweight="bold", color="white")
    ax_pie_week.legend(
        wedges_w, labels_w, title="Title", loc="center left", bbox_to_anchor=(1.0, 0.5),
        facecolor="#121212", edgecolor="white", labelcolor="white"
    )

    y_vals = daily_totals.values.astype(float)
    x_vals = daily_totals.index
    ax_line_trend.plot(x_vals, y_vals, marker="o", linewidth=2)
    ax_line_trend.set_title(f"Last 14 Days — {trend_start_str} → {pretty_date}",
                            fontsize=13, fontweight="bold", color="white")
    ax_line_trend.set_xlabel("Date", fontsize=11, color="lightgray")
    ax_line_trend.set_ylabel("Minutes", fontsize=11, color="lightgray")
    ax_line_trend.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax_line_trend.tick_params(colors="lightgray", axis="x", rotation=45)
    ax_line_trend.tick_params(colors="lightgray", axis="y")
    if len(y_vals):
        ymax = y_vals.max()
        ax_line_trend.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)
    for x, y in zip(x_vals, y_vals):
        ax_line_trend.annotate(
            f"{int(y)}m", xy=(x, y), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.4), ec="none")
        )

    bottom = np.zeros(len(stack_df), dtype=float)
    for t in stack_titles:
        vals = stack_df[t].values
        ax_stack_week.bar(
            stack_df.index, vals, bottom=bottom,
            label=t, color=title_to_color.get(t, palette[0]),
            edgecolor="white", linewidth=0.5
        )
        bottom += vals
    ax_stack_week.set_title(f"Last 7 Days — {pretty_week}",
                            fontsize=13, fontweight="bold", color="white")
    ax_stack_week.set_xlabel("Date", fontsize=11, color="lightgray")
    ax_stack_week.set_ylabel("Minutes", fontsize=11, color="lightgray")
    ax_stack_week.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
    ax_stack_week.tick_params(colors="lightgray", axis="x", rotation=45)
    ax_stack_week.tick_params(colors="lightgray", axis="y")
    ax_stack_week.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        facecolor="#121212", edgecolor="white", labelcolor="white", title="Title"
    )

    plt.tight_layout()
    plt.show()
