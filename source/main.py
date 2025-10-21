import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import os, csv, random, sys, platform
from pathlib import Path
import threading
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


APP_NAME = "Concentria"


def app_data_dir() -> Path:
    if platform.system() == "Darwin":
        p = Path.home() / "Library" / "Application Support" / APP_NAME
    elif platform.system() == "Windows":
        p = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    else:
        p = Path.home() / f".{APP_NAME.lower()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


CSV_FILE = str(app_data_dir() / "items.csv")
CSV_FIELDS = ["date", "clock", "title", "duration", "note", "hardness"]

plt.style.use("dark_background")



def run_dashboard(csv_path: str = CSV_FILE):
    # --- load & validate (unchanged) ---
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {e}")

    required = {"date", "title", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    if "hardness" not in df.columns:
        df["hardness"] = np.nan

    df["date_parsed"] = pd.to_datetime(df["date"], format="%d-%m-%y", dayfirst=True, errors="coerce")
    if df["date_parsed"].isna().any():
        mask = df["date_parsed"].isna()
        df.loc[mask, "date_parsed"] = pd.to_datetime(df.loc[mask, "date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["date_parsed"])
    if df.empty:
        raise RuntimeError("No valid dates in CSV.")
    df["day"] = df["date_parsed"].dt.normalize()
    latest_day = df["day"].max()

    today_df = df[df["day"] == latest_day].copy()
    if today_df.empty:
        raise RuntimeError("No rows for the latest day. Check your date format or data.")

    week_start = latest_day - pd.Timedelta(days=6)
    trend_start = latest_day - pd.Timedelta(days=13)

    week_df = df[(df["day"] >= week_start) & (df["day"] <= latest_day)].copy()

    for col in ("duration", "hardness"):
        today_df[col] = pd.to_numeric(today_df[col], errors="coerce")
        week_df[col] = pd.to_numeric(week_df[col], errors="coerce")

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

    total_day_duration = grouped_today["total_duration"].sum()
    total_week_duration = grouped_week["total_duration"].sum()

    # --- STREAK CALCULATIONS (unchanged) ---
    unique_days = sorted(pd.to_datetime(df["day"].unique()))
    if not unique_days:
        max_streak = 0
    else:
        max_streak = 1
        current_run = 1
        for i in range(1, len(unique_days)):
            if unique_days[i] == unique_days[i - 1] + pd.Timedelta(days=1):
                current_run += 1
                if current_run > max_streak:
                    max_streak = current_run
            else:
                current_run = 1

    set_days = set(pd.to_datetime(df["day"].unique()))
    current_streak = 0
    day_ptr = pd.to_datetime(latest_day)
    while day_ptr in set_days:
        current_streak += 1
        day_ptr -= pd.Timedelta(days=1)
    # --- end streaks ---

    # colors and stacks (unchanged)
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

    # ---------------- Improved layout using GridSpec (reserve right margin for legends) ----------------
    import matplotlib.gridspec as gridspec

    # set right < 1.0 to reserve room for pie legends (adjust if your legends are wider)
    fig = plt.figure(figsize=(14, 10), facecolor="#121212")
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.5, hspace=0.6,
                           left=0.06, right=0.75, top=0.88, bottom=0.06)

    ax_bar_today = fig.add_subplot(gs[0, 0:2])
    ax_pie_today = fig.add_subplot(gs[0, 2])
    ax_bar_week = fig.add_subplot(gs[1, 0:2])
    ax_pie_week = fig.add_subplot(gs[1, 2])
    ax_line_trend = fig.add_subplot(gs[2, 0:2])
    ax_stack_week = fig.add_subplot(gs[2, 2])

    # compact, consistent font sizes
    small_title = dict(fontsize=12, fontweight="bold", color="white")
    label_style = dict(fontsize=10, color="lightgray")
    annot_fs = 9

    pretty_date = latest_day.strftime("%d-%m-%y")
    pretty_week = f"{week_start.strftime('%d-%m-%y')} → {pretty_date}"
    trend_start_str = trend_start.strftime('%d-%m-%y')

    # header: show primary totals and streaks in a single short suptitle
    fig.suptitle(
        f"Today: {pretty_date} {total_day_duration:.0f}m   |   7d: {total_week_duration:.0f}m   |   "
        f"Max streak: {max_streak}d   |   Current streak: {current_streak}d",
        fontsize=13, fontweight="bold", color="white"
    )

    # --- Today bar ---
    bars = ax_bar_today.bar(
        grouped_today["title"], grouped_today["total_duration"],
        color=colors_for(grouped_today), edgecolor="white", linewidth=0.8
    )
    for b in bars:
        h = b.get_height()
        ax_bar_today.annotate(f"{h:.0f}", xy=(b.get_x() + b.get_width() / 2, h),
                              xytext=(0, 3), textcoords="offset points",
                              ha="center", va="bottom", fontsize=annot_fs, color="white")
    ax_bar_today.set_title("Today (by title)", **small_title)
    ax_bar_today.set_xlabel("", **label_style)
    ax_bar_today.set_ylabel("Minutes", **label_style)
    if len(grouped_today):
        ax_bar_today.set_ylim(0, max(1, grouped_today["total_duration"].max() * 1.15))
    ax_bar_today.grid(axis="y", linestyle="--", alpha=0.25, color="gray")
    ax_bar_today.tick_params(colors="lightgray", labelsize=9, axis="x", rotation=25)

    # --- Today pie (compact, shrunk radius) ---
    sizes = grouped_today["total_duration"].values
    labels = grouped_today["title"].values
    total = sizes.sum() if len(sizes) else 0
    autopct = (lambda p: f"{p:.0f}%\n{p * total / 100:.0f}m") if total > 0 else None
    wedges, texts, autotexts = ax_pie_today.pie(
        sizes, labels=None, autopct=autopct, startangle=90,
        colors=colors_for(grouped_today), pctdistance=0.68, radius=0.82,
        textprops=dict(color="white", fontsize=8),
        wedgeprops=dict(edgecolor="#1a1a1a", linewidth=0.6)
    )
    ax_pie_today.axis("equal")
    ax_pie_today.set_title("Today (share)", **small_title)
    # place legend in the reserved right margin area (inside figure now)
    if len(labels):
        ax_pie_today.legend(wedges, labels, title="Title", loc="center left",
                            bbox_to_anchor=(1.02, 0.5), frameon=True, facecolor="#121212",
                            edgecolor="white", fontsize=9, handlelength=1.0)

    # --- Week bar ---
    bars_w = ax_bar_week.bar(
        grouped_week["title"], grouped_week["total_duration"],
        color=colors_for(grouped_week), edgecolor="white", linewidth=0.8
    )
    for b in bars_w:
        h = b.get_height()
        ax_bar_week.annotate(f"{h:.0f}", xy=(b.get_x() + b.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=annot_fs, color="white")
    ax_bar_week.set_title("Last 7 Days (by title)", **small_title)
    ax_bar_week.set_xlabel("", **label_style)
    ax_bar_week.set_ylabel("Minutes", **label_style)
    if len(grouped_week):
        ax_bar_week.set_ylim(0, max(1, grouped_week["total_duration"].max() * 1.15))
    ax_bar_week.grid(axis="y", linestyle="--", alpha=0.25, color="gray")
    ax_bar_week.tick_params(colors="lightgray", labelsize=9, axis="x", rotation=25)

    # --- Week pie (compact, shrunk radius) ---
    sizes_w = grouped_week["total_duration"].values
    labels_w = grouped_week["title"].values
    total_w = sizes_w.sum() if len(sizes_w) else 0
    autopct_w = (lambda p: f"{p:.0f}%\n{p * total_w / 100:.0f}m") if total_w > 0 else None
    wedges_w, texts_w, autotexts_w = ax_pie_week.pie(
        sizes_w, labels=None, autopct=autopct_w, startangle=90,
        colors=colors_for(grouped_week), pctdistance=0.68, radius=0.82,
        textprops=dict(color="white", fontsize=8),
        wedgeprops=dict(edgecolor="#1a1a1a", linewidth=0.6)
    )
    ax_pie_week.axis("equal")
    ax_pie_week.set_title("Last 7 Days (share)", **small_title)
    if len(labels_w):
        ax_pie_week.legend(wedges_w, labels_w, title="Title", loc="center left",
                           bbox_to_anchor=(1.02, 0.5), frameon=True, facecolor="#121212",
                           edgecolor="white", fontsize=9, ncol=1, handlelength=1.0)

    # --- 14-day trend line (wider) ---
    y_vals = daily_totals.values.astype(float)
    x_vals = daily_totals.index
    ax_line_trend.plot(x_vals, y_vals, marker="o", linewidth=1.75)
    ax_line_trend.set_title(f"Last 14 Days — {trend_start_str} → {pretty_date}", **small_title)
    ax_line_trend.set_xlabel("Date", **label_style)
    ax_line_trend.set_ylabel("Minutes", **label_style)
    ax_line_trend.grid(True, linestyle="--", alpha=0.25, color="gray")
    ax_line_trend.tick_params(colors="lightgray", axis="x", labelrotation=35, labelsize=9)
    ax_line_trend.tick_params(colors="lightgray", axis="y", labelsize=9)
    if len(y_vals):
        ymax = y_vals.max()
        ax_line_trend.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    for x, y in zip(x_vals, y_vals):
        if y > 0:
            ax_line_trend.annotate(f"{int(y)}m", xy=(x, y), xytext=(0, 5),
                                   textcoords="offset points", ha="center", va="bottom",
                                   fontsize=8, color="white",
                                   bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.4), ec="none"))

    # --- Stacked small column for last 7 days (compact) ---
    bottom = np.zeros(len(stack_df), dtype=float)
    for t in stack_titles:
        vals = stack_df[t].values
        ax_stack_week.bar(
            stack_df.index, vals, bottom=bottom,
            label=t, color=title_to_color.get(t, palette[0]),
            edgecolor="#1a1a1a", linewidth=0.4
        )
        bottom += vals
    ax_stack_week.set_title("Last 7 Days — Daily stack", **small_title)
    ax_stack_week.set_xlabel("", **label_style)
    ax_stack_week.set_ylabel("Minutes", **label_style)
    ax_stack_week.grid(axis="y", linestyle="--", alpha=0.25, color="gray")
    ax_stack_week.tick_params(colors="lightgray", axis="x", labelrotation=35, labelsize=8)
    ax_stack_week.tick_params(colors="lightgray", axis="y", labelsize=9)
    if len(stack_titles):
        ax_stack_week.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                             facecolor="#121212", edgecolor="white", title="Title", fontsize=8)

    # Reserve the right margin for legends (do not call tight_layout)
    fig.subplots_adjust(right=0.75)  # tweak 0.75 -> 0.70/0.78 if you need more/less room

    try:
        plt.show()
    finally:
        try:
            plt.close("all")
        except Exception:
            pass


def resource_path(relative_path: str) -> Path:
    base_path = getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
    return Path(base_path) / relative_path


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Concentria v2.0")
        self.geometry("1440x960")
        self.minsize(1100, 680)
        self.configure(padx=14, pady=14)
        self.day_index = {}
        self.entries = []
        self._suppress_save = False
        self._timer_after_id = None
        self._timer_running = False
        self._timer_total_secs = 0
        self._timer_remaining_secs = 0
        self._closing = False
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()
        self._quotes_after_id = None
        self.quote_interval_ms = 10 * 60 * 1000
        self.quotes = []
        self.quote_index = -1
        self._load_quotes()
        self._show_next_quote(schedule_next=True)
        self.load_entries_from_csv()

    def _build_ui(self):
        style = ttk.Style(self)
        try:
            self.call("tk", "scaling", 1.25)
        except tk.TclError:
            pass
        BG = "#0d1117"
        SURFACE = "#161b22"
        FIELD = "#1e242c"
        HEAD_BG = "#151a21"
        BORDER = "#2d333b"
        ACCENT = "#f59e0b"
        ACCENT_HOV = "#fbbf24"
        ACCENT_PRS = "#d97706"
        DANGER = "#ef4444"
        FG = "#e5e7eb"
        FG_MUTED = "#ffffff"
        ROW_ODD = "#181e25"
        ROW_EVEN = "#1e242c"
        SELECT_BG = "#b45309"
        SELECT_FG = "#ffffff"
        self.configure(bg=BG)
        style.theme_use("clam")
        if platform.system() == "Darwin":
            base_font = ("Helvetica", 12)
            label_font = ("Helvetica", 12)
            title_font = ("Helvetica", 13, "bold")
            header_font = ("Helvetica", 12, "bold")
            big_font = ("Helvetica", 56, "bold")
            hero_font = ("Helvetica", 10, "bold")
            quote_font = ("Helvetica", 22, "italic")
        elif platform.system() == "Windows":
            base_font = ("Segoe UI", 12)
            label_font = ("Segoe UI", 12)
            title_font = ("Segoe UI", 13, "bold")
            header_font = ("Segoe UI", 12, "bold")
            big_font = ("Segoe UI", 56, "bold")
            hero_font = ("Segoe UI", 7, "bold")
            quote_font = ("Segoe UI", 22, "italic")
        else:
            base_font = ("Ubuntu", 12)
            label_font = ("Ubuntu", 12)
            title_font = ("Ubuntu", 13, "bold")
            header_font = ("Ubuntu", 12, "bold")
            big_font = ("Ubuntu", 56, "bold")
            hero_font = ("Ubuntu", 10, "bold")
            quote_font = ("Ubuntu", 22, "italic")
        style.configure(".", background=BG, foreground=FG, font=base_font)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=label_font)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("Card.TLabelframe", background=SURFACE, borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=SURFACE, foreground=FG, font=title_font)

        style.configure("TEntry", fieldbackground=FIELD, foreground=FG, background=FIELD)
        try:
            style.configure("TEntry", bordercolor=BORDER, lightcolor=ACCENT, darkcolor=BORDER)
        except tk.TclError:
            pass

        self.option_add("*TEntry*insertBackground", FG)
        style.configure("TButton", padding=(12, 10), borderwidth=0, focusthickness=0)
        style.configure("Primary.TButton", background=ACCENT, foreground="#001018")
        style.map("Primary.TButton", background=[("active", ACCENT_HOV), ("pressed", ACCENT_PRS)], foreground=[("disabled", FG_MUTED)])
        style.configure("Secondary.TButton", background="#0f1b2d", foreground=FG)
        style.map("Secondary.TButton", background=[("active", "#13223a"), ("pressed", "#0d192d")])
        style.configure("Danger.TButton", background=DANGER, foreground="#0b0f14")
        style.map("Danger.TButton", background=[("active", "#f87171"), ("pressed", "#dc2626")])
        style.configure("TProgressbar", troughcolor="#07101b", background=ACCENT, borderwidth=0)
        style.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=FG, rowheight=34, borderwidth=0)
        style.map("Treeview", background=[("selected", SELECT_BG)], foreground=[("selected", SELECT_FG)])
        style.configure("Treeview.Heading", background=HEAD_BG, foreground=FG, font=header_font, relief="flat")
        style.map("Treeview.Heading", background=[("active", "#0e1a2b")])
        self.grid_columnconfigure(0, weight=3, minsize=740)
        self.grid_columnconfigure(1, weight=2, minsize=420)
        self.grid_rowconfigure(2, weight=1)
        header = ttk.Frame(self)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Concentria created by Ali Aliyev 2025", font=hero_font, foreground="#ffffff").grid(row=0, column=0, sticky="w")
        self.subtitle_lbl = ttk.Label(header, text="", foreground=FG_MUTED, font=quote_font, wraplength=1000, justify="left")
        self.subtitle_lbl.grid(row=1, column=0, sticky="w", pady=(2, 0))
        inputs_card = ttk.LabelFrame(self, text="Quick Add", style="Card.TLabelframe")
        inputs_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 12))
        for c in range(0, 7):
            inputs_card.columnconfigure(c, weight=1)
        ttk.Label(inputs_card, text="Title").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        self.title_var = tk.StringVar()
        ttk.Entry(inputs_card, textvariable=self.title_var).grid(row=1, column=0, columnspan=4, sticky="we", padx=12)
        ttk.Label(inputs_card, text="Duration (min)").grid(row=0, column=4, sticky="w", padx=12, pady=(12, 6))
        self.duration_var = tk.StringVar(value="0")
        ttk.Entry(inputs_card, textvariable=self.duration_var, width=12).grid(row=1, column=4, sticky="w", padx=12)
        ttk.Label(inputs_card, text="Hardness (1–10)").grid(row=0, column=5, sticky="w", padx=12, pady=(12, 6))
        self.hardness_var = tk.StringVar(value="5")
        self.hardness_cb = ttk.Combobox(inputs_card, textvariable=self.hardness_var, values=[str(i) for i in range(1, 11)], width=6, state="readonly")
        self.hardness_cb.grid(row=1, column=5, sticky="w", padx=12)
        ttk.Button(inputs_card, text="Add Entry", style="Primary.TButton", command=self.on_add).grid(row=1, column=6, sticky="ew", padx=(4, 12))
        ttk.Label(inputs_card, text="Note").grid(row=2, column=0, sticky="w", padx=12, pady=(12, 6))
        note_wrap = ttk.Frame(inputs_card, style="Card.TLabelframe")
        note_wrap.grid(row=3, column=0, columnspan=7, sticky="nsew", padx=12, pady=(0, 12))
        inputs_card.rowconfigure(3, weight=1)
        note_wrap.columnconfigure(0, weight=1)
        note_wrap.rowconfigure(0, weight=1)
        self.note_text = tk.Text(note_wrap, height=5, wrap="word", bg=FIELD, fg=FG, insertbackground=FG, relief="flat", padx=8, pady=8, font=base_font)
        self.note_text.grid(row=0, column=0, sticky="nsew")
        note_scroll = ttk.Scrollbar(note_wrap, orient="vertical", command=self.note_text.yview)
        note_scroll.grid(row=0, column=1, sticky="ns")
        self.note_text.configure(yscrollcommand=note_scroll.set)
        toolbar = ttk.Frame(self)
        toolbar.grid(row=2, column=0, sticky="ew", padx=(0, 10), pady=(0, 12))
        for i in range(7):
            toolbar.columnconfigure(i, weight=1)
        ttk.Button(toolbar, text="Remove Selected", style="Secondary.TButton", command=self.remove_selected).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(toolbar, text="Clear All", style="Danger.TButton", command=self.clear_all).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(toolbar, text="Reload CSV", style="Secondary.TButton", command=self.reload_csv).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(toolbar, text="Quit", style="Secondary.TButton", command=self.on_close).grid(row=0, column=5, sticky="e", padx=4)
        ttk.Button(toolbar, text="Analyze", style="Secondary.TButton", command=self.on_analyze).grid(row=0, column=6, sticky="e", padx=4)
        list_card = ttk.LabelFrame(self, text="Items (grouped by day)", style="Card.TLabelframe")
        list_card.grid(row=3, column=0, sticky="nsew", padx=(0, 10))
        self.grid_rowconfigure(3, weight=1)
        list_card.rowconfigure(0, weight=1)
        list_card.columnconfigure(0, weight=1)
        cols = ("Duration", "Hardness", "Title", "Note")
        self.tree = ttk.Treeview(list_card, columns=cols, show="tree headings", selectmode="extended")
        self.tree.heading("Duration", text="Duration")
        self.tree.heading("Hardness", text="Hardness")
        self.tree.heading("Title", text="Title")
        self.tree.heading("Note", text="Note")
        self.tree.heading("#0", text="Day / Item (HH:MM or Total)")
        self.tree.column("#0", width=320)
        self.tree.column("Duration", width=120, anchor="center")
        self.tree.column("Hardness", width=110, anchor="center")
        self.tree.column("Title", width=280)
        self.tree.column("Note", width=360)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        vsb = ttk.Scrollbar(list_card, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(list_card, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.grid(row=0, column=1, sticky="ns", pady=10)
        hsb.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=(0, 10))
        self.tree.tag_configure("oddrow", background=ROW_ODD)
        self.tree.tag_configure("evenrow", background=ROW_EVEN)
        timer = ttk.LabelFrame(self, text="Focus Timer", style="Card.TLabelframe")
        timer.grid(row=1, column=1, rowspan=3, sticky="nsew")
        for c in range(2):
            timer.columnconfigure(c, weight=1)
        ttk.Label(timer, text="Minutes").grid(row=0, column=0, sticky="w", padx=16, pady=(16, 6))
        self.timer_minutes_var = tk.StringVar(value="25")
        ttk.Entry(timer, textvariable=self.timer_minutes_var, width=14).grid(row=0, column=1, sticky="w", padx=(0, 16), pady=(16, 6))
        self.timer_display = ttk.Label(timer, text="00:00", font=big_font, foreground="#e6fbff")
        self.timer_display.grid(row=1, column=0, columnspan=2, pady=(4, 2))
        self.timer_progress = ttk.Progressbar(timer, mode="determinate", length=260)
        self.timer_progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))
        btns = ttk.Frame(timer)
        btns.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(6, 16))
        for i in range(3):
            btns.columnconfigure(i, weight=1)
        ttk.Button(btns, text="Start", style="Primary.TButton", command=self.timer_start).grid(row=0, column=0, sticky="ew", padx=6)
        self.btn_pause = ttk.Button(btns, text="Pause", style="Secondary.TButton", command=self.timer_pause, state="disabled")
        self.btn_pause.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(btns, text="Reset", style="Secondary.TButton", command=self.timer_reset).grid(row=0, column=2, sticky="ew", padx=6)
        self.log_when_done = tk.BooleanVar(value=False)
        ttk.Checkbutton(timer, text="On finish, auto-add entry", variable=self.log_when_done).grid(row=4, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 16))


    def _day_key(self, dt: datetime) -> str:
        return dt.strftime("%d-%m-%y")

    def _day_label(self, key: str, count: int) -> str:
        return f"{key} ({count})"

    def _parse_day_from_parent_text(self, text: str) -> str:
        return text.split(" ")[0]

    def _find_parent_for_day(self, day_key: str):
        for rid in self.tree.get_children(""):
            txt = self.tree.item(rid, "text") or ""
            if txt.startswith(day_key + " ") or txt == day_key or txt.startswith(day_key + " — "):
                return rid
        return None

    def _find_footer_id(self, parent_id: str):
        for cid in self.tree.get_children(parent_id):
            if (self.tree.item(cid, "text") or "").startswith("Total"):
                return cid
        return None

    def _parse_minutes(self, s: str) -> int:
        try:
            val = int(float(str(s).strip()))
            return max(0, val)
        except Exception:
            return 0

    def _day_total_minutes(self, day_key: str) -> int:
        return sum(self._parse_minutes(e.get("duration", 0)) for e in self.entries if e.get("date") == day_key)

    def save_entries_to_csv(self):
        if self._suppress_save:
            return
        try:
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for e in self.entries:
                    if "hardness" not in e:
                        e["hardness"] = ""
                    writer.writerow(e)
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save CSV: {e}")

    def load_entries_from_csv(self):
        self._suppress_save = True
        self.clear_visual_only()
        self.entries = []
        if os.path.exists(CSV_FILE):
            try:
                with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row:
                            continue
                        try:
                            date_key = (row.get("date", "") or "").strip()
                            clock = (row.get("clock", "") or "").strip()
                            title = (row.get("title", "") or "").strip()
                            duration = (row.get("duration", "") or "").strip()
                            note = (row.get("note", "") or "").strip()
                            hardness = (row.get("hardness", "") or "").strip()
                            self.entries.append({"date": date_key, "clock": clock, "title": title, "duration": duration, "note": note, "hardness": hardness})
                        except Exception:
                            continue
            except Exception as e:
                messagebox.showerror("Load error", f"Failed to read CSV: {e}")
        for e in self.entries:
            self._insert_visual(e["date"], e["clock"], e["title"], e["duration"], e["note"], e.get("hardness", ""))
        for day_key in {e["date"] for e in self.entries}:
            self._update_total_footer(day_key)
        self._suppress_save = False
        self._retag_tree()

    def reload_csv(self):
        self.load_entries_from_csv()

    def on_add(self):
        title = self.title_var.get().strip()
        note = self.note_text.get("1.0", "end-1c").strip()
        duration_raw = self.duration_var.get().strip()
        hardness_raw = (self.hardness_var.get() or "").strip()
        if not title:
            messagebox.showwarning("Missing title", "Please fill in the Title.")
            return
        if duration_raw == "":
            messagebox.showwarning("Missing duration", "Please fill in the Duration.")
            return
        try:
            h = int(hardness_raw)
            if h < 1 or h > 10:
                raise ValueError
        except Exception:
            h = 5
            self.hardness_var.set(str(h))
        hardness = str(h)
        now = datetime.now()
        date_key = self._day_key(now)
        clock = now.strftime("%H:%M")
        duration = duration_raw
        self.entries.append({"date": date_key, "clock": clock, "title": title, "duration": duration, "note": note, "hardness": hardness})
        self.save_entries_to_csv()
        self._insert_visual(date_key, clock, title, duration, note, hardness)
        self._update_total_footer(date_key)
        self.title_var.set("")
        self.duration_var.set("0")
        self.note_text.delete("1.0", "end")

    def _load_quotes(self):
        lines = []
        try:
            path = resource_path("quotes.txt")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        lines.append(s)
        except Exception:
            lines = []
        self.quotes = lines
        self.quote_index = -1

    def _show_next_quote(self, schedule_next=False):
        if not getattr(self, "subtitle_lbl", None):
            return
        if not getattr(self, "quotes", None):
            self._load_quotes()
        if self.quotes:
            if len(self.quotes) == 1:
                new_idx = 0
            else:
                last = getattr(self, "quote_index", -1)
                new_idx = random.randrange(len(self.quotes))
                if last != -1 and len(self.quotes) > 1:
                    if new_idx == last:
                        new_idx = random.randrange(len(self.quotes))
            self.quote_index = new_idx
            quote = self.quotes[new_idx]
            try:
                self.subtitle_lbl.configure(text=f'“{quote}”')
            except Exception:
                pass
        else:
            try:
                self.subtitle_lbl.configure(text="")
            except Exception:
                pass
        if schedule_next and not self._closing:
            try:
                self._quotes_after_id = self.after(self.quote_interval_ms, self._show_next_quote, True)
            except Exception:
                self._quotes_after_id = None

    def _insert_visual(self, day_key: str, clock: str, title: str, duration: str, note: str, hardness: str):
        if day_key not in self.day_index:
            iid = self.tree.insert("", "end", text=f"{day_key}", values=(duration, hardness, title, note))
            self.day_index[day_key] = {"mode": "single", "item_id": iid, "clock": clock}
            self._retag_tree()
            return
        state = self.day_index[day_key]
        if state["mode"] == "single":
            single_id = state["item_id"]
            first_clock = state.get("clock", "")
            old_vals = self.tree.item(single_id, "values")
            index = self.tree.index(single_id)
            parent_id = self.tree.insert("", index, text=self._day_label(day_key, 2), open=True)
            child1 = self.tree.insert(parent_id, "end", text=first_clock or "--:--", values=old_vals)
            self.tree.insert(parent_id, "end", text=clock, values=(duration, hardness, title, note))
            self.tree.delete(single_id)
            self.day_index[day_key] = {"mode": "group", "parent_id": parent_id, "children": [child1]}
            self.day_index[day_key]["children"].append(self.tree.get_children(parent_id)[-1])
            self._update_total_footer(day_key)
        else:
            parent_id = state.get("parent_id")
            if not parent_id or not self.tree.exists(parent_id):
                recovered = self._find_parent_for_day(day_key)
                if recovered:
                    parent_id = recovered
                    state["parent_id"] = parent_id
                    state["children"] = [cid for cid in self.tree.get_children(parent_id) if not (self.tree.item(cid, "text") or "").startswith("Total")]
                else:
                    parent_id = self.tree.insert("", "end", text=self._day_label(day_key, 0), open=True)
                    state["mode"] = "group"
                    state["parent_id"] = parent_id
                    state["children"] = []
            existing_footer = self._find_footer_id(parent_id)
            if existing_footer:
                idx = self.tree.index(existing_footer)
                child = self.tree.insert(parent_id, idx, text=clock, values=(duration, hardness, title, note))
            else:
                child = self.tree.insert(parent_id, "end", text=clock, values=(duration, hardness, title, note))
            state["children"].append(child)
            self.tree.item(parent_id, text=self._day_label(day_key, len(state["children"])))
            self._update_total_footer(day_key)
        self._retag_tree()

    def remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        changed = False
        affected_days = set()
        for iid in list(sel):
            parent = self.tree.parent(iid)
            if parent:
                parent_text = self.tree.item(parent, "text")
                day_key = self._parse_day_from_parent_text(parent_text)
                vals = self.tree.item(iid, "values")
                duration, hardness, title, note = vals
                clock = self.tree.item(iid, "text")
                self.tree.delete(iid)
                state = self.day_index.get(day_key)
                if state and state.get("mode") == "group":
                    if iid in state["children"]:
                        state["children"].remove(iid)
                        if len(state["children"]) == 1:
                            remaining = state["children"][0]
                            r_vals = self.tree.item(remaining, "values")
                            r_clock = self.tree.item(remaining, "text")
                            index = self.tree.index(parent)
                            new_single = self.tree.insert("", index, text=f"{day_key}", values=r_vals)
                            self.tree.delete(remaining)
                            self.tree.delete(parent)
                            self.day_index[day_key] = {"mode": "single", "item_id": new_single, "clock": r_clock}
                        else:
                            self.tree.item(parent, text=self._day_label(day_key, len(state["children"])))
                if self._remove_first_matching_entry(day_key, clock, title, duration, note, hardness):
                    changed = True
                    affected_days.add(day_key)
            else:
                if self.tree.get_children(iid):
                    parent_text = self.tree.item(iid, "text")
                    day_key = self._parse_day_from_parent_text(parent_text)
                    for cid in list(self.tree.get_children(iid)):
                        ctext = self.tree.item(cid, "text") or ""
                        if ctext.startswith("Total"):
                            self.tree.delete(cid)
                            continue
                        dur, hardness, title, note = self.tree.item(cid, "values")
                        clock = ctext
                        if self._remove_first_matching_entry(day_key, clock, title, dur, note, hardness):
                            changed = True
                        self.tree.delete(cid)
                    self.tree.delete(iid)
                    if self.day_index.get(day_key, {}).get("parent_id") == iid:
                        del self.day_index[day_key]
                    affected_days.add(day_key)
                else:
                    text = self.tree.item(iid, "text")
                    day_key = text.split(" — ")[0]
                    vals = self.tree.item(iid, "values")
                    if vals:
                        duration, hardness, title, note = vals
                    else:
                        duration = hardness = title = note = ""
                    state = self.day_index.get(day_key, {})
                    stored_clock = state.get("clock", "")
                    self.tree.delete(iid)
                    if state.get("item_id") == iid:
                        del self.day_index[day_key]
                    if stored_clock:
                        if self._remove_first_matching_entry(day_key, stored_clock, title, duration, note, hardness):
                            changed = True
                            affected_days.add(day_key)
                    else:
                        if self._remove_first_matching_single(day_key, title, duration, note, hardness):
                            changed = True
                            affected_days.add(day_key)
        if changed:
            self.save_entries_to_csv()
            for day_key in affected_days:
                self._update_total_footer(day_key)
        self._retag_tree()

    def _calc_points(self, total_minutes: int, avg_hardness: float, alpha: float = 0.7, beta: float = 0.5) -> float:
        if total_minutes <= 0 or avg_hardness <= 0:
            return 0.0
        t_term = max(0.0, total_minutes) / 480.0
        h_term = max(0.1, avg_hardness) / 6.0
        raw = (t_term ** alpha) * (h_term ** beta)
        return min(100.0, 100.0 * raw)

    def _update_total_footer(self, day_key: str):
        total = self._day_total_minutes(day_key)
        hvals = []
        for e in self.entries:
            if e.get("date") == day_key:
                try:
                    h = float(str(e.get("hardness", "")).strip())
                    if 1.0 <= h <= 10.0:
                        hvals.append(h)
                except Exception:
                    pass
        avg_hardness = (sum(hvals) / len(hvals)) if hvals else 1.0
        points = self._calc_points(total, avg_hardness, alpha=0.7, beta=0.5)
        points_str = f"{points:.2f}"
        avg_h_str = f"{avg_hardness:.2f}"
        state = self.day_index.get(day_key)
        if not state:
            return
        if state.get("mode") == "group":
            parent_id = state.get("parent_id")
            if not parent_id or not self.tree.exists(parent_id):
                return
            footer_id = self._find_footer_id(parent_id)
            footer_text = f"Total • Points: {points_str} (Avg H: {avg_h_str})"
            footer_vals = (str(total), "", "", "")
            if footer_id:
                self.tree.item(footer_id, text=footer_text, values=footer_vals)
                self.tree.move(footer_id, parent_id, "end")
            else:
                self.tree.insert(parent_id, "end", text=footer_text, values=footer_vals)
        else:
            iid = state.get("item_id")
            if iid and self.tree.exists(iid):
                self.tree.item(iid, text=f"{day_key} — Total: {total} — Points: {points_str} (Avg H: {avg_h_str})")
        self._retag_tree()

    def _remove_first_matching_entry(self, day_key, clock, title, duration, note, hardness) -> bool:
        for i, e in enumerate(self.entries):
            if (e.get("date") == day_key and e.get("clock") == clock and e.get("title") == title and
                str(e.get("duration")) == str(duration) and e.get("note") == note and
                str(e.get("hardness", "")) == str(hardness)):
                del self.entries[i]
                return True
        for i, e in enumerate(self.entries):
            if (e.get("date") == day_key and e.get("clock") == clock and e.get("title") == title and
                str(e.get("duration")) == str(duration) and e.get("note") == note):
                del self.entries[i]
                return True
        return False

    def _remove_first_matching_single(self, day_key, title, duration, note, hardness) -> bool:
        for i, e in enumerate(self.entries):
            if (e.get("date") == day_key and e.get("title") == title and
                str(e.get("duration")) == str(duration) and e.get("note") == note and
                str(e.get("hardness", "")) == str(hardness)):
                del self.entries[i]
                return True
        for i, e in enumerate(self.entries):
            if (e.get("date") == day_key and e.get("title") == title and
                str(e.get("duration")) == str(duration) and e.get("note") == note):
                del self.entries[i]
                return True
        return False

    def clear_visual_only(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.day_index.clear()

    def clear_all(self):
        self.clear_visual_only()
        self.entries = []
        self.save_entries_to_csv()
        self._retag_tree()

    def _format_mmss(self, secs: int) -> str:
        secs = max(0, int(secs))
        m = secs // 60
        s = secs % 60
        return f"{m:02d}:{s:02d}"

    def timer_start(self):
        if self._timer_running and self._timer_after_id:
            return
        if self._timer_remaining_secs <= 0:
            try:
                mins = float(self.timer_minutes_var.get().strip())
            except Exception:
                messagebox.showwarning("Invalid minutes", "Please enter a number of minutes (e.g., 25).")
                return
            if mins <= 0:
                messagebox.showwarning("Invalid minutes", "Minutes must be greater than 0.")
                return
            self._timer_total_secs = int(mins * 60)
            self._timer_remaining_secs = self._timer_total_secs
            self.timer_progress.configure(maximum=self._timer_total_secs)
            self.timer_progress['value'] = 0
            self.timer_display.configure(text=self._format_mmss(self._timer_remaining_secs))
        self._timer_running = True
        self.btn_pause.configure(state="normal", text="Pause")
        self._schedule_tick()

    def timer_pause(self):
        if not self._timer_running:
            if self._timer_remaining_secs > 0 and not self._closing:
                self._timer_running = True
                self.btn_pause.configure(text="Pause")
                self._schedule_tick()
            return
        self._timer_running = False
        self.btn_pause.configure(text="Resume")
        if self._timer_after_id:
            try:
                self.after_cancel(self._timer_after_id)
            except Exception:
                pass
            self._timer_after_id = None

    def timer_reset(self):
        self._timer_running = False
        if self._timer_after_id:
            try:
                self.after_cancel(self._timer_after_id)
            except Exception:
                pass
            self._timer_after_id = None
        self._timer_total_secs = 0
        self._timer_remaining_secs = 0
        if self.winfo_exists():
            self.timer_display.configure(text="00:00")
            self.timer_progress['value'] = 0
            self.btn_pause.configure(state="disabled", text="Pause")

    def on_analyze(self):
        if not os.path.exists(CSV_FILE):
            messagebox.showinfo("Nothing to analyze", "No data yet. Add at least one entry first.")
            return
        try:
            p = multiprocessing.Process(target=run_dashboard, args=(CSV_FILE,), daemon=True)
            p.start()
        except Exception as exc:
            messagebox.showerror("Analyze error", f"Failed to start analysis process:\n{exc}")

    def on_close(self):
        self._closing = True
        self._timer_running = False
        if self._timer_after_id:
            try:
                self.after_cancel(self._timer_after_id)
            except Exception:
                pass
            self._timer_after_id = None
        if getattr(self, "_quotes_after_id", None):
            try:
                self.after_cancel(self._quotes_after_id)
            except Exception:
                pass
            self._quotes_after_id = None

        try:
            self.destroy()
        except Exception:
            pass

    def _schedule_tick(self):
        if not self._timer_running or self._closing:
            return
        self._timer_after_id = self.after(1000, self._tick)

    def _tick(self):
        if not self._timer_running or self._closing or not self.winfo_exists():
            return
        self._timer_remaining_secs -= 1
        if not self.winfo_exists():
            return
        self.timer_display.configure(text=self._format_mmss(self._timer_remaining_secs))
        elapsed = max(0, self._timer_total_secs - self._timer_remaining_secs)
        self.timer_progress['value'] = elapsed
        if self._timer_remaining_secs <= 0:
            self._timer_running = False
            if self.winfo_exists():
                self.btn_pause.configure(state="disabled", text="Pause")
                try:
                    self.bell()
                except Exception:
                    pass
            if self.log_when_done.get():
                mins = max(1, self._timer_total_secs // 60)
                self.duration_var.set(str(mins))
                if not self.title_var.get().strip():
                    self.title_var.set("Timed Session")
                self.on_add()
            return
        self._schedule_tick()

    def _retag_tree(self):
        for i, rid in enumerate(self.tree.get_children("")):
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            self.tree.item(rid, tags=(tag,))
            for j, cid in enumerate(self.tree.get_children(rid)):
                ctag = "evenrow" if j % 2 == 0 else "oddrow"
                self.tree.item(cid, tags=(ctag,))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    App().mainloop()

