

"""
modules/recovery_card_generator.py
FootballIQ — Sophisticated visual recovery card (matplotlib)
─────────────────────────────────────────────────────────────
Generates a multi-section dark-theme PDF-quality card:
  ① Header banner — player ID, overall risk tier (colour-coded)
  ② Tissue risk radar + horizontal score bars (5 tissues, each coloured)
  ③ Injury flag panels (each tissue gets its own coloured row)
  ④ Day-by-day return-to-train schedule (intensity gradient bar)
  ⑤ Pain-relief protocol (immediate / 48h / medication per tissue)
  ⑥ Workload fingerprint (speed band breakdown)
  ⑦ Nutrition & supplement note
  ⑧ Footer
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec


# ── Palette ───────────────────────────────────────────────────────────────────
BG        = '#0D1117'   # page background
PANEL     = '#161B22'   # card/panel background
PANEL2    = '#1C2330'   # alternate panel
BORDER    = '#21262D'
TEXT      = '#E6EDF3'
MUTED     = '#7D8590'
ACCENT    = '#00E5FF'

TIER_COL = {
    'CRITICAL': '#E74C3C',
    'HIGH':     '#E67E22',
    'MODERATE': '#F1C40F',
    'LOW':      '#2ECC71',
    'MINIMAL':  '#1ABC9C',
}
TIER_BG = {
    'CRITICAL': '#3D0D0D',
    'HIGH':     '#3D1F0D',
    'MODERATE': '#3D3000',
    'LOW':      '#0D3D1A',
    'MINIMAL':  '#0D2E2B',
}

TISSUE_COLS = {
    'Hamstring':     '#FF3354',
    'Quadriceps':    '#FF7043',
    'Calf/Achilles': '#FFAB00',
    'Hip Flexors':   '#C77DFF',
    'Knee/Ligament': '#00E5FF',
}
BAND_COLS = ['#3D5C72', '#5A7A90', '#00E5FF', '#FFAB00', '#FF3354']


def _tier(score):
    if score >= 75: return 'CRITICAL'
    if score >= 55: return 'HIGH'
    if score >= 35: return 'MODERATE'
    if score >= 18: return 'LOW'
    return 'MINIMAL'


def _rounded_bar(ax, x, y, width, height, color, radius=0.003, alpha=1.0, zorder=2):
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle=f'round,pad={radius}',
                          facecolor=color, edgecolor='none',
                          alpha=alpha, zorder=zorder)
    ax.add_patch(box)


class RecoveryCardGenerator:

    def __init__(self):
        print("🎨 RecoveryCardGenerator — dark-theme visual card loaded")

    # ── Main entry point ──────────────────────────────────────────────────────
    def generate_card(self, recovery_plan, output_path=None):
        try:
            fig = plt.figure(figsize=(18, 26), facecolor=BG)
            fig.subplots_adjust(left=0.04, right=0.96, top=0.97, bottom=0.02,
                                hspace=0.55, wspace=0.35)

            gs = GridSpec(7, 2, figure=fig,
                          height_ratios=[0.9, 2.2, 2.0, 1.8, 2.4, 2.0, 0.5],
                          hspace=0.55, wspace=0.35)

            self._header(fig, gs[0, :], recovery_plan)
            self._tissue_bars(fig, gs[1, 0], recovery_plan)
            self._workload_panel(fig, gs[1, 1], recovery_plan)
            self._injury_flags(fig, gs[2, :], recovery_plan)
            self._schedule_panel(fig, gs[3, :], recovery_plan)
            self._pain_relief_panel(fig, gs[4, :], recovery_plan)
            self._speed_bands(fig, gs[5, 0], recovery_plan)
            self._nutrition_panel(fig, gs[5, 1], recovery_plan)
            self._footer(fig, gs[6, :])

            plt.close('all')  # prevent display leak

            if output_path:
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches='tight',
                            facecolor=BG, edgecolor='none')
                print(f"[OK] Recovery card saved: {output_path}")
            return fig

        except Exception as e:
            import traceback
            print(f"[!] RecoveryCardGenerator error: {e}")
            traceback.print_exc()
            return None

    # ── 1. Header ─────────────────────────────────────────────────────────────
    def _header(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(BG); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        tier  = p.get('risk_tier', 'LOW')
        col   = TIER_COL.get(tier, '#888')
        pid   = p.get('player_id', '?')
        pos   = p.get('position', '')
        score = p.get('overall_risk', 0)
        rest  = p.get('rest_days', 0)

        # Coloured tier banner strip
        _rounded_bar(ax, 0, 0.55, 1, 0.45, TIER_BG.get(tier, PANEL), radius=0.005)
        ax.axhline(0.55, color=col, linewidth=3, alpha=0.9)

        ax.text(0.50, 0.85, f'[R]  PERSONAL RECOVERY PLAN', fontsize=22,
                fontweight='bold', ha='center', va='center', color=TEXT,
                fontfamily='monospace', transform=ax.transAxes)
        ax.text(0.50, 0.65, f'Player #{pid}  ·  {pos}  ·  Injury Risk Score {score:.0f}/100',
                fontsize=12, ha='center', va='center', color=MUTED,
                fontfamily='monospace', transform=ax.transAxes)

        # Tier badge
        ax.text(0.12, 0.25, f'{tier}  RISK', fontsize=20, fontweight='black',
                ha='center', va='center', color=col, fontfamily='monospace',
                transform=ax.transAxes)
        ax.text(0.38, 0.25, f'{score:.0f}/100', fontsize=20, fontweight='bold',
                ha='center', va='center', color=TEXT, fontfamily='monospace',
                transform=ax.transAxes)
        ax.text(0.62, 0.25, f'{rest}d  REST', fontsize=20, fontweight='bold',
                ha='center', va='center', color=col, fontfamily='monospace',
                transform=ax.transAxes)
        ax.text(0.86, 0.25, f"READY: {p.get('next_match_ready','—')}",
                fontsize=13, fontweight='bold', ha='center', va='center',
                color=TEXT, fontfamily='monospace', transform=ax.transAxes)

    # ── 2. Tissue risk bars ────────────────────────────────────────────────────
    def _tissue_bars(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        tissues = p.get('tissue_scores', {})
        if not tissues:
            ax.text(0.5, 0.5, 'No tissue data', ha='center', va='center',
                    color=MUTED, transform=ax.transAxes)
            return

        ax.text(0.05, 0.95, '[T]  TISSUE RISK SCORES', fontsize=11,
                fontweight='bold', color=ACCENT, va='top',
                fontfamily='monospace', transform=ax.transAxes)

        tissue_order = ['Hamstring', 'Quadriceps', 'Calf/Achilles', 'Hip Flexors', 'Knee/Ligament']
        n = len(tissue_order)
        row_h = 0.80 / n
        y_start = 0.88

        for i, name in enumerate(tissue_order):
            score = tissues.get(name, 0)
            t     = _tier(score)
            col   = TISSUE_COLS.get(name, ACCENT)
            y     = y_start - i * row_h - 0.04

            # Background track
            _rounded_bar(ax, 0.04, y - 0.025, 0.92, 0.055, PANEL2, radius=0.002, alpha=0.7)
            # Filled bar (0–100 mapped to 0–0.84 width)
            bar_w = max(score / 100 * 0.84, 0.004)
            _rounded_bar(ax, 0.04, y - 0.020, bar_w, 0.044, col, radius=0.002, alpha=0.85)

            ax.text(0.06, y + 0.008, name, fontsize=9.5, fontweight='bold',
                    color=TEXT, va='center', fontfamily='monospace', transform=ax.transAxes)
            ax.text(0.94, y + 0.008, f'{score:.0f}  {t}', fontsize=9,
                    color=col, va='center', ha='right', fontfamily='monospace',
                    transform=ax.transAxes)

    # ── 3. Workload panel ─────────────────────────────────────────────────────
    def _workload_panel(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.05, 0.95, '⚡  WORKLOAD FINGERPRINT', fontsize=11,
                fontweight='bold', color=ACCENT, va='top',
                fontfamily='monospace', transform=ax.transAxes)

        wl = p.get('workload', {})
        items = [
            ('Total Dist',    f"{wl.get('total_distance_km', 0):.2f} km"),
            ('Hi-Int Dist',   f"{wl.get('hi_intensity_km', 0):.2f} km"),
            ('VHI Dist',      f"{wl.get('very_hi_intensity_km', 0):.2f} km"),
            ('Sprint Dist',   f"{wl.get('sprint_distance_km', 0):.2f} km"),
            ('Sprint Count',  str(wl.get('sprint_count', 0))),
            ('Max Speed',     f"{wl.get('max_speed_kmh', 0):.1f} km/h"),
            ('Hard Accels',   str(wl.get('hard_accelerations', 0))),
            ('Hard Decels',   str(wl.get('hard_decelerations', 0))),
            ('Dir. Changes',  str(wl.get('direction_changes', 0))),
        ]
        cols = ['#00E5FF', '#00E676', '#FFAB00', '#FF7043',
                '#FF3354', '#C77DFF', '#FF9800', '#F44336', '#B0BEC5']

        n = len(items)
        for i, ((lbl, val), col) in enumerate(zip(items, cols)):
            row = i // 3
            col_idx = i % 3
            x = 0.05 + col_idx * 0.32
            y = 0.84 - row * 0.22

            _rounded_bar(ax, x - 0.01, y - 0.10, 0.30, 0.19, PANEL2,
                         radius=0.003, alpha=0.8)
            ax.text(x + 0.13, y + 0.04, val, fontsize=11, fontweight='bold',
                    ha='center', va='center', color=col,
                    fontfamily='monospace', transform=ax.transAxes)
            ax.text(x + 0.13, y - 0.05, lbl, fontsize=8, ha='center',
                    va='center', color=MUTED, fontfamily='monospace',
                    transform=ax.transAxes)

    # ── 4. Injury flags ───────────────────────────────────────────────────────
    def _injury_flags(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.02, 0.97, '[!]  INJURY RISK FLAGS', fontsize=11,
                fontweight='bold', color=ACCENT, va='top',
                fontfamily='monospace', transform=ax.transAxes)

        flags = p.get('injury_flags', [])
        if not flags:
            ax.text(0.5, 0.5, '[/] No significant injury flags — good recovery expected',
                    ha='center', va='center', color=TIER_COL['MINIMAL'],
                    fontsize=11, fontfamily='monospace', transform=ax.transAxes)
            return

        n = len(flags[:5])
        col_w = 0.96 / max(n, 1)

        for i, flag in enumerate(flags[:5]):
            t    = flag.get('severity', flag.get('tier', 'LOW'))
            col  = TIER_COL.get(t, '#888')
            bg   = TIER_BG.get(t, PANEL2)
            x    = 0.02 + i * (col_w + 0.005)
            prob = flag.get('probability_pct', 0)
            score = flag.get('score', 0)

            _rounded_bar(ax, x, 0.08, col_w - 0.01, 0.82, bg, radius=0.004, alpha=0.9)
            ax.axhline(0.90, xmin=x + 0.01, xmax=x + col_w - 0.01,
                       color=col, linewidth=2.5, alpha=0.9)

            # Tier label
            ax.text(x + col_w * 0.5, 0.90, t, fontsize=9.5, fontweight='black',
                    ha='center', va='bottom', color=col,
                    fontfamily='monospace', transform=ax.transAxes)

            # Injury name (wrap long names)
            name = flag.get('type', flag.get('tissue', ''))
            ax.text(x + col_w * 0.5, 0.78, name, fontsize=9, fontweight='bold',
                    ha='center', va='center', color=TEXT,
                    fontfamily='monospace', transform=ax.transAxes)

            # Risk score circle
            circle = plt.Circle((x + col_w * 0.5, 0.57), 0.055,
                                  transform=ax.transAxes,
                                  color=col, alpha=0.2, zorder=2)
            ax.add_patch(circle)
            ax.text(x + col_w * 0.5, 0.57, f'{score:.0f}', fontsize=14,
                    fontweight='bold', ha='center', va='center',
                    color=col, fontfamily='monospace', transform=ax.transAxes)

            # Probability bar
            ax.text(x + col_w * 0.5, 0.43, f'{prob:.0f}% risk', fontsize=9,
                    ha='center', va='center', color=TEXT,
                    fontfamily='monospace', transform=ax.transAxes)
            bar_full_w = col_w * 0.78
            _rounded_bar(ax, x + col_w * 0.11, 0.32, bar_full_w, 0.04,
                         PANEL2, radius=0.002, alpha=0.8)
            _rounded_bar(ax, x + col_w * 0.11, 0.32,
                         bar_full_w * prob / 60.0, 0.04, col, radius=0.002, alpha=0.85)

            # Cause text (clipped to width)
            cause = flag.get('cause', '')[:48]
            ax.text(x + col_w * 0.5, 0.20, cause, fontsize=7,
                    ha='center', va='center', color=MUTED,
                    fontfamily='monospace', wrap=True,
                    transform=ax.transAxes)

    # ── 5. Return-to-train schedule ───────────────────────────────────────────
    def _schedule_panel(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.02, 0.97, '[>]  RETURN-TO-TRAIN SCHEDULE', fontsize=11,
                fontweight='bold', color=ACCENT, va='top',
                fontfamily='monospace', transform=ax.transAxes)

        sched = p.get('schedule', [])
        if not sched:
            ax.text(0.5, 0.5, 'No schedule', ha='center', va='center',
                    color=MUTED, transform=ax.transAxes)
            return

        n   = len(sched)
        col_w = 0.96 / n
        grad_cols = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71',
                     '#1ABC9C', '#00E5FF', '#00B4D8', '#0096B7']

        for i, day in enumerate(sched):
            pct = float(str(day.get('intensity', '0')).replace('%', '') or 0)
            col = grad_cols[min(i, len(grad_cols) - 1)]
            x   = 0.02 + i * col_w

            # Day column background
            _rounded_bar(ax, x, 0.08, col_w - 0.01, 0.84, PANEL2,
                         radius=0.003, alpha=0.7)

            # Intensity fill bar at bottom
            bar_h = max(0.01, pct / 100 * 0.45)
            _rounded_bar(ax, x + 0.005, 0.09, col_w - 0.02, bar_h,
                         col, radius=0.002, alpha=0.35)

            # Day label
            ax.text(x + col_w * 0.5, 0.88, day.get('label', f'D{i+1}'),
                    fontsize=9, fontweight='bold', ha='center', va='center',
                    color=col, fontfamily='monospace', transform=ax.transAxes)

            # Activity text
            act = day.get('activity', '')
            words = act.split()
            lines, cur = [], []
            for w in words:
                cur.append(w)
                if len(' '.join(cur)) > 14:
                    lines.append(' '.join(cur[:-1])); cur = [w]
            if cur: lines.append(' '.join(cur))
            for li, line in enumerate(lines[:3]):
                ax.text(x + col_w * 0.5, 0.73 - li * 0.09, line,
                        fontsize=7.5, ha='center', va='center',
                        color=TEXT, fontfamily='monospace',
                        transform=ax.transAxes)

            # Intensity badge
            ax.text(x + col_w * 0.5, 0.36, day.get('intensity', '?'),
                    fontsize=11, fontweight='bold', ha='center', va='center',
                    color=col, fontfamily='monospace', transform=ax.transAxes)

            # Notes (truncated)
            note = day.get('notes', '')[:30]
            ax.text(x + col_w * 0.5, 0.17, note, fontsize=6.5,
                    ha='center', va='center', color=MUTED,
                    fontfamily='monospace', transform=ax.transAxes)

    # ── 6. Pain-relief protocol cards ────────────────────────────────────────
    def _pain_relief_panel(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.02, 0.98, '[+]  TARGETED PAIN-RELIEF PROTOCOL', fontsize=11,
                fontweight='bold', color=ACCENT, va='top',
                fontfamily='monospace', transform=ax.transAxes)

        relief = p.get('pain_relief', [])[:4]
        if not relief:
            ax.text(0.5, 0.5, 'No specific pain-relief protocol required',
                    ha='center', va='center', color=TIER_COL['MINIMAL'],
                    fontsize=11, fontfamily='monospace', transform=ax.transAxes)
            return

        n = len(relief)
        col_w = 0.96 / n

        for i, pr in enumerate(relief):
            col  = TISSUE_COLS.get(pr['target'], ACCENT)
            t    = pr.get('tier', 'LOW')
            bg   = TIER_BG.get(t, PANEL2)
            x    = 0.02 + i * col_w
            sc   = pr.get('score', 0)

            _rounded_bar(ax, x, 0.04, col_w - 0.01, 0.88, bg, radius=0.004, alpha=0.8)
            ax.axhline(0.92, xmin=x + 0.005, xmax=x + col_w - 0.01,
                       color=col, linewidth=2.5)

            ax.text(x + col_w * 0.5, 0.91, pr['target'], fontsize=10,
                    fontweight='bold', ha='center', va='bottom',
                    color=col, fontfamily='monospace', transform=ax.transAxes)
            ax.text(x + col_w * 0.5, 0.83, f'Score {sc:.0f}/100  ·  {t}',
                    fontsize=8, ha='center', va='center', color=MUTED,
                    fontfamily='monospace', transform=ax.transAxes)

            # IMMEDIATE
            ax.text(x + 0.01, 0.78, '>> IMMEDIATE (0–6h)', fontsize=7.5,
                    fontweight='bold', color='#00E5FF', va='top',
                    fontfamily='monospace', transform=ax.transAxes)
            y_txt = 0.72
            for item in pr.get('immediate', [])[:3]:
                wrapped = item[:40]
                ax.text(x + 0.015, y_txt, f'• {wrapped}', fontsize=7,
                        color=TEXT, va='top', fontfamily='monospace',
                        transform=ax.transAxes)
                y_txt -= 0.075

            # NEXT 48h
            ax.text(x + 0.01, 0.47, '>> NEXT 48h', fontsize=7.5,
                    fontweight='bold', color='#2ECC71', va='top',
                    fontfamily='monospace', transform=ax.transAxes)
            y_txt = 0.41
            for item in pr.get('next_48h', [])[:3]:
                wrapped = item[:40]
                ax.text(x + 0.015, y_txt, f'• {wrapped}', fontsize=7,
                        color=TEXT, va='top', fontfamily='monospace',
                        transform=ax.transAxes)
                y_txt -= 0.075

            # Medication
            _rounded_bar(ax, x + 0.01, 0.04, col_w - 0.025, 0.12,
                         '#3D3000', radius=0.003, alpha=0.9)
            ax.text(x + 0.015, 0.13, '[+] ' + pr.get('medication', '')[:55],
                    fontsize=6.5, color='#F1C40F', va='top',
                    fontfamily='monospace', transform=ax.transAxes)

    # ── 7. Speed bands ────────────────────────────────────────────────────────
    def _speed_bands(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.05, 0.96, '[~]  SPEED BAND BREAKDOWN', fontsize=10,
                fontweight='bold', color=ACCENT, va='top',
                fontfamily='monospace', transform=ax.transAxes)

        wl    = p.get('workload', {})
        bands = wl.get('speed_bands', {})
        names = ['walk', 'jog', 'run', 'fast', 'sprint']
        labels = ['Walk\n<7', 'Jog\n7–14', 'Run\n14–20', 'Fast\n20–25', 'Sprint\n>25']

        for i, (name, lbl) in enumerate(zip(names, labels)):
            val = bands.get(name, 0)
            col = BAND_COLS[i]
            y   = 0.82 - i * 0.155

            _rounded_bar(ax, 0.27, y - 0.03, 0.68, 0.09, PANEL2, radius=0.003, alpha=0.8)
            bw = max(val / 100 * 0.68, 0.003)
            _rounded_bar(ax, 0.27, y - 0.03, bw, 0.09, col, radius=0.003, alpha=0.85)

            ax.text(0.24, y + 0.015, lbl, fontsize=8, ha='right', va='center',
                    color=col, fontfamily='monospace', transform=ax.transAxes)
            ax.text(0.96, y + 0.015, f'{val:.0f}%', fontsize=9, ha='right',
                    va='center', color=col, fontfamily='monospace',
                    transform=ax.transAxes)

    # ── 8. Nutrition panel ────────────────────────────────────────────────────
    def _nutrition_panel(self, fig, gs_item, p):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.05, 0.96, '[N]  NUTRITION & SUPPLEMENTS', fontsize=10,
                fontweight='bold', color=ACCENT, va='top',
                fontfamily='monospace', transform=ax.transAxes)

        diet = p.get('diet_suggestions', {})
        macro = diet.get('macronutrients', {})
        supps = diet.get('supplements', [])

        y = 0.84
        for lbl, key in [('Protein', 'protein'), ('Carbs', 'carbs'), ('Hydration', 'hydration')]:
            val = macro.get(key, '—')
            ax.text(0.06, y, f'{lbl}:', fontsize=9, fontweight='bold',
                    color=MUTED, va='top', fontfamily='monospace',
                    transform=ax.transAxes)
            ax.text(0.32, y, val, fontsize=9, color=TEXT, va='top',
                    fontfamily='monospace', transform=ax.transAxes)
            y -= 0.10

        ax.text(0.06, y - 0.01, 'Supplements:', fontsize=9, fontweight='bold',
                color=MUTED, va='top', fontfamily='monospace',
                transform=ax.transAxes)
        y -= 0.10
        for s in supps[:6]:
            ax.text(0.08, y, f'• {s}', fontsize=8.5, color='#00E676',
                    va='top', fontfamily='monospace', transform=ax.transAxes)
            y -= 0.09

        note = p.get('nutrition_note', '')
        if note:
            _rounded_bar(ax, 0.03, 0.02, 0.94, 0.13, '#0D3D1A', radius=0.003)
            # wrap note
            words = note.split()
            lines, cur = [], []
            for w in words:
                cur.append(w)
                if len(' '.join(cur)) > 55:
                    lines.append(' '.join(cur[:-1])); cur = [w]
            if cur: lines.append(' '.join(cur))
            for li, ln in enumerate(lines[:2]):
                ax.text(0.06, 0.13 - li * 0.05, ln, fontsize=8,
                        color=TIER_COL['LOW'], va='top',
                        fontfamily='monospace', transform=ax.transAxes)

    # ── 9. Footer ─────────────────────────────────────────────────────────────
    def _footer(self, fig, gs_item):
        ax = fig.add_subplot(gs_item)
        ax.set_facecolor(BG); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.7, '[o]  FootballIQ Recovery System  ·  AI-generated report',
                ha='center', va='center', fontsize=9, color=MUTED,
                fontfamily='monospace', style='italic', transform=ax.transAxes)
        ax.text(0.5, 0.2, 'Consult a qualified physiotherapist or sports medicine doctor for serious injuries',
                ha='center', va='center', fontsize=8, color=MUTED,
                fontfamily='monospace', transform=ax.transAxes)

    # ── Simple text fallback ──────────────────────────────────────────────────
    def generate_simple_text_report(self, p):
        lines = [
            '=' * 64,
            f"RECOVERY PLAN — Player #{p.get('player_id', '?')}",
            f"Risk Tier: {p.get('risk_tier','?')}  ·  Score: {p.get('overall_risk',0):.0f}/100",
            f"Position:  {p.get('position','?')}",
            f"Rest:      {p.get('rest_days',0)} days",
            '=' * 64,
        ]
        for m in p.get('key_metrics', []):
            lines.append(f'  • {m}')
        lines.append('')
        for f in p.get('injury_flags', []):
            lines.append(f"  [!]  {f['type']} — {f['severity']} ({f['probability_pct']}%): {f['cause']}")
        lines.append('')
        for d in p.get('schedule', []):
            lines.append(f"  {d['label']}: {d['activity']} [{d['intensity']}] — {d['notes']}")
        lines.append('=' * 64)
        return '\n'.join(lines)