

# modules/heatmap_generator.py
"""
HeatmapGenerator — mirror-corrected version
─────────────────────────────────────────────
Root cause of mirror image:
  numpy histogram2d returns data[x, y].
  imshow(origin='lower') + invert_yaxis() means:
    - x goes left → right  ✓
    - y goes bottom → top, then gets flipped → goes top → bottom

  The raw Y from ViewTransformer is 0 at the top of the pitch.
  After invert_yaxis() the bottom of the screen becomes Y=0,
  which is visually the TOP of the pitch → mirror.

Fix: flip Y before building the histogram:
    pos[:, 1] = PITCH_HEIGHT - pos[:, 1]

This is applied in the API server endpoint AND here as a safety net
(the generator checks a flip_y parameter, default True).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0


class HeatmapGenerator:
    """Generates heatmaps with a correctly-oriented football pitch background."""

    def __init__(self):
        self.pitch_length    = PITCH_LENGTH
        self.pitch_width     = PITCH_WIDTH
        self.pitch_color     = '#3CB371'
        self.line_color      = 'white'
        self.background_color= '#2E8B57'

        colors = ['#000033', '#87CEEB', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000']
        self.cmap = LinearSegmentedColormap.from_list('custom_heat', colors, N=100)

    def generate_heatmap(self, positions, player_id=None,
                         position_name="Unknown", bins=60,
                         flip_y: bool = True):
        """
        Generate heatmap with football pitch background.

        Args:
            positions:     np.ndarray shape (N, 2), columns = [x_meters, y_meters].
                           x=0 is the defensive goal-line, x=105 attacking.
                           y=0 is the top touchline (as returned by ViewTransformer).
            player_id:     shown in title
            position_name: shown in title
            bins:          histogram resolution
            flip_y:        if True (default), flip Y so top touchline appears at
                           the top of the image.  Set False only if you have
                           already flipped externally.

        Returns:
            matplotlib.figure.Figure
        """
        if positions is None or len(positions) == 0:
            return None

        pos = np.array(positions, dtype=float)

        # ── Mirror fix ────────────────────────────────────────────────────────
        # ViewTransformer maps pixel Y linearly: Y=0 → top of pitch.
        # imshow(origin='lower') puts Y=0 at the BOTTOM, then invert_yaxis()
        # flips it back — but that would make Y=0 appear at the top again only
        # if the data were already flipped.  We flip here so that after the
        # imshow + invert_yaxis chain the defensive end stays on the left.
        if flip_y:
            pos = pos.copy()
            pos[:, 1] = self.pitch_width - pos[:, 1]

        # Clip to pitch boundaries
        pos[:, 0] = np.clip(pos[:, 0], 0, self.pitch_length)
        pos[:, 1] = np.clip(pos[:, 1], 0, self.pitch_width)

        fig, ax = plt.subplots(figsize=(12, 8))
        self._draw_pitch_background(ax)

        heatmap, xedges, yedges = np.histogram2d(
            pos[:, 0], pos[:, 1],
            bins=bins,
            range=[[0, self.pitch_length], [0, self.pitch_width]],
        )
        try:
            heatmap = gaussian_filter(heatmap, sigma=2)
        except Exception:
            pass

        extent = [0, self.pitch_length, 0, self.pitch_width]
        im = ax.imshow(
            heatmap.T, extent=extent, origin='lower',
            cmap=self.cmap, aspect='auto', alpha=0.7,
            interpolation='bilinear',
        )

        title = f"Player #{player_id} — {position_name} Movement Heatmap" \
                if player_id is not None else "Movement Heatmap"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Field Length (meters)')
        ax.set_ylabel('Field Width (meters)')
        plt.colorbar(im, ax=ax, label='Activity Density', fraction=0.046, pad=0.04)

        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.invert_yaxis()   # now top touchline is at the top of the figure ✓

        plt.tight_layout()
        return fig

    # ── Pitch drawing ─────────────────────────────────────────────────────────
    def _draw_pitch_background(self, ax):
        ax.add_patch(Rectangle((0, 0), self.pitch_length, self.pitch_width,
                               lw=2, edgecolor=self.line_color,
                               facecolor=self.pitch_color, alpha=0.7))
        ax.axvline(x=self.pitch_length / 2, color=self.line_color, lw=2)
        ax.add_patch(Circle((self.pitch_length/2, self.pitch_width/2), 9.15,
                            color=self.line_color, fill=False, lw=2))
        ax.plot(self.pitch_length/2, self.pitch_width/2, 'wo', markersize=6)

        pw, pl = self.pitch_width, self.pitch_length
        # Penalty areas
        for x0 in [0, pl - 16.5]:
            ax.add_patch(Rectangle((x0, (pw-40.3)/2), 16.5, 40.3,
                                   lw=2, edgecolor=self.line_color, facecolor='none'))
        # Goal areas
        for x0 in [0, pl - 5.5]:
            ax.add_patch(Rectangle((x0, (pw-18.3)/2), 5.5, 18.3,
                                   lw=2, edgecolor=self.line_color, facecolor='none'))
        # Penalty spots
        ax.plot(11, pw/2, 'wo', markersize=4)
        ax.plot(pl - 11, pw/2, 'wo', markersize=4)
        # Corner arcs
        r = 1
        for (ox, oy), t1, t2 in [
            ((0,  pw), 270, 360),
            ((pl, pw), 180, 270),
            ((0,  0 ),   0,  90),
            ((pl,  0),  90, 180),
        ]:
            ax.add_patch(Arc((ox, oy), r*2, r*2, angle=0,
                             theta1=t1, theta2=t2,
                             color=self.line_color, lw=2))

        ax.annotate('← Defensive', xy=(8, pw + 2), fontsize=10,
                    color='white', fontweight='bold')
        ax.annotate('Attacking →', xy=(pl - 40, pw + 2), fontsize=10,
                    color='white', fontweight='bold')
        ax.set_facecolor(self.background_color)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    def generate_comparison_heatmap(self, positions_list,
                                    player_ids=None, position_names=None,
                                    flip_y: bool = True):
        if not positions_list:
            return None
        n = len(positions_list)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 8))
        if n == 1:
            axes = [axes]

        for i, (positions, ax) in enumerate(zip(positions_list, axes)):
            pid   = player_ids[i]   if player_ids   and i < len(player_ids)   else None
            pname = position_names[i] if position_names and i < len(position_names) else f"Player {i+1}"

            self._draw_pitch_background(ax)

            if positions is not None and len(positions) > 0:
                pos = np.array(positions, dtype=float)
                if flip_y:
                    pos = pos.copy()
                    pos[:, 1] = self.pitch_width - pos[:, 1]
                pos[:, 0] = np.clip(pos[:, 0], 0, self.pitch_length)
                pos[:, 1] = np.clip(pos[:, 1], 0, self.pitch_width)

                heatmap, _, _ = np.histogram2d(
                    pos[:, 0], pos[:, 1], bins=40,
                    range=[[0, self.pitch_length], [0, self.pitch_width]],
                )
                try:
                    heatmap = gaussian_filter(heatmap, sigma=2)
                except Exception:
                    pass
                ax.imshow(heatmap.T,
                          extent=[0, self.pitch_length, 0, self.pitch_width],
                          origin='lower', cmap=self.cmap,
                          aspect='auto', alpha=0.7, interpolation='bilinear')

            ax.set_title(pname, fontsize=12, fontweight='bold')
            ax.set_xlim(0, self.pitch_length)
            ax.set_ylim(0, self.pitch_width)
            ax.invert_yaxis()

        plt.tight_layout()
        return fig