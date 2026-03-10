"""
Advanced Visualization Engine for Football Analysis
Handles Voronoi diagrams, eclipse annotations, heatmaps, and pitch analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter
import io
import base64
from pathlib import Path


class VisualizationEngine:
    """Generates advanced visualizations for football analytics"""
    
    def __init__(self, pitch_width=105, pitch_height=68):
        """Initialize with pitch dimensions"""
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.fig_dpi = 150
        
    def generate_heatmap(self, positions, player_id=None, width=1920, height=1080):
        """
        Generate player movement heatmap
        
        Args:
            positions: List of (x, y) coordinates in pixels
            player_id: Player ID for labeling
            width: Frame width
            height: Frame height
            
        Returns:
            Base64 encoded PNG image
        """
        if len(positions) < 10:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect('equal')
        
        # Create histogram
        positions = np.array(positions)
        h, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=[int(width/50), int(height/50)],
            range=[[0, width], [0, height]]
        )
        
        # Apply Gaussian filter for smoothing
        h = gaussian_filter(h.T, sigma=2)
        
        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        im = ax.imshow(h, extent=extent, cmap='hot', alpha=0.8, origin='upper')
        
        ax.set_title(f'Player #{player_id} Movement Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Activity Intensity')
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_voronoi_diagram(self, team1_positions, team2_positions, 
                                  team1_color=(0, 0, 255), team2_color=(255, 0, 0),
                                  pitch_width=None, pitch_height=None, 
                                  with_eclipse=True):
        """
        Generate Voronoi pitch control diagram with team differentiation
        
        Args:
            team1_positions: List of (x, y) for team 1
            team2_positions: List of (x, y) for team 2
            team1_color: BGR color for team 1
            team2_color: BGR color for team 2
            with_eclipse: Include eclipse annotations for players
            
        Returns:
            Base64 encoded PNG image
        """
        if pitch_width is None:
            pitch_width = self.pitch_width
        if pitch_height is None:
            pitch_height = self.pitch_height
            
        # Create blank pitch
        img = self._create_pitch(pitch_width, pitch_height, scale=10)
        
        all_positions = np.array(list(team1_positions) + list(team2_positions))
        
        if len(all_positions) < 3:
            return self._img_to_base64(img)
        
        # Calculate Voronoi
        try:
            vor = Voronoi(all_positions)
            
            # Draw Voronoi cells with team colors
            for point_idx, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                
                if -1 not in region and len(region) > 0:
                    region_points = vor.vertices[region]
                    region_points = np.int32(region_points * 10)
                    
                    # Determine team
                    is_team1 = point_idx < len(team1_positions)
                    color = team1_color if is_team1 else team2_color
                    
                    cv2.fillPoly(img, [region_points], color)
                    cv2.polylines(img, [region_points], True, (200, 200, 200), 1)
        except Exception as e:
            print(f"⚠️ Voronoi generation error: {e}")
        
        # Add eclipse annotations
        if with_eclipse:
            for i, pos in enumerate(team1_positions):
                x, y = int(pos[0] * 10), int(pos[1] * 10)
                cv2.ellipse(img, (x, y), (20, 30), 0, 0, 360, (0, 0, 255), 2)
                cv2.putText(img, str(i+1), (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (0, 0, 255), 1)
            
            for i, pos in enumerate(team2_positions):
                x, y = int(pos[0] * 10), int(pos[1] * 10)
                cv2.ellipse(img, (x, y), (20, 30), 0, 0, 360, (255, 0, 0), 2)
                cv2.putText(img, str(i+1), (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (255, 0, 0), 1)
        
        # Convert BGR to RGB and create matplotlib figure
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
        ax.imshow(img_rgb)
        ax.set_title('Voronoi Pitch Control Analysis with Eclipse Annotations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0/255, 0/255, 255/255), label='Team 1'),
            Patch(facecolor=(255/255, 0/255, 0/255), label='Team 2')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_eclipse_annotation(self, frame, detections, player_ids, 
                                     team1_color=(100, 149, 237), 
                                     team2_color=(255, 69, 0),
                                     thickness=3, radius=30):
        """
        Generate frame with eclipse annotations for all players
        
        Args:
            frame: Video frame (BGR)
            detections: Player detections with bboxes
            player_ids: Player ID for each detection
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for i, (bbox, player_id) in enumerate(zip(detections, player_ids)):
            x1, y1, x2, y2 = bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Determine color based on team (simplified for now)
            color = team1_color if player_id % 2 == 0 else team2_color
            
            # Draw eclipse
            cv2.ellipse(annotated, (cx, cy), (radius, radius + 10), 0, 0, 360, color, thickness)
            
            # Add player ID
            cv2.putText(annotated, f"#{player_id}", (cx - 15, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated
    
    def generate_pitch_control_heatmap(self, team1_positions, team2_positions,
                                       pitch_width=None, pitch_height=None):
        """
        Generate pitch control intensity heatmap for each team
        
        Args:
            team1_positions: Player positions for team 1
            team2_positions: Player positions for team 2
            
        Returns:
            Base64 encoded combined heatmap
        """
        if pitch_width is None:
            pitch_width = self.pitch_width
        if pitch_height is None:
            pitch_height = self.pitch_height
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=100)
        
        # Create grids for heatmaps
        grid_x = np.linspace(0, pitch_width, 30)
        grid_y = np.linspace(0, pitch_height, 20)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        # Team 1 heatmap
        if len(team1_positions) > 0:
            team1_pos = np.array(team1_positions)
            Z1 = np.zeros_like(X, dtype=float)
            for i, j in np.ndindex(X.shape):
                distances = np.sqrt((X[i, j] - team1_pos[:, 0])**2 + 
                                   (Y[i, j] - team1_pos[:, 1])**2)
                Z1[i, j] = np.exp(-distances.min() / 5)
            
            im1 = axes[0].contourf(X, Y, Z1, cmap='Blues', levels=15)
            axes[0].scatter(team1_pos[:, 0], team1_pos[:, 1], c='darkblue', s=100, marker='o')
            axes[0].set_title('Team 1 Pitch Control', fontweight='bold')
            axes[0].set_xlim(0, pitch_width)
            axes[0].set_ylim(0, pitch_height)
            plt.colorbar(im1, ax=axes[0])
        
        # Team 2 heatmap
        if len(team2_positions) > 0:
            team2_pos = np.array(team2_positions)
            Z2 = np.zeros_like(X, dtype=float)
            for i, j in np.ndindex(X.shape):
                distances = np.sqrt((X[i, j] - team2_pos[:, 0])**2 + 
                                   (Y[i, j] - team2_pos[:, 1])**2)
                Z2[i, j] = np.exp(-distances.min() / 5)
            
            im2 = axes[1].contourf(X, Y, Z2, cmap='Reds', levels=15)
            axes[1].scatter(team2_pos[:, 0], team2_pos[:, 1], c='darkred', s=100, marker='o')
            axes[1].set_title('Team 2 Pitch Control', fontweight='bold')
            axes[1].set_xlim(0, pitch_width)
            axes[1].set_ylim(0, pitch_height)
            plt.colorbar(im2, ax=axes[1])
        
        # Combined view
        if len(team1_positions) > 0 and len(team2_positions) > 0:
            Z_combined = Z1 - Z2
            im3 = axes[2].contourf(X, Y, Z_combined, cmap='RdBu', levels=15)
            axes[2].scatter(team1_pos[:, 0], team1_pos[:, 1], c='blue', s=100, marker='o', label='Team 1')
            axes[2].scatter(team2_pos[:, 0], team2_pos[:, 1], c='red', s=100, marker='s', label='Team 2')
            axes[2].set_title('Pitch Control Difference', fontweight='bold')
            axes[2].set_xlim(0, pitch_width)
            axes[2].set_ylim(0, pitch_height)
            axes[2].legend()
            plt.colorbar(im3, ax=axes[2], label='Team 1 ← → Team 2')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_pitch(self, width, height, scale=1, color=(0, 100, 0)):
        """Create a blank pitch image"""
        img = np.full((int(height*scale), int(width*scale), 3), color, dtype=np.uint8)
        
        # Draw field lines
        h, w = img.shape[:2]
        
        # Center line
        cv2.line(img, (w//2, 0), (w//2, h), (255, 255, 255), 2*scale)
        
        # Goal lines
        cv2.line(img, (0, 0), (0, h), (255, 255, 255), 2*scale)
        cv2.line(img, (w, 0), (w, h), (255, 255, 255), 2*scale)
        
        # Touchlines
        cv2.line(img, (0, 0), (w, 0), (255, 255, 255), 2*scale)
        cv2.line(img, (0, h), (w, h), (255, 255, 255), 2*scale)
        
        # Center circle
        cv2.circle(img, (w//2, h//2), int(50*scale), (255, 255, 255), 2*scale)
        cv2.circle(img, (w//2, h//2), int(2*scale), (255, 255, 255), -1)
        
        # Goal areas (simplified)
        goal_width = int(40.3 * scale)
        goal_height = int(18.3 * scale)
        
        # Left goal
        cv2.rectangle(img, (0, (h-goal_height)//2), 
                     (goal_width, (h+goal_height)//2), (255, 255, 255), 2*scale)
        
        # Right goal
        cv2.rectangle(img, (w-goal_width, (h-goal_height)//2), 
                     (w, (h+goal_height)//2), (255, 255, 255), 2*scale)
        
        return img
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.fig_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _img_to_base64(self, img):
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
