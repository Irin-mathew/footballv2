# player_filter.py - Filter to keep only 22 players
import numpy as np
from collections import defaultdict

class PlayerFilter:
    """
    Filters tracked players to keep only the best 22 (+ 2 goalkeepers)
    Based on tracking quality and consistency
    """
    
    def __init__(self, max_players=22):
        self.max_players = max_players
        # Changed to dict instead of defaultdict(lambda) for pickle compatibility
        self.player_quality_scores = {}
    
    def _get_default_score(self):
        """Return default score dict"""
        return {
            'frame_count': 0,
            'avg_confidence': 0.0,
            'total_confidence': 0.0,
            'last_seen': 0,
            'is_goalkeeper': False
        }
    
    def update_quality_scores(self, tracker_ids, confidences, class_ids, frame_idx):
        """Update quality scores for tracked players"""
        for i, tid in enumerate(tracker_ids):
            if tid not in self.player_quality_scores:
                self.player_quality_scores[tid] = self._get_default_score()
            
            score = self.player_quality_scores[tid]
            
            score['frame_count'] += 1
            conf = confidences[i] if confidences is not None else 0.5
            score['total_confidence'] += conf
            score['avg_confidence'] = score['total_confidence'] / score['frame_count']
            score['last_seen'] = frame_idx
            
            # Mark goalkeepers
            if class_ids is not None and i < len(class_ids) and class_ids[i] == 1:
                score['is_goalkeeper'] = True
    
    def get_quality_score(self, player_id):
        """Calculate overall quality score for a player"""
        score = self.player_quality_scores[player_id]
        
        # Quality = frame_count * avg_confidence
        # Goalkeepers get slight boost
        quality = score['frame_count'] * score['avg_confidence']
        
        if score['is_goalkeeper']:
            quality *= 1.1  # 10% boost for goalkeepers
        
        return quality
    
    def get_filtered_player_ids(self):
        """
        Get the best N player IDs to keep
        
        Returns: Set of player IDs to keep
        """
        if len(self.player_quality_scores) <= self.max_players:
            return set(self.player_quality_scores.keys())
        
        # Score all players
        scored_players = [
            (pid, self.get_quality_score(pid)) 
            for pid in self.player_quality_scores.keys()
        ]
        
        # Sort by quality score (descending)
        scored_players.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top N players
        keep_ids = {pid for pid, _ in scored_players[:self.max_players]}
        
        # Always include goalkeepers regardless of quality
        for pid, score_data in self.player_quality_scores.items():
            if score_data['is_goalkeeper'] and len(keep_ids) < self.max_players + 2:
                keep_ids.add(pid)
        
        return keep_ids
    
    def filter_detections(self, detections, tracker_ids):
        """
        Filter detections to keep only valid players
        
        Args:
            detections: Supervision Detections object
            tracker_ids: Array of tracker IDs
        
        Returns: Filtered detections and tracker IDs
        """
        if tracker_ids is None or len(tracker_ids) == 0:
            return detections, tracker_ids
        
        # Safety check: ensure tracker_ids matches detections length
        if len(tracker_ids) != len(detections):
            # If lengths don't match, return unfiltered
            # This avoids boolean indexing dimension mismatch
            return detections, tracker_ids
        
        keep_ids = self.get_filtered_player_ids()
        
        # Create mask for players to keep
        mask = np.array([tid in keep_ids for tid in tracker_ids], dtype=bool)
        
        if not np.any(mask):
            return detections, tracker_ids
        
        # Filter detections
        filtered_detections = detections[mask]
        filtered_tracker_ids = tracker_ids[mask]
        
        return filtered_detections, filtered_tracker_ids
    
    def get_statistics(self):
        """Get filtering statistics"""
        total = len(self.player_quality_scores)
        kept = len(self.get_filtered_player_ids())
        
        goalkeepers = sum(
            1 for score in self.player_quality_scores.values() 
            if score['is_goalkeeper']
        )
        
        return {
            'total_detected': total,
            'kept': kept,
            'filtered_out': total - kept,
            'goalkeepers': goalkeepers
        }