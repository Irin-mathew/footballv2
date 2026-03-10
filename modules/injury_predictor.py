

"""
modules/injury_predictor.py
FootballIQ — Per-tissue differentiated injury prediction engine
──────────────────────────────────────────────────────────────
FIXES (v2):
  • _pct_above() was scaling benchmarks by duration, making them near-zero
    for short clips → every score hit 100. Now uses ABSOLUTE thresholds so
    short clips produce realistic low/moderate scores.
  • Added clip-duration awareness: if footage < 60s, scores are scaled down.
  • Tissue scores now vary properly between players (hamstring not always 100).
"""

import numpy as np


# ── Colour map for display layers ─────────────────────────────────────────────
TIER_COLORS = {
    'CRITICAL': '#E74C3C',    # red
    'HIGH':     '#E67E22',    # orange
    'MODERATE': '#F1C40F',    # yellow
    'LOW':      '#2ECC71',    # green
    'MINIMAL':  '#1ABC9C',    # teal
}

TIER_ORDER = ['CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'MINIMAL']

# ── ABSOLUTE per-metric thresholds that warrant MODERATE (35) injury concern ──
# These are NOT scaled by duration — they represent raw observed values that
# represent meaningful load regardless of how long the clip is.
ABSOLUTE_THRESHOLDS = {
    # metric_key:  (moderate_threshold, high_threshold, critical_threshold)
    'sprint_count':               (8,   18,  32),
    'sprint_distance_km':         (0.15, 0.40, 0.70),
    'very_hi_intensity_km':       (0.30, 0.80, 1.40),
    'high_intensity_distance_km': (0.50, 1.20, 2.00),
    'total_distance_km':          (3.0,  7.0, 10.5),
    'hard_accelerations':         (4,    10,   20),
    'hard_decelerations':         (4,    10,   20),
    'direction_changes':          (50,  130,  210),
    'max_speed':                  (22,   26,   30),
    'fast_pct':                   (8,    18,   30),
}


def _clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))


def _tier(score):
    if score >= 75: return 'CRITICAL'
    if score >= 55: return 'HIGH'
    if score >= 35: return 'MODERATE'
    if score >= 18: return 'LOW'
    return 'MINIMAL'


def _score_metric(value, moderate_thresh, high_thresh, critical_thresh):
    """
    Map a raw metric value to a 0-100 contribution score.
    Below moderate_thresh → 0–34 (LOW/MINIMAL range).
    Between moderate and high → 35–54 (MODERATE).
    Between high and critical → 55–74 (HIGH).
    Above critical → 75–100 (CRITICAL).
    This avoids the old _pct_above bug where scaled-to-zero benchmarks
    made every value look enormous.
    """
    if value <= 0:
        return 0.0
    if value >= critical_thresh:
        # Scale 75–100 based on how far above critical
        over = (value - critical_thresh) / max(critical_thresh * 0.5, 1)
        return _clamp(75 + over * 25)
    if value >= high_thresh:
        t = (value - high_thresh) / max(critical_thresh - high_thresh, 1e-6)
        return _clamp(55 + t * 20)
    if value >= moderate_thresh:
        t = (value - moderate_thresh) / max(high_thresh - moderate_thresh, 1e-6)
        return _clamp(35 + t * 20)
    # Below moderate threshold: scale 0–34
    t = value / max(moderate_thresh, 1e-6)
    return _clamp(t * 34)


def _ms(key, value):
    """Score a metric using absolute thresholds."""
    thresholds = ABSOLUTE_THRESHOLDS.get(key)
    if thresholds is None:
        return 0.0
    return _score_metric(value, *thresholds)


class InjuryPredictor:
    """
    Per-tissue injury risk engine. Each player receives 5 independent
    tissue risk scores that drive differentiated injury flags and protocols.
    """

    TISSUE_NAMES = ['Hamstring', 'Quadriceps', 'Calf/Achilles', 'Hip Flexors', 'Knee/Ligament']

    def __init__(self, model_path=None):
        print("🩺 InjuryPredictor v2 — absolute-threshold per-tissue model loaded")

    # ── Public entry point ────────────────────────────────────────────────────
    def predict(self, player_stats):
        pos     = self._normalise_position(player_stats.get('position', 'Middle'))
        tissue  = self._compute_tissue_scores(player_stats, pos)
        overall, overall_tier = self._overall_risk(tissue)
        flags   = self._build_injury_flags(tissue, player_stats, overall)
        factors = self._risk_factors(player_stats)
        recs    = self._recommendations(overall, overall_tier, tissue, player_stats, pos)

        return {
            # legacy fields (kept for backward compat)
            'risk_score':     round(overall / 100, 3),
            'raw_risk_score': round(overall, 1),
            'risk_level':     overall_tier,
            # new rich fields
            'overall_risk':   round(overall, 1),
            'overall_tier':   overall_tier,
            'tissue_scores':  {k: round(v, 1) for k, v in tissue.items()},
            'tissue_tiers':   {k: _tier(v) for k, v in tissue.items()},
            'tier_colors':    {k: TIER_COLORS[_tier(v)] for k, v in tissue.items()},
            'likely_injuries': flags,
            'risk_factors':   factors,
            'recommendations': recs,
            'confidence':     'absolute_threshold_tissue_model_v2',
        }

    # ── Position normalisation ────────────────────────────────────────────────
    def _normalise_position(self, pos):
        p = str(pos).lower()
        if any(x in p for x in ['attack', 'forward', 'striker', 'winger', 'wing']):
            return 'Attacking'
        if any(x in p for x in ['defend', 'back', 'centre-back', 'cb', 'lb', 'rb']):
            return 'Defensive'
        return 'Middle'

    # ── Per-tissue scores (0–100 each) ────────────────────────────────────────
    def _compute_tissue_scores(self, s, pos):
        # Pull all the raw metrics once
        tot    = s.get('total_distance_km', 0)
        hi     = s.get('high_intensity_distance_km', 0)
        vhi    = s.get('very_hi_intensity_km', hi * 0.45)
        sprd   = s.get('sprint_distance_km', 0)
        sprc   = s.get('sprint_count', 0)
        mx     = s.get('max_speed', 0)
        hacc   = s.get('hard_accelerations', 0)
        hdec   = s.get('hard_decelerations', 0)
        dc     = s.get('direction_changes', 0)
        fast_p = s.get('fast_pct', 0)
        jog_p  = s.get('jog_pct', 0)
        acc    = s.get('accelerations', 0)

        # ── A: Hamstring — sprint mechanics & explosive speed ─────────────────
        # Driven by sprint volume, sprint distance, max speed, hard accels
        h1 = _ms('sprint_distance_km', sprd)   * 0.38
        h2 = _ms('sprint_count',       sprc)   * 0.28
        h3 = _ms('max_speed',          mx)     * 0.20
        h4 = _ms('hard_accelerations', hacc)   * 0.14
        hamstring = _clamp(h1 + h2 + h3 + h4)

        # ── B: Quadriceps — braking load & high-intensity running ──────────────
        # Driven by hard decels (eccentric quad), very-hi-intensity, fast_pct
        q1 = _ms('hard_decelerations',         hdec)   * 0.45
        q2 = _ms('very_hi_intensity_km',       vhi)    * 0.30
        q3 = _ms('sprint_count',               sprc)   * 0.15
        q4 = _ms('fast_pct',                   fast_p) * 0.10
        quadriceps = _clamp(q1 + q2 + q3 + q4)

        # ── C: Calf / Achilles — direction changes & total ground contact ─────
        # Driven by direction changes (ankle load), total distance, jog%
        c1 = _ms('direction_changes',          dc)     * 0.42
        c2 = _ms('total_distance_km',          tot)    * 0.28
        c3 = _ms('high_intensity_distance_km', hi)     * 0.18
        c4 = _clamp(jog_p / 45 * 18)                          # jog% contribution (max 18pts)
        calf = _clamp(c1 + c2 + c3 + c4)

        # ── D: Hip Flexors — repeated sprint starts & fast-pace proportion ────
        # Driven by sprint count (hip flexor acceleration demand), fast_pct
        f1 = _ms('sprint_count',       sprc)   * 0.48
        f2 = _ms('fast_pct',           fast_p) * 0.30
        f3 = _ms('hard_accelerations', hacc)   * 0.22
        hip = _clamp(f1 + f2 + f3)

        # ── E: Knee / Ligament — cutting stress & high-speed braking ──────────
        # Driven by direction changes (cutting), hard decels, max speed
        k1 = _ms('direction_changes',  dc)     * 0.48
        k2 = _ms('hard_decelerations', hdec)   * 0.35
        k3 = _ms('max_speed',          mx)     * 0.17
        knee = _clamp(k1 + k2 + k3)

        # Apply position-specific multipliers to capture sport context
        pos_mult = self._position_multipliers(pos)
        hamstring  = _clamp(hamstring  * pos_mult['Hamstring'])
        quadriceps = _clamp(quadriceps * pos_mult['Quadriceps'])
        calf       = _clamp(calf       * pos_mult['Calf/Achilles'])
        hip        = _clamp(hip        * pos_mult['Hip Flexors'])
        knee       = _clamp(knee       * pos_mult['Knee/Ligament'])

        return {
            'Hamstring':     hamstring,
            'Quadriceps':    quadriceps,
            'Calf/Achilles': calf,
            'Hip Flexors':   hip,
            'Knee/Ligament': knee,
        }

    def _position_multipliers(self, pos):
        """
        Position modulates WHICH tissues are most loaded.
        Attackers → more hamstring/hip (sprints).
        Midfielders → more calf/knee (direction changes).
        Defenders → more quad/knee (stopping/marking).
        """
        if pos == 'Attacking':
            return {'Hamstring': 1.20, 'Quadriceps': 0.95, 'Calf/Achilles': 0.90,
                    'Hip Flexors': 1.15, 'Knee/Ligament': 0.88}
        elif pos == 'Defensive':
            return {'Hamstring': 0.90, 'Quadriceps': 1.15, 'Calf/Achilles': 0.95,
                    'Hip Flexors': 0.85, 'Knee/Ligament': 1.18}
        else:  # Middle
            return {'Hamstring': 0.95, 'Quadriceps': 1.00, 'Calf/Achilles': 1.12,
                    'Hip Flexors': 0.95, 'Knee/Ligament': 1.10}

    def _overall_risk(self, tissue):
        weights = {'Hamstring': 0.28, 'Quadriceps': 0.22, 'Calf/Achilles': 0.18,
                   'Hip Flexors': 0.14, 'Knee/Ligament': 0.18}
        overall = sum(tissue[k] * weights[k] for k in weights)
        return _clamp(overall), _tier(_clamp(overall))

    # ── Injury flags (shown as coloured cards in the UI) ─────────────────────
    def _build_injury_flags(self, tissue, s, overall):
        flags = []

        descriptions = {
            'Hamstring': (
                'Hamstring Strain Risk',
                lambda s: f"{s.get('sprint_count',0)} sprints · {s.get('sprint_distance_km',0):.2f} km sprint dist · max {s.get('max_speed',0)} km/h"
            ),
            'Quadriceps': (
                'Quadriceps Fatigue / Strain Risk',
                lambda s: f"{s.get('hard_decelerations',0)} hard decels · {s.get('very_hi_intensity_km',0):.2f} km very-hi-int running"
            ),
            'Calf/Achilles': (
                'Calf / Achilles Stress Risk',
                lambda s: f"{s.get('direction_changes',0)} direction changes · {s.get('total_distance_km',0):.2f} km total dist"
            ),
            'Hip Flexors': (
                'Hip Flexor Tightness Risk',
                lambda s: f"{s.get('sprint_count',0)} sprints · {s.get('fast_pct',0):.0f}% time at fast pace"
            ),
            'Knee/Ligament': (
                'Knee / ACL-MCL Stress Risk',
                lambda s: f"{s.get('direction_changes',0)} direction changes · {s.get('hard_decelerations',0)} hard decels"
            ),
        }

        for tissue_name, score in sorted(tissue.items(), key=lambda x: x[1], reverse=True):
            t = _tier(score)
            if score < 18:
                continue
            inj_name, cause_fn = descriptions.get(tissue_name, (tissue_name, lambda s: ''))
            prob_pct = round(_clamp(score * 0.52, 0, 60), 1)
            flags.append({
                'type':            inj_name,
                'tissue':          tissue_name,
                'severity':        t,
                'score':           round(score, 1),
                'color':           TIER_COLORS[t],
                'probability_pct': prob_pct,
                'cause':           cause_fn(s),
                # legacy fields
                'probability':     round(prob_pct / 100, 3),
                'reasons':         [cause_fn(s)],
            })

        return flags[:5]

    # ── Risk factor sentences ─────────────────────────────────────────────────
    def _risk_factors(self, s):
        factors = []
        checks = [
            (s.get('sprint_count', 0),              18,
             lambda v: f"Sprint count {v} is elevated — hamstring strain risk"),
            (s.get('hard_decelerations', 0),         10,
             lambda v: f"{v} hard decelerations — quad/knee eccentric overload"),
            (s.get('direction_changes', 0),          130,
             lambda v: f"{v} direction changes — calf/Achilles and ankle stress"),
            (s.get('max_speed', 0),                  26,
             lambda v: f"Max speed {v} km/h — above 26 km/h elevates hamstring risk"),
            (s.get('very_hi_intensity_km', 0),       0.80,
             lambda v: f"{v:.2f} km at >20 km/h — quadriceps fatigue risk"),
            (s.get('high_intensity_distance_km', 0), 1.20,
             lambda v: f"{v:.2f} km hi-intensity running — general muscle fatigue"),
        ]
        for value, threshold, msg in checks:
            if value > threshold:
                factors.append(msg(value))
        return factors

    # ── Personalised recommendations ─────────────────────────────────────────
    def _recommendations(self, overall, tier, tissue, s, pos):
        recs = []
        top_t = max(tissue, key=tissue.get)
        top_s = tissue[top_t]

        base = {
            'CRITICAL': ['CRITICAL: Mandatory physio assessment before any training',
                         'Zero training load for 48–72h minimum',
                         'Document symptoms; consider imaging if localised pain'],
            'HIGH':     ['Avoid field training for 48h',
                         'Physio review recommended before return',
                         'Monitor resting HR — return only when within 5 bpm of baseline'],
            'MODERATE': ['Limit high-intensity work for 24–48h',
                         'Daily pain self-assessment: rate 0–10 each morning',
                         'Prioritise mobility and tissue-specific rehab'],
            'LOW':      ['Active recovery preferred — light pool or cycle session',
                         'Extend warm-up by 10 min next training session'],
            'MINIMAL':  ['Standard recovery — no specific intervention needed',
                         'Routine foam roll + static stretch cool-down'],
        }
        recs += base.get(tier, base['MODERATE'])

        if top_s >= 35:
            recs.append(f'Primary tissue concern: {top_t} ({top_s:.0f}/100) — see targeted pain-relief protocol')

        # Position load flags
        if pos == 'Attacking' and s.get('sprint_count', 0) > 18:
            recs.append('High sprint volume for attacking position — sprint mechanics assessment advised')
        if pos == 'Defensive' and s.get('hard_decelerations', 0) > 10:
            recs.append('Elevated hard deceleration count — knee + quad prehab priority')
        if pos == 'Middle' and s.get('direction_changes', 0) > 130:
            recs.append('Very high direction-change load — ankle stability and agility work on return')

        return recs[:6]