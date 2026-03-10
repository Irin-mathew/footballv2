
"""
modules/recovery_planner.py
FootballIQ — Sophisticated per-player recovery planner
───────────────────────────────────────────────────────
Consumes the per-tissue InjuryPredictor output and generates:
  • Position-adjusted fatigue score (players aren't equally expected to run the same)
  • Per-tissue pain-relief protocols (hamstring ≠ knee ≠ calf)
  • Day-by-day return-to-train schedule (1–8 days depending on tier)
  • Differentiated nutrition & supplement note
  • Targeted pain relief (immediate 0–6h + next 48h + medication)
"""

# ── Tier colour palette (mirrors injury_predictor) ────────────────────────────
TIER_COLORS = {
    'CRITICAL': '#E74C3C',
    'HIGH':     '#E67E22',
    'MODERATE': '#F1C40F',
    'LOW':      '#2ECC71',
    'MINIMAL':  '#1ABC9C',
}


class RecoveryPlanner:

    def __init__(self):
        print("💊 RecoveryPlanner — per-tissue differentiated model loaded")

    # ── Main entry point ──────────────────────────────────────────────────────
    def generate_recovery_plan(self, player_stats, injury_prediction):

        # Pull tissue scores from the new predictor output (or fall back)
        tissue = injury_prediction.get('tissue_scores', {})
        tier   = injury_prediction.get('overall_tier',
                 injury_prediction.get('risk_level', 'LOW'))
        overall = injury_prediction.get('overall_risk',
                  injury_prediction.get('raw_risk_score', 0))
        flags   = injury_prediction.get('likely_injuries', [])

        pos = self._normalise_position(
            player_stats.get('position', 'Middle'))

        fatigue_score = self._fatigue_score(player_stats, pos)
        fatigue_level, fatigue_color = self._fatigue_label(fatigue_score)

        rest_days = self._rest_days(tier, fatigue_score)

        plan = {
            # ── identification ───────────────────────────────────────────
            'player_id':          player_stats.get('player_id', 'Unknown'),
            'position':           pos,
            'match_date':         player_stats.get('match_date', 'Recent Match'),

            # ── risk summary ─────────────────────────────────────────────
            'overall_risk':       round(overall, 1),
            'risk_tier':          tier,
            'risk_color':         TIER_COLORS.get(tier, '#888'),
            'next_match_ready':   f'{rest_days + 1}–{rest_days + 2} days',

            # ── fatigue (position-adjusted) ──────────────────────────────
            'fatigue_level':      fatigue_level,
            'fatigue_color':      fatigue_color,
            'fatigue_score':      int(fatigue_score),
            'fatigue_max':        500,

            # ── tissue scores pass-through ───────────────────────────────
            'tissue_scores':      tissue,
            'tissue_tiers':       {k: self._tissue_tier(v) for k, v in tissue.items()},
            'tissue_colors':      {k: TIER_COLORS[self._tissue_tier(v)] for k, v in tissue.items()},

            # ── injury flags ─────────────────────────────────────────────
            'injury_flags':       flags,

            # ── rest / workload ──────────────────────────────────────────
            'rest_days':          rest_days,
            'key_metrics':        self._key_metrics(player_stats),
            'workload':           self._workload_dict(player_stats),

            # ── detailed protocols ───────────────────────────────────────
            'pain_relief':        self._pain_relief(tissue, tier),
            'schedule':           self._schedule(tier, pos),
            'position_insights':  self._position_insights(pos, player_stats),
            'recovery_prescription': self._prescription(fatigue_score, tier, pos, player_stats),
            'warnings':           self._warnings(tier, overall, flags, fatigue_score, player_stats),
            'timeline':           self._timeline(fatigue_score, tier),  # legacy text list

            # ── nutrition ────────────────────────────────────────────────
            'nutrition_note':     self._nutrition(fatigue_score, tier, player_stats),
            'diet_suggestions':   self._diet(fatigue_score, player_stats),
            'recommendations':    injury_prediction.get('recommendations', []),
        }
        return plan

    # ── Position helper ───────────────────────────────────────────────────────
    def _normalise_position(self, pos):
        p = str(pos).lower()
        if any(x in p for x in ['attack', 'forward', 'striker', 'winger']):
            return 'Attacking'
        if any(x in p for x in ['defend', 'back', 'goalkeeper']):
            return 'Defensive'
        return 'Middle'

    def _tissue_tier(self, score):
        if score >= 75: return 'CRITICAL'
        if score >= 55: return 'HIGH'
        if score >= 35: return 'MODERATE'
        if score >= 18: return 'LOW'
        return 'MINIMAL'

    # ── Position-adjusted fatigue score (0–500) ───────────────────────────────
    def _fatigue_score(self, s, pos):
        # Midfielders are expected to run more — normalise fatigue against that
        pos_multiplier = {'Attacking': 1.15, 'Middle': 1.0, 'Defensive': 1.10}
        m = pos_multiplier.get(pos, 1.0)

        score = (
            s.get('sprint_count', 0)               * 3.5
          + s.get('total_distance_km', 0)           * 14
          + s.get('accelerations', 0)               * 0.9
          + s.get('decelerations', 0)               * 1.3
          + s.get('hard_accelerations', 0)          * 2.5
          + s.get('hard_decelerations', 0)          * 3.0
          + s.get('max_speed', 0)                   * 1.8
          + s.get('high_intensity_distance_km', 0)  * 22
          + s.get('very_hi_intensity_km', 0)        * 18
          + s.get('sprint_distance_km', 0)          * 60
          + s.get('direction_changes', 0)           * 0.5
        )
        return min(score * m, 500)

    def _fatigue_label(self, score):
        levels = [
            (450, 'CRITICAL',     TIER_COLORS['CRITICAL']),
            (380, 'VERY HIGH',    TIER_COLORS['HIGH']),
            (300, 'HIGH',         TIER_COLORS['HIGH']),
            (240, 'MODERATE-HIGH',TIER_COLORS['MODERATE']),
            (180, 'MODERATE',     TIER_COLORS['MODERATE']),
            (100, 'MILD',         TIER_COLORS['LOW']),
            (0,   'LOW',          TIER_COLORS['MINIMAL']),
        ]
        for thresh, label, color in levels:
            if score >= thresh:
                return label, color
        return 'LOW', TIER_COLORS['MINIMAL']

    def _rest_days(self, tier, fatigue):
        if tier == 'CRITICAL' or fatigue >= 420: return 4
        if tier == 'HIGH'     or fatigue >= 310: return 3
        if tier == 'MODERATE' or fatigue >= 200: return 2
        if tier == 'LOW'      or fatigue >= 100: return 1
        return 0

    # ── Tissue-targeted pain relief protocols ─────────────────────────────────
    def _pain_relief(self, tissue, tier):
        protocols = []
        sorted_t = sorted(tissue.items(), key=lambda x: x[1], reverse=True)

        PROTOCOLS = {
            'Hamstring': {
                'immediate': [
                    'Ice pack posterior thigh 20 min × 3 sessions today',
                    'Compression shorts or bandage for first 24h',
                    'Avoid any explosive movement for 24h',
                ],
                'next_48h': [
                    'Nordic hamstring curl — bodyweight only, 3 × 8',
                    'Prone hip extension stretch — 3 × 30s each side',
                    'Foam roll: glute → hamstring → calf (slow, 10 min)',
                    'PNF stretch with partner or band if available',
                ],
                'medication': 'Ibuprofen 400mg with food if pain > 4/10 (consult physio). Topical Diclofenac gel twice daily on posterior thigh.',
            },
            'Quadriceps': {
                'immediate': [
                    'Ice anterior thigh 15 min × 3 sessions',
                    'Elevate legs when resting — 30° above heart',
                    'Avoid stairs and downhill walking',
                ],
                'next_48h': [
                    'Standing quad stretch 3 × 40s each side',
                    'Leg press — light load (30% 1RM) 3 × 15, stop at discomfort',
                    'Step-down eccentric: 3 × 10 slow descents on each leg',
                    'Stationary bike (low resistance) 15 min for tissue perfusion',
                ],
                'medication': 'Topical Diclofenac gel on anterior thigh if tender. NSAIDs if swelling present.',
            },
            'Calf/Achilles': {
                'immediate': [
                    'Ice calf + Achilles tendon 20 min every 2h for first 6h',
                    'Avoid barefoot walking on hard surfaces',
                    'Heel raise insoles to off-load Achilles',
                ],
                'next_48h': [
                    'Eccentric heel drops off step: 3 × 15 reps each leg',
                    'Gastrocnemius stretch (straight knee) 3 × 45s',
                    'Soleus stretch (bent knee) 3 × 45s',
                    'Soft tissue massage on calf belly — moderate pressure',
                ],
                'medication': 'NSAIDs if swelling present. Magnesium glycinate 300mg nightly reduces cramp risk.',
            },
            'Hip Flexors': {
                'immediate': [
                    'Heat pack anterior hip 15 min (NOT ice — hip flexors respond better to heat)',
                    'Supine knees-to-chest stretch hold 30s × 4',
                    'Rest in supine with pillow under knees',
                ],
                'next_48h': [
                    'Thomas stretch 3 × 40s each side',
                    'Hip flexor activation with light resistance band',
                    'Pigeon pose yoga stretch 2 × 60s each side',
                    'Consider dry needling if chronic tightness persists — physio referral',
                ],
                'medication': 'Warming topical gel (capsaicin-based) for chronic tightness. No NSAIDs unless acute strain confirmed.',
            },
            'Knee/Ligament': {
                'immediate': [
                    'Ice knee 20 min every 2h — do NOT apply directly to skin',
                    'Compression bandage: firm but not circulation-restricting',
                    'Avoid pivoting, cutting and loaded twisting',
                    'Elevation when seated',
                ],
                'next_48h': [
                    'Straight-leg raises 3 × 20 (VMO activation)',
                    'Stationary bike low resistance 15 min — joint lubrication',
                    'Single-leg balance board 3 × 45s per leg',
                    'Terminal knee extensions with resistance band',
                ],
                'medication': 'Ibuprofen 400mg with food if swelling present. Significant swelling → consult doctor (may require aspiration/ultrasound).',
            },
        }

        for tissue_name, score in sorted_t:
            if score < 18:
                continue
            p = PROTOCOLS.get(tissue_name)
            if p:
                protocols.append({
                    'target':    tissue_name,
                    'score':     round(score, 1),
                    'tier':      self._tissue_tier(score),
                    'color':     TIER_COLORS[self._tissue_tier(score)],
                    **p,
                })

        if not protocols:
            protocols.append({
                'target':    'General',
                'score':     0,
                'tier':      'MINIMAL',
                'color':     TIER_COLORS['MINIMAL'],
                'immediate': ['Contrast shower (2 min hot / 30s cold × 5 cycles)',
                              'Full-body foam roll 10 min'],
                'next_48h':  ['8–9h sleep for muscle protein synthesis',
                              'Dynamic mobility flow 15 min'],
                'medication': 'No specific medication required.',
            })
        return protocols

    # ── Day-by-day schedule ───────────────────────────────────────────────────
    def _schedule(self, tier, pos):
        pos_note = {
            'Attacking':  'Sprint mechanics assessment before return to full sprint work',
            'Middle':     'Include agility ladder from Day 4 / Day 6 onwards',
            'Defensive':  'Hip abductor + glute activation priority before return',
        }.get(pos, '')

        base = {
            'MINIMAL': [
                {'day': 1, 'label': 'Day 1', 'activity': 'Full training', 'intensity': '100%',
                 'notes': 'Standard warm-up + cool-down'},
            ],
            'LOW': [
                {'day': 1, 'label': 'Day 1', 'activity': 'Active recovery — pool or light cycling', 'intensity': '25%',
                 'notes': 'HR < 130 bpm, no running'},
                {'day': 2, 'label': 'Day 2', 'activity': 'Mobility + light gym (upper body)', 'intensity': '45%',
                 'notes': 'Avoid heavy lower body load'},
                {'day': 3, 'label': 'Day 3', 'activity': 'Return to full training', 'intensity': '100%',
                 'notes': 'Extend warm-up by 10 min. ' + pos_note},
            ],
            'MODERATE': [
                {'day': 1, 'label': 'Day 1', 'activity': 'Complete rest + ice / compression', 'intensity': '0%',
                 'notes': 'Ice 20 min × 3, elevate legs, no standing for long periods'},
                {'day': 2, 'label': 'Day 2', 'activity': 'Pool walking + upper body gym', 'intensity': '15%',
                 'notes': 'No impact running, HR < 120 bpm'},
                {'day': 3, 'label': 'Day 3', 'activity': 'Light jog (20 min, flat surface)', 'intensity': '35%',
                 'notes': 'Stop immediately if pain > 3/10'},
                {'day': 4, 'label': 'Day 4', 'activity': 'Tempo run + ball work', 'intensity': '60%',
                 'notes': 'No sprinting or sharp turns'},
                {'day': 5, 'label': 'Day 5', 'activity': 'Full group training', 'intensity': '90%',
                 'notes': 'Physio clearance advised. ' + pos_note},
            ],
            'HIGH': [
                {'day': 1, 'label': 'Day 1', 'activity': 'Complete rest', 'intensity': '0%',
                 'notes': 'Ice bath 10 min, compression sleeves, sleep 9–10h'},
                {'day': 2, 'label': 'Day 2', 'activity': 'Complete rest + physio assessment', 'intensity': '0%',
                 'notes': '⚠️ Physio sign-off mandatory before progressing'},
                {'day': 3, 'label': 'Day 3', 'activity': 'Hydrotherapy + passive stretching', 'intensity': '10%',
                 'notes': 'Pool only, no weight-bearing sprint or jump'},
                {'day': 4, 'label': 'Day 4', 'activity': 'Bike / elliptical', 'intensity': '25%',
                 'notes': 'Monitor for referred pain or tightness'},
                {'day': 5, 'label': 'Day 5', 'activity': 'Light jog — straight lines only', 'intensity': '40%',
                 'notes': 'No cutting, turning, or deceleration drills'},
                {'day': 6, 'label': 'Day 6', 'activity': 'Controlled training with ball', 'intensity': '60%',
                 'notes': 'No competitive contact or full sprints'},
                {'day': 7, 'label': 'Day 7', 'activity': 'Full training if symptom-free', 'intensity': '85%',
                 'notes': '✅ Physio + coach sign-off mandatory. ' + pos_note},
            ],
            'CRITICAL': [
                {'day': 1, 'label': 'Day 1', 'activity': 'Complete rest', 'intensity': '0%',
                 'notes': '🚨 Ice bath 15 min, full compression, 10h+ sleep target'},
                {'day': 2, 'label': 'Day 2', 'activity': 'Complete rest', 'intensity': '0%',
                 'notes': '🚨 Mandatory medical/physio assessment today'},
                {'day': 3, 'label': 'Day 3', 'activity': 'Complete rest or gentle hydrotherapy', 'intensity': '0%',
                 'notes': 'Physio-directed only'},
                {'day': 4, 'label': 'Day 4', 'activity': 'Pool walking / gentle cycling', 'intensity': '10%',
                 'notes': 'Only if cleared by physio'},
                {'day': 5, 'label': 'Day 5', 'activity': 'Low-impact cardio (bike/swim)', 'intensity': '20%',
                 'notes': 'HR < 120 bpm, no impact loading'},
                {'day': 6, 'label': 'Day 6', 'activity': 'Walking + mobility', 'intensity': '30%',
                 'notes': 'Progress only if pain < 2/10'},
                {'day': 7, 'label': 'Day 7', 'activity': 'Light jog if symptom-free', 'intensity': '40%',
                 'notes': 'Medical clearance required'},
                {'day': 8, 'label': 'Day 8', 'activity': 'Team assessment + return decision', 'intensity': '60%',
                 'notes': '✅ Full medical + physio + coach sign-off. ' + pos_note},
            ],
        }
        return base.get(tier, base['MODERATE'])

    # ── Key metrics list ──────────────────────────────────────────────────────
    def _key_metrics(self, s):
        items = [
            ('sprint_count',                '{} explosive sprints'),
            ('total_distance_km',           '{:.2f} km total distance'),
            ('high_intensity_distance_km',  '{:.2f} km hi-intensity'),
            ('very_hi_intensity_km',        '{:.2f} km very-hi-intensity'),
            ('sprint_distance_km',          '{:.2f} km sprint distance'),
            ('accelerations',               '{} rapid accelerations'),
            ('hard_accelerations',          '{} hard accelerations (>4 m/s²)'),
            ('decelerations',               '{} hard decelerations'),
            ('hard_decelerations',          '{} very hard decelerations (>4 m/s²)'),
            ('max_speed',                   '{:.1f} km/h top speed'),
            ('direction_changes',           '{} direction changes'),
        ]
        out = []
        for key, fmt in items:
            v = s.get(key, 0)
            if v and v > 0:
                out.append(fmt.format(v))
        return out

    def _workload_dict(self, s):
        return {
            'total_distance_km':          s.get('total_distance_km', 0),
            'hi_intensity_km':            s.get('high_intensity_distance_km', 0),
            'very_hi_intensity_km':       s.get('very_hi_intensity_km', 0),
            'sprint_distance_km':         s.get('sprint_distance_km', 0),
            'sprint_count':               s.get('sprint_count', 0),
            'max_speed_kmh':              s.get('max_speed', 0),
            'hard_accelerations':         s.get('hard_accelerations', 0),
            'hard_decelerations':         s.get('hard_decelerations', 0),
            'direction_changes':          s.get('direction_changes', 0),
            'speed_bands': {
                'walk':   s.get('walk_pct', 0),
                'jog':    s.get('jog_pct', 0),
                'run':    s.get('run_pct', 0),
                'fast':   s.get('fast_pct', 0),
                'sprint': s.get('sprint_pct', 0),
            },
        }

    # ── Position insights ─────────────────────────────────────────────────────
    def _position_insights(self, pos, s):
        base = {
            'Attacking': [
                'High explosive sprint demand detected',
                'Quad and calf loading primary concern',
                'Focus: lower leg and hamstring recovery',
            ],
            'Middle': [
                'Extreme cardiovascular demand with high direction-change volume',
                'Full-body fatigue with knee and calf stress from repeated cutting',
                'Focus: cardiovascular recovery + ankle/knee stability work',
            ],
            'Defensive': [
                'High eccentric loading from repeated stopping and marking',
                'Knee and hamstring stress from duelling and interceptions',
                'Focus: quad/knee prehab and hip-abductor activation',
            ],
        }
        insights = base.get(pos, base['Middle'])[:]

        if s.get('direction_changes', 0) > 150:
            insights.append(f"⚡ {s['direction_changes']} direction changes — well above average")
        if s.get('sprint_count', 0) > 25:
            insights.append(f"⚡ {s['sprint_count']} sprints — elevated hamstring risk")
        if s.get('max_speed', 0) > 28:
            insights.append(f"⚡ Max speed {s['max_speed']} km/h — above sprint risk threshold")
        return insights

    # ── Prescription (for legacy UI support) ─────────────────────────────────
    def _prescription(self, fatigue, tier, pos, s):
        rest_map = {
            'CRITICAL': (4, 'Drink 3.5L+ water today, 2.8L next 3 days', 'CRITICAL: 10h+ for 4 nights'),
            'HIGH':     (3, 'Drink 3.0L water today, 2.5L next 2 days',  'Aim 9–10h for 3 nights'),
            'MODERATE': (2, 'Drink 2.5L water today',                     'Aim 8.5–9h for 2 nights'),
            'LOW':      (1, 'Drink 2.0L water today',                     'Aim 8h tonight'),
            'MINIMAL':  (0, 'Normal hydration (3L)',                      '7–8h recommended'),
        }
        rest, hydration, sleep = rest_map.get(tier, rest_map['MODERATE'])
        activities = [d['activity'] for d in self._schedule(tier, pos)][:3]
        nutrition  = [self._nutrition(fatigue, tier, s)]
        return {
            'rest_days':  rest,
            'hydration':  hydration,
            'sleep':      sleep,
            'activities': activities,
            'nutrition':  nutrition,
        }

    # ── Warnings ──────────────────────────────────────────────────────────────
    def _warnings(self, tier, overall, flags, fatigue, s):
        w = []
        if tier == 'CRITICAL':
            w.append(f'🚨 CRITICAL injury risk ({overall:.0f}/100) — immediate medical assessment REQUIRED')
            w.append('Do NOT train until cleared by medical staff')
        elif tier == 'HIGH':
            w.append(f'⚠️ HIGH injury risk ({overall:.0f}/100) — physio consultation strongly advised')
            w.append('48h rest minimum before any intensive activity')
        elif tier == 'MODERATE':
            w.append(f'Moderate risk ({overall:.0f}/100) — monitor symptoms carefully')

        for f in flags[:3]:
            if f['probability_pct'] > 45:
                w.append(f"High {f['tissue']} risk ({f['probability_pct']}%) — {f['cause']}")

        if fatigue >= 380:
            w.append('Extreme fatigue detected — minimum 4 days before next match')
        elif fatigue >= 300:
            w.append('High fatigue — allow 48–72h full recovery')
        elif fatigue >= 200:
            w.append('Moderate fatigue — 24–48h light activity only')

        if s.get('sprint_count', 0) > 40:
            w.append('Very high sprint count — monitor hamstring tightness for 72h')
        return w

    # ── Legacy text timeline ──────────────────────────────────────────────────
    def _timeline(self, fatigue, tier):
        return [f"{d['label']}: {d['activity']} ({d['intensity']})"
                for d in self._schedule(tier, 'Middle')]

    # ── Nutrition ─────────────────────────────────────────────────────────────
    def _nutrition(self, fatigue, tier, s):
        pos          = self._normalise_position(s.get('position', 'Middle'))
        sprint_count = s.get('sprint_count', 0)
        max_speed    = s.get('max_speed', 0)
        hard_dec     = s.get('hard_decelerations', 0)
        dir_changes  = s.get('direction_changes', 0)

        # Base note varies by tier
        if tier == 'CRITICAL':
            base = 'CRITICAL load: high-carb + 40g protein meal within 20 min post-match.'
            extras = [
                'Omega-3 (2g) + Collagen (15g) + Vitamin C before sleep — tissue repair.',
                'Electrolyte drink every 2h for 8h — sodium, potassium, magnesium.',
                'Target 2.0–2.2g protein per kg bodyweight across next 24h.',
            ]
        elif tier == 'HIGH':
            base = 'High load: carb + protein recovery meal within 30 min post-match.'
            extras = [
                'Omega-3 (1–2g) daily to reduce inflammation.',
                'Electrolyte drink within 1h — sodium + potassium critical.',
                'Target 1.8–2.0g protein per kg bodyweight in 24h recovery window.',
            ]
        elif tier == 'MODERATE':
            base = 'Moderate load: carb-rich meal within 45 min post-match.'
            extras = [
                'Standard electrolyte replenishment post-session.',
                'Target 1.6–1.8g protein per kg bodyweight.',
            ]
        else:
            base = 'Low load: normal balanced meal within 60 min post-match.'
            extras = ['Standard electrolyte replenishment.', 'Target 1.4–1.6g protein per kg.']

        # Position-specific nutrition note
        if pos == 'Attacking':
            extras.append('Attacker priority: fast-release carbs pre-training (banana + rice cakes) to fuel sprint demands.')
        elif pos == 'Middle':
            extras.append('Midfielder priority: complex carbs (oats, sweet potato) for sustained energy over high-volume distance.')
        elif pos == 'Defensive':
            extras.append('Defender priority: anti-inflammatory foods (salmon, berries, turmeric) to counter hard-deceleration joint load.')

        # Metric-specific extras
        if sprint_count > 20:
            extras.append(f'High sprint count ({sprint_count}): extra magnesium (300mg nightly) for cramp prevention.')
        if max_speed > 26:
            extras.append(f'Max speed {max_speed} km/h: L-Glutamine (5g) post-match for high-speed tissue repair.')
        if hard_dec > 10:
            extras.append(f'{hard_dec} hard decelerations: glucosamine + chondroitin to protect knee cartilage.')
        if dir_changes > 130:
            extras.append(f'{dir_changes} direction changes: collagen peptides (10g) + Vit-C to support tendon recovery.')

        return ' '.join([base] + extras[:4])

    def _diet(self, fatigue, s):
        """
        Detailed diet structure — varies by fatigue level, position, dominant
        tissue risk and specific high-load metrics so no two players get the
        same plan.
        """
        pos          = self._normalise_position(s.get('position', 'Middle'))
        sprint_count = s.get('sprint_count', 0)
        dir_changes  = s.get('direction_changes', 0)
        max_speed    = s.get('max_speed', 0)
        hard_dec     = s.get('hard_decelerations', 0)
        hard_acc     = s.get('hard_accelerations', 0)
        vhi          = s.get('very_hi_intensity_km', 0)

        # ── Base macros by fatigue tier ────────────────────────────────────────
        if fatigue >= 380:
            protein   = '2.0–2.2g/kg'
            carbs     = '8–10g/kg'
            hydration = '55–60 ml/kg'
        elif fatigue >= 280:
            protein   = '1.8–2.0g/kg'
            carbs     = '6–8g/kg'
            hydration = '45–55 ml/kg'
        elif fatigue >= 180:
            protein   = '1.6–1.8g/kg'
            carbs     = '5–7g/kg'
            hydration = '40–50 ml/kg'
        else:
            protein   = '1.4–1.6g/kg'
            carbs     = '4–6g/kg'
            hydration = '35–45 ml/kg'

        # ── Core supplement base (everyone gets these at minimum) ──────────────
        base_supps = ['Electrolyte drink (sodium + potassium)', 'Vitamin C 500mg']

        # ── Fatigue-tier additions ─────────────────────────────────────────────
        if fatigue >= 380:
            base_supps += ['Whey protein (30–40g post-match)', 'BCAAs 5g × 3 daily',
                           'Omega-3 2g anti-inflammatory', 'Zinc 25mg (immune + muscle repair)',
                           'Vitamin D 2000 IU', 'Collagen peptides 15g + Vit-C before sleep']
        elif fatigue >= 280:
            base_supps += ['Whey protein (25–30g post-match)', 'BCAAs 5g × 2 daily',
                           'Omega-3 1g daily']
        elif fatigue >= 180:
            base_supps += ['Whey protein (20g post-match)', 'Vitamin C 1000mg post-exercise']
        else:
            base_supps += ['Light electrolyte replenishment only']

        # ── Position-specific supplement additions ─────────────────────────────
        if pos == 'Attacking':
            base_supps.append('Creatine monohydrate 3–5g/day (sprint power maintenance)')
            base_supps.append('Beta-alanine 3g/day (sprint endurance buffer)')
        elif pos == 'Middle':
            base_supps.append('Magnesium glycinate 300mg nightly (cramp prevention from high volume)')
            base_supps.append('Iron + B12 check recommended if sustained high-distance load')
        elif pos == 'Defensive':
            base_supps.append('Glucosamine 1500mg/day (knee joint support from hard deceleration load)')
            base_supps.append('Curcumin/Turmeric 500mg (joint inflammation — stopping/marking load)')

        # ── Per-metric targeted additions (creates unique plans per player) ────
        if sprint_count > 20:
            if 'Magnesium glycinate 300mg nightly (cramp prevention from high volume)' not in base_supps:
                base_supps.append('Magnesium glycinate 300mg nightly (sprint-induced cramp prevention)')
        if max_speed > 26:
            base_supps.append('L-Glutamine 5g post-match (high-speed tissue micro-tear repair)')
        if hard_dec > 12:
            if 'Glucosamine 1500mg/day (knee joint support from hard deceleration load)' not in base_supps:
                base_supps.append('Glucosamine 1500mg + Chondroitin (deceleration-loaded knee support)')
        if vhi > 1.0:
            base_supps.append('Tart cherry juice 30ml × 2 daily (reduces oxidative stress from sustained fast running)')
        if dir_changes > 150:
            base_supps.append('Collagen peptides 10g + Vit-C pre-training (tendon/ligament resilience)')
        if hard_acc > 12:
            base_supps.append('Carnitine 1g/day (mitochondrial recovery from repeated sprint-start load)')

        # De-duplicate while preserving order
        seen = set()
        unique_supps = []
        for s_item in base_supps:
            key = s_item[:30]
            if key not in seen:
                seen.add(key)
                unique_supps.append(s_item)

        # ── Meal timing — also varies by fatigue/position ─────────────────────
        timing = ['Post-match high-carb + protein meal within 30 min']
        if fatigue >= 280:
            timing.append('Protein snack (20–25g) every 3h during 24h recovery window')
            timing.append('Casein protein shake (30g) before sleep for overnight muscle repair')
            timing.append('Breakfast: oatmeal + eggs + banana within 1h of waking')
        elif fatigue >= 180:
            timing.append('Protein snack every 4–5h during recovery day')
            timing.append('Casein protein before sleep recommended')
        else:
            timing.append('Normal meal schedule — ensure post-session protein intake')

        if pos == 'Attacking':
            timing.append('Pre-training: 30g fast carbs 45 min before session (sprint fuel)')
        elif pos == 'Middle':
            timing.append('Pre-training: 50g complex carbs 90 min before (sustained cardio fuel)')
        elif pos == 'Defensive':
            timing.append('Post-session: anti-inflammatory meal (salmon + leafy greens + turmeric)')

        return {
            'macronutrients': {
                'protein':   protein,
                'carbs':     carbs,
                'hydration': hydration,
            },
            'supplements': unique_supps[:9],   # cap to avoid UI overflow
            'timing':      timing,
        }