"""Deterministic, disjoint layout banks. No learned policy selects these seeds."""
import numpy as np
import config

FAMILIES = ('ring_default', 'irregular_100', 'irregular_085',
            'irregular_070', 'irregular_055', 'irregular_edge')
SPLIT_BASE = {'train': 1000000, 'validation': 2000000, 'test': 3000000}
SPACING = (1., 1., .85, .70, .55, .85)


def scenario_bank(split):
    if split not in SPLIT_BASE:
        raise ValueError(f'Unknown scenario split: {split}')
    count = getattr(config, split.upper() + '_LAYOUTS_PER_FAMILY')
    if not 1 <= count <= 90:
        raise ValueError('Layout count must be 1..90 to keep seed ranges disjoint')
    return [dict(split=split, family=family, layout_index=i,
                 layout_seed=SPLIT_BASE[split]+10000*j+101*i,
                 rollout_seed=SPLIT_BASE[split]+500000+10000*j+101*i)
            for i in range(count) for j,family in enumerate(FAMILIES)]


def training_scenario(episode):
    bank = scenario_bank('train')
    return bank[episode % len(bank)]


def ap_positions(spec):
    j = FAMILIES.index(spec['family'])
    rng = np.random.default_rng(np.random.SeedSequence([spec['layout_seed'],18011]))
    angles = np.arange(config.NUM_APS)*2*np.pi/config.NUM_APS
    radii = np.full(config.NUM_APS, .35*config.AREA_SIZE)
    if j:
        angles += rng.uniform(-.18,.18,config.NUM_APS)
        radii *= rng.uniform(.75,1.15,config.NUM_APS)
    return (np.column_stack((np.cos(angles),np.sin(angles))) *
            (SPACING[j]*radii[:,None]) + config.AREA_SIZE/2).tolist()


def place_edge_stas(spec, aps, stas):
    if spec['family'] != 'irregular_edge':
        return
    for sta in stas:
        ap = aps[sta['ap_id']]
        dx,dy = sta['x']-ap['x'],sta['y']-ap['y']
        factor = .9*config.AP_STA_RADIUS_MAX[ap['id']]/max(np.hypot(dx,dy),1e-12)
        sta['x'],sta['y'] = ap['x']+factor*dx,ap['y']+factor*dy
