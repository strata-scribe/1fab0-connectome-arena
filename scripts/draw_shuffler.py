#!/usr/bin/env python3
"""
Off-Harness Control Graph Generator (Grant 1fab0)
Implements the 'authored-elsewhere' custody rule (c59011 / c59078).

Generates, cryptographically hashes, and seals a pool of degree-preserving
Maslov-Sneppen configuration model matrices independently of the simulation harness.
The simulation engine is strictly a read-only consumer of this sealed bank.
"""

import os
import sys
import json
import hashlib
import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.connectome import FlyConnectome, NUM_NEURONS, SYM_PAIR

def generate_sealed_controls(num_controls=50, num_swaps=50, output_path=None):
    conn = FlyConnectome(seed=42)
    controls = []
    
    for i in range(num_controls):
        seed = 1000 + i * 37
        W_shuf = conn.generate_shuffled_control(num_swaps=num_swaps, seed=seed)
        
        # Verify strict invariants
        assert conn.verify_degrees_conserved(W_shuf), f'Control #{i}: Degree sequence not conserved'
        assert conn.verify_bilateral_symmetry(W_shuf), f'Control #{i}: Bilateral symmetry violated'
        
        # Verify Dale's Principle
        for u in range(NUM_NEURONS):
            orig_signs = set(np.sign(conn.W[u, conn.W[u, :] != 0]))
            shuf_signs = set(np.sign(W_shuf[u, W_shuf[u, :] != 0]))
            if len(orig_signs) > 0 and len(shuf_signs) > 0:
                assert orig_signs == shuf_signs, f'Control #{i}: Dale principle violated at neuron {u}'
        
        mat_bytes = W_shuf.tobytes()
        mat_hash = hashlib.sha256(mat_bytes).hexdigest()
        
        controls.append({
            'id': i,
            'seed': seed,
            'sha256': mat_hash,
            'num_swaps': num_swaps,
            'degrees_conserved': True,
            'symmetry_verified': True,
            'matrix': W_shuf.tolist()
        })
    
    bank = {
        'format': '1fab0-sealed-controls-v1',
        'author': 'off-harness-generator',
        'custody_rule': 'authored-elsewhere (c59011/c59078)',
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'num_controls': len(controls),
        'num_neurons': NUM_NEURONS,
        'controls': controls
    }
    
    if output_path is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, 'sealed_controls.json')
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bank, f, indent=2)
        
    print(f'[SHUFFLER] Successfully generated and sealed {len(controls)} controls into {output_path}')
    print(f'[SHUFFLER] All controls verified: degree-invariant, bilateral-symmetric, Dale compliant.')
    return output_path

def verify_sealed_controls(path):
    if not os.path.exists(path):
        print(f'[ERROR] Sealed controls file not found: {path}')
        return False
    with open(path, 'r', encoding='utf-8') as f:
        bank = json.load(f)
    conn = FlyConnectome(seed=42)
    controls = bank.get('controls', [])
    print(f'[VERIFY] Checking {len(controls)} sealed controls from {bank.get("author")}...')
    for item in controls:
        W = np.array(item['matrix'], dtype=np.float32)
        assert conn.verify_degrees_conserved(W), f'Failed degree invariance on control #{item["id"]}'
        assert conn.verify_bilateral_symmetry(W), f'Failed symmetry on control #{item["id"]}'
        h = hashlib.sha256(W.tobytes()).hexdigest()
        assert h == item['sha256'], f'Hash mismatch on control #{item["id"]}'
    print(f'[VERIFY] ✓ All {len(controls)} controls verified against canonical G_traced constraints.')
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        p = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'data', 'sealed_controls.json')
        ok = verify_sealed_controls(p)
        sys.exit(0 if ok else 1)
    else:
        generate_sealed_controls()
