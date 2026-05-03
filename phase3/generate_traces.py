# ============================================================
# Phase 3 — ASCON-128 Power Trace Generation via Rainbow
#
# Uses Ledger-Donjon's rainbow (ARM Cortex-M3 emulator on top
# of Unicorn) to execute ascon128.elf instruction-by-instruction
# and collect Hamming-Weight register leakage as a power trace.
#
# Leakage target:
#   x[2] of ascon_state_t immediately after the FIRST call to
#   ascon_sbox() in round 0 of the initialisation permutation.
#   label = HW(x[2] & 0xff)  →  9 classes (0–8)
#
# Run:
#   python generate_traces.py
# ============================================================

import os
import sys
import struct
import h5py
import numpy as np
from tqdm import tqdm

from rainbow.generics.cortexm import rainbow_cortexm
from rainbow import TraceConfig, HammingWeight

# ── paths & constants ──────────────────────────────────────
ELF_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "../phase2/ascon128.elf")
TRACE_LEN    = 1000
N_PROFILING  = 40000
N_ATTACK     = 10000
FIXED_KEY    = bytes(range(16))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# Memory layout inside the emulator
STATE_ADDR   = 0x20000000   # 40 bytes  (5 × uint64)
KEY_ADDR     = 0x20000100   # 16 bytes
NONCE_ADDR   = 0x20000200   # 16 bytes
DUMMY_CT_ADDR= 0x20000300   # 32 bytes  (empty PT encrypt → just tag)


def hw(v: int) -> int:
    return bin(int(v) & 0xff).count('1')


def downsample(trace: list, target_len: int) -> np.ndarray:
    arr = np.array(trace, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(arr) <= target_len:
        return np.pad(arr, (0, target_len - len(arr))).astype(np.float32)
    factor = max(1, len(arr) // target_len)
    trimmed = arr[:factor * target_len]
    return trimmed.reshape(target_len, factor).mean(axis=1).astype(np.float32)


def build_emulator() -> rainbow_cortexm:
    """Create a fresh emulator with HW register leakage tracing."""
    em = rainbow_cortexm(trace_config=TraceConfig(register=HammingWeight))
    em.load(ELF_PATH, typ='.elf')

    # Map memory regions
    em.map_space(STATE_ADDR,    STATE_ADDR    + 0x100)
    em.map_space(KEY_ADDR,      KEY_ADDR      + 0x100)
    em.map_space(NONCE_ADDR,    NONCE_ADDR    + 0x100)
    em.map_space(DUMMY_CT_ADDR, DUMMY_CT_ADDR + 0x100)

    return em


def run_trace(em: rainbow_cortexm,
              key: bytes, nonce: bytes) -> tuple[np.ndarray, int]:
    """
    Execute ascon_permutation(state, 12) with the ASCON-128 init state
    pre-loaded, collect the power trace, and capture x[2] after the
    first ascon_sbox() call.

    Returns (trace_array, label).
    """
    # Build the initial ASCON-128 state (big-endian 64-bit words)
    def be64(b): return int.from_bytes(b, 'big')

    IV   = 0x80400c0600000000
    k0   = be64(key[:8]);  k1 = be64(key[8:])
    n0   = be64(nonce[:8]); n1 = be64(nonce[8:])
    state_words = [IV, k0, k1, n0, n1]

    # Write state into emulator memory (each word as 8 big-endian bytes)
    state_bytes = b''.join(w.to_bytes(8, 'big') for w in state_words)
    em[STATE_ADDR] = state_bytes

    # Write key and nonce (used to finish init after permutation, but we
    # only need the permutation trace here)
    em[KEY_ADDR]   = key
    em[NONCE_ADDR] = nonce

    # Reset CPU state
    em.reset()
    em[STATE_ADDR] = state_bytes   # reset() clears memory too

    # ── Hook: capture x[2] after FIRST ascon_sbox() call ──
    captured = {}
    sbox_addr  = em.functions["ascon_sbox"]
    call_count = [0]

    def sbox_hook(em_ref):
        call_count[0] += 1
        if call_count[0] == 1:
            # r0 holds ascon_state_t*, x[2] is at offset 16 (2×8 bytes)
            state_ptr = em_ref["r0"]
            x2_bytes  = bytes(em_ref[state_ptr + 16: state_ptr + 24])
            captured['x2'] = int.from_bytes(x2_bytes, 'big')

    em.stub(sbox_addr, sbox_hook)

    # Reset trace and run permutation
    em.reset_trace()
    perm_addr = em.functions["ascon_permutation"]

    em["r0"] = STATE_ADDR
    em["r1"] = 12

    # end address: 0xdeadbeef — execution stops when function returns
    em.start(perm_addr, 0xdeadbeef, timeout=10_000_000)

    raw_trace = em.trace
    trace     = downsample(raw_trace, TRACE_LEN)
    label     = hw(captured.get('x2', 0))

    return trace, label


# ── Dataset generation ─────────────────────────────────────

def generate_dataset(em, n_profiling, n_attack, fixed_key=None):
    rng = np.random.default_rng(seed=42 if fixed_key else 7)

    prof_traces, prof_labels, prof_nonces = [], [], []
    atk_traces,  atk_labels,  atk_nonces  = [], [], []
    prof_keys,   atk_keys                  = [], []

    total = n_profiling + n_attack
    desc  = "fixed-key" if fixed_key else "variable-key"

    for i in tqdm(range(total), desc=f"Generating {desc} traces"):
        key   = fixed_key if fixed_key else bytes(rng.integers(0, 256, 16))
        nonce = bytes(rng.integers(0, 256, 16))

        trace, label = run_trace(em, key, nonce)

        if i < n_profiling:
            prof_traces.append(trace)
            prof_labels.append(label)
            prof_nonces.append(list(nonce))
            if not fixed_key:
                prof_keys.append(list(key))
        else:
            atk_traces.append(trace)
            atk_labels.append(label)
            atk_nonces.append(list(nonce))
            if not fixed_key:
                atk_keys.append(list(key))

    return (np.array(prof_traces, dtype=np.float32),
            np.array(prof_labels,  dtype=np.uint8),
            np.array(prof_nonces,  dtype=np.uint8),
            np.array(atk_traces,  dtype=np.float32),
            np.array(atk_labels,   dtype=np.uint8),
            np.array(atk_nonces,   dtype=np.uint8),
            np.array(prof_keys,    dtype=np.uint8) if not fixed_key else None,
            np.array(atk_keys,     dtype=np.uint8) if not fixed_key else None)


def save_fixed_key(em):
    print("\n=== Fixed-key dataset ===")
    pt, pl, pn, at, al, an, _, _ = generate_dataset(
        em, N_PROFILING, N_ATTACK, fixed_key=FIXED_KEY
    )
    path = os.path.join(OUTPUT_DIR, "fixed_key_traces.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("profiling/traces",     data=pt, compression="gzip")
        f.create_dataset("profiling/labels",     data=pl)
        f.create_dataset("profiling/plaintexts", data=pn)
        f.create_dataset("attack/traces",        data=at, compression="gzip")
        f.create_dataset("attack/labels",        data=al)
        f.create_dataset("attack/plaintexts",    data=an)
        f["metadata/key"]            = np.frombuffer(FIXED_KEY, dtype=np.uint8)
        f["metadata/leakage_model"]  = "HammingWeight_x2_sbox_round0_byte0"
    print(f"Saved: {path}")
    print(f"  profiling: {pt.shape}   attack: {at.shape}")
    u, c = np.unique(pl, return_counts=True)
    print("  Label dist:", dict(zip(u.tolist(), c.tolist())))


def save_variable_key(em):
    print("\n=== Variable-key dataset ===")
    rng = np.random.default_rng(seed=99)

    # Generate disjoint key pools
    all_keys  = [bytes(rng.integers(0, 256, 16)) for _ in range(N_PROFILING + N_ATTACK + 500)]
    prof_keys = list(dict.fromkeys(all_keys))[:N_PROFILING]
    remaining = [k for k in all_keys if k not in set(prof_keys)]
    atk_keys  = list(dict.fromkeys(remaining))[:N_ATTACK]

    def _gen(keys_list, desc):
        traces, labels, nonces, keys = [], [], [], []
        rng2 = np.random.default_rng(seed=999)
        for key in tqdm(keys_list, desc=desc):
            nonce = bytes(rng2.integers(0, 256, 16))
            trace, label = run_trace(em, key, nonce)
            traces.append(trace)
            labels.append(label)
            nonces.append(list(nonce))
            keys.append(list(key))
        return (np.array(traces, dtype=np.float32),
                np.array(labels, dtype=np.uint8),
                np.array(nonces, dtype=np.uint8),
                np.array(keys,   dtype=np.uint8))

    pt, pl, pn, pk = _gen(prof_keys, "variable-key profiling")
    at, al, an, ak = _gen(atk_keys,  "variable-key attack")

    path = os.path.join(OUTPUT_DIR, "variable_key_traces.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("profiling/traces",     data=pt, compression="gzip")
        f.create_dataset("profiling/labels",     data=pl)
        f.create_dataset("profiling/plaintexts", data=pn)
        f.create_dataset("profiling/keys",       data=pk)
        f.create_dataset("attack/traces",        data=at, compression="gzip")
        f.create_dataset("attack/labels",        data=al)
        f.create_dataset("attack/plaintexts",    data=an)
        f.create_dataset("attack/keys",          data=ak)
        f["metadata/leakage_model"] = "HammingWeight_x2_sbox_round0_byte0"
    print(f"Saved: {path}")
    print(f"  profiling: {pt.shape}   attack: {at.shape}")
    u, c = np.unique(pl, return_counts=True)
    print("  Label dist:", dict(zip(u.tolist(), c.tolist())))


if __name__ == "__main__":
    print("Loading ELF ...")
    em = build_emulator()
    print(f"Functions: {list(em.function_names.values())}")

    save_fixed_key(em)
    save_variable_key(em)
    print("\nDone.")
