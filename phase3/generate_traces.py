# ============================================================
# Phase 3 — ASCON-128 Power Trace Generation via Rainbow
#
# Uses Ledger-Donjon's rainbow (ARM Cortex-M3 emulator on top
# of Unicorn) to execute ascon128.elf and collect Hamming-Weight
# register leakage as a power trace.
#
# Leakage target:
#   x[2] of ascon_state_t immediately after the FIRST call to
#   ascon_sbox() in round 0 of the initialisation permutation.
#   We capture it by hooking the ENTRY of ascon_linear() (called
#   right after ascon_sbox() in every permutation round).
#   label = HW(x[2] & 0xff)  →  9 classes (0–8)
#
# Run:
#   python generate_traces.py
# ============================================================

import os
import h5py
import numpy as np
from tqdm import tqdm
import unicorn as uc

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

STATE_ADDR   = 0x20000000   # 40 bytes  (5 × uint64, big-endian)
RETURN_ADDR  = 0xdeadbeef   # sentinel — execution stops here on function return


def hw(v: int) -> int:
    return bin(int(v) & 0xff).count('1')


def downsample(trace: list, target_len: int) -> np.ndarray:
    arr = np.array(trace, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(arr) <= target_len:
        return np.pad(arr, (0, target_len - len(arr))).astype(np.float32)
    factor = max(1, len(arr) // target_len)
    return arr[:factor * target_len].reshape(target_len, factor).mean(axis=1).astype(np.float32)


def build_emulator() -> rainbow_cortexm:
    """Create a shared emulator instance with HW register leakage tracing."""
    em = rainbow_cortexm(trace_config=TraceConfig(register=HammingWeight()))
    em.load(ELF_PATH, typ='.elf')
    em.map_space(STATE_ADDR, STATE_ADDR + 0x100)
    return em


def setup_hooks(em: rainbow_cortexm) -> list:
    """
    Register a Unicorn block hook on ascon_linear's entry address.

    Rainbow's hook_prolog registers at the symbol address (Thumb bit
    set, odd), but Unicorn fires UC_HOOK_BLOCK at the even instruction
    address.  We use em.emu directly with addr & ~1 to match correctly.

    captured[0] is reset to None before each trace run; the hook
    stores x[2] only on the first fire per trace (None sentinel).
    """
    captured = [None]
    linear_addr = em.functions['ascon_linear'][0] & ~1  # strip Thumb bit

    def block_hook(uci, addr, size, user_data):
        if addr == linear_addr and captured[0] is None:
            state_ptr = em['r0']
            x2_bytes  = bytes(em[state_ptr + 16: state_ptr + 24])
            captured[0] = int.from_bytes(x2_bytes, 'big')

    em.emu.hook_add(uc.UC_HOOK_BLOCK, block_hook,
                    begin=linear_addr, end=linear_addr)
    return captured


def run_trace(em: rainbow_cortexm,
              captured: list,
              key: bytes,
              nonce: bytes) -> tuple:
    """
    Execute ascon_permutation(state, 12) with the ASCON-128 init state,
    collect power trace, return (trace_array, label).
    """
    def be64(b): return int.from_bytes(b, 'big')

    IV          = 0x80400c0600000000
    k0, k1      = be64(key[:8]),   be64(key[8:])
    n0, n1      = be64(nonce[:8]), be64(nonce[8:])
    state_words = [IV, k0, k1, n0, n1]
    state_bytes = b''.join(w.to_bytes(8, 'big') for w in state_words)

    # Reset capture sentinel and trace
    captured[0] = None
    em.reset_trace()
    em.reset_regs()

    # Write initial state and set up registers
    em[STATE_ADDR] = state_bytes
    em['r0']       = STATE_ADDR
    em['r1']       = 12
    em['lr']       = RETURN_ADDR

    perm_addr = em.functions['ascon_permutation'][0]
    em.start(perm_addr, RETURN_ADDR, timeout=10_000_000)

    raw_trace = [e.get("register", 0) for e in em.trace if isinstance(e, dict)]
    trace     = downsample(raw_trace, TRACE_LEN)
    label     = hw(captured[0]) if captured[0] is not None else 0

    return trace, label


# ── Dataset generation ─────────────────────────────────────

def generate_dataset(em, captured, n_profiling, n_attack, fixed_key=None):
    rng = np.random.default_rng(seed=42 if fixed_key else 7)

    prof_traces, prof_labels, prof_nonces = [], [], []
    atk_traces,  atk_labels,  atk_nonces  = [], [], []
    prof_keys,   atk_keys                  = [], []

    total = n_profiling + n_attack
    desc  = "fixed-key" if fixed_key else "variable-key"

    for i in tqdm(range(total), desc=f"Generating {desc} traces"):
        key   = fixed_key if fixed_key else bytes(rng.integers(0, 256, 16).tolist())
        nonce = bytes(rng.integers(0, 256, 16).tolist())

        trace, label = run_trace(em, captured, key, nonce)

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


def save_fixed_key(em, captured):
    print("\n=== Fixed-key dataset ===")
    pt, pl, pn, at, al, an, _, _ = generate_dataset(
        em, captured, N_PROFILING, N_ATTACK, fixed_key=FIXED_KEY
    )
    path = os.path.join(OUTPUT_DIR, "fixed_key_traces.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("profiling/traces",     data=pt, compression="gzip")
        f.create_dataset("profiling/labels",     data=pl)
        f.create_dataset("profiling/plaintexts", data=pn)
        f.create_dataset("attack/traces",        data=at, compression="gzip")
        f.create_dataset("attack/labels",        data=al)
        f.create_dataset("attack/plaintexts",    data=an)
        f["metadata/key"]           = np.frombuffer(FIXED_KEY, dtype=np.uint8)
        f["metadata/leakage_model"] = "HammingWeight_x2_sbox_round0_byte0"
    print(f"Saved: {path}")
    print(f"  profiling: {pt.shape}   attack: {at.shape}")
    u, c = np.unique(pl, return_counts=True)
    print("  Label dist:", dict(zip(u.tolist(), c.tolist())))


def save_variable_key(em, captured):
    print("\n=== Variable-key dataset ===")
    rng = np.random.default_rng(seed=99)

    all_keys  = [bytes(rng.integers(0, 256, 16).tolist())
                 for _ in range(N_PROFILING + N_ATTACK + 500)]
    prof_keys = list(dict.fromkeys(all_keys))[:N_PROFILING]
    remaining = [k for k in all_keys if k not in set(prof_keys)]
    atk_keys  = list(dict.fromkeys(remaining))[:N_ATTACK]

    def _gen(keys_list, desc):
        rng2 = np.random.default_rng(seed=999)
        traces, labels, nonces, keys = [], [], [], []
        for key in tqdm(keys_list, desc=desc):
            nonce = bytes(rng2.integers(0, 256, 16).tolist())
            trace, label = run_trace(em, captured, key, nonce)
            traces.append(trace); labels.append(label)
            nonces.append(list(nonce)); keys.append(list(key))
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
    print("Functions:", list(em.functions.keys()))

    captured = setup_hooks(em)

    save_fixed_key(em, captured)
    save_variable_key(em, captured)
    print("\nDone.")
