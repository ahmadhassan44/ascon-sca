# CLAUDE.md — Lab 11: Side-Channel Analysis of ASCON-128

# CS-360 Cyber Security | BSCS-2K23 | Spring 2026

## Context

End-to-end Deep Learning Side-Channel Attack (DL-SCA) on the ASCON-128 lightweight cipher.
This is a university lab project. No GPU on the Ubuntu host — Phase 4 scripts are written
locally but executed on Google Colab.

## Project Layout

```
lab11-ascon-sca/
├── CLAUDE.md
├── phase2/
│   ├── ascon128.h
│   ├── ascon128.c
│   ├── test_vectors.c
│   └── Makefile
├── phase3/
│   ├── generate_traces.py
│   └── visualize_traces.py
├── phase4_colab/
│   ├── attack_fixed_key.py
│   └── attack_variable_key.py
└── submission/
    ├── package.json
    ├── generate_doc.js
    └── Lab11_Submission.docx   ← produced by generate_doc.js
```

---

## TASK 1 — Phase 2: C Implementation

### Do this first. Do not proceed to Task 2 until `make native` shows all PASS.

Create all files under `phase2/`.

---

### `phase2/ascon128.h`

```c
#ifndef ASCON128_H
#define ASCON128_H

#include <stdint.h>
#include <stddef.h>

#define ASCON128_KEY_LEN   16
#define ASCON128_NONCE_LEN 16
#define ASCON128_TAG_LEN   16

typedef struct {
    uint64_t x[5];
} ascon_state_t;

/* Encrypt plaintext. Writes ciphertext (same length as plaintext) + 16-byte tag.
   ct must have capacity plaintext_len + ASCON128_TAG_LEN. */
void ascon128_encrypt(
    const uint8_t *key,        /* 16 bytes */
    const uint8_t *nonce,      /* 16 bytes */
    const uint8_t *ad,         /* associated data, may be NULL */
    size_t         ad_len,
    const uint8_t *plaintext,
    size_t         plaintext_len,
    uint8_t       *ciphertext  /* out: plaintext_len + 16 bytes */
);

/* Returns 0 on success (tag valid), -1 on authentication failure. */
int ascon128_decrypt(
    const uint8_t *key,
    const uint8_t *nonce,
    const uint8_t *ad,
    size_t         ad_len,
    const uint8_t *ciphertext,
    size_t         ciphertext_len, /* includes 16-byte tag */
    uint8_t       *plaintext       /* out: ciphertext_len - 16 bytes */
);

/* Expose permutation for trace generation hooks */
void ascon_permutation(ascon_state_t *s, int rounds);

#endif /* ASCON128_H */
```

---

### `phase2/ascon128.c`

Implement the full ASCON-128 AEAD specification. Every detail below is from the official spec.

**Byte order:** The 64-bit state words are big-endian (most significant byte first).
On a little-endian host (x86/ARM LE), load/store with byte-swap.
Use this helper:

```c
static uint64_t load64(const uint8_t *b) {
    return ((uint64_t)b[0] << 56) | ((uint64_t)b[1] << 48) |
           ((uint64_t)b[2] << 40) | ((uint64_t)b[3] << 32) |
           ((uint64_t)b[4] << 24) | ((uint64_t)b[5] << 16) |
           ((uint64_t)b[6] <<  8) |  (uint64_t)b[7];
}
static void store64(uint8_t *b, uint64_t v) {
    b[0]=v>>56; b[1]=v>>48; b[2]=v>>40; b[3]=v>>32;
    b[4]=v>>24; b[5]=v>>16; b[6]=v>>8;  b[7]=v;
}
#define ROTR(x,n) (((x)>>(n))|((x)<<(64-(n))))
```

**Round constants** (XOR into x2 before S-box each round, indexed 0..11):

```
0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b
```

For p(a) starting at round (12-a): use constants[(12-a)..11].

**S-box** (applied to 5-bit columns across all 64 columns of the state, implemented as
operations on the five 64-bit words simultaneously):

```c
static void ascon_sbox(ascon_state_t *s) {
    uint64_t t0, t1, t2, t3, t4;
    s->x[0] ^= s->x[4]; s->x[4] ^= s->x[3]; s->x[2] ^= s->x[1];
    t0=s->x[0]; t1=s->x[1]; t2=s->x[2]; t3=s->x[3]; t4=s->x[4];
    s->x[0] = t0 ^ (~t1 & t2);
    s->x[1] = t1 ^ (~t2 & t3);
    s->x[2] = t2 ^ (~t3 & t4);
    s->x[3] = t3 ^ (~t4 & t0);
    s->x[4] = t4 ^ (~t0 & t1);
    s->x[1] ^= s->x[0]; s->x[0] ^= s->x[4];
    s->x[3] ^= s->x[2]; s->x[2] = ~s->x[2];
}
```

**Linear diffusion** (sigma functions, one per word):

```c
static void ascon_linear(ascon_state_t *s) {
    s->x[0] ^= ROTR(s->x[0],19) ^ ROTR(s->x[0],28);
    s->x[1] ^= ROTR(s->x[1],61) ^ ROTR(s->x[1],39);
    s->x[2] ^= ROTR(s->x[2], 1) ^ ROTR(s->x[2], 6);
    s->x[3] ^= ROTR(s->x[3],10) ^ ROTR(s->x[3],17);
    s->x[4] ^= ROTR(s->x[4], 7) ^ ROTR(s->x[4],41);
}
```

**Permutation p(a)** — `a` rounds, starting from round index `(12-a)`:

```c
void ascon_permutation(ascon_state_t *s, int rounds) {
    static const uint8_t RC[12] = {
        0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b
    };
    int start = 12 - rounds;
    for (int i = start; i < 12; i++) {
        s->x[2] ^= RC[i];   /* AddConstant */
        ascon_sbox(s);       /* SubBytes    */
        ascon_linear(s);     /* LinearDiffusion */
    }
}
```

**Initialization** (ASCON-128, rate=64 bits, key=128 bits):

```
IV = 0x80400c0600000000
State after load: x0=IV, x1=key[0..7], x2=key[8..15], x3=nonce[0..7], x4=nonce[8..15]
Run p12.
x3 ^= key_word0; x4 ^= key_word1;
```

**Associated Data** (skip loop if ad_len == 0, but always do the domain separation):

```
For each 8-byte block of AD (pad last block with 0x80 then zeros):
  x0 ^= block
  run p6
After all AD (including if ad_len==0): x4 ^= 1
```

**Encryption** (rate = 8 bytes per block):

```
For each full 8-byte plaintext block (not the last if plaintext_len is multiple of 8):
  x0 ^= block
  ciphertext_block = x0
  run p6
For the last block (may be 0..8 bytes, len t):
  pad: 8-byte buffer = 0x00*8, copy t plaintext bytes, set byte[t] = 0x80
  x0 ^= padded_block
  ciphertext = top t bytes of x0
  (no p6 after last block)
Special case: if plaintext_len is exactly a multiple of 8 and > 0,
  the last full block still gets p6, then process an empty "last block" with just padding.
  Simpler: treat the block before last differently only when plaintext_len % 8 == 0.
  Recommended approach: process all full blocks with p6, then always process one partial/empty block.
```

**Finalization**:

```
x1 ^= key_word0; x2 ^= key_word1;
run p12
x3 ^= key_word0; x4 ^= key_word1;
tag = store64(x3) || store64(x4)
```

Write `ascon128_encrypt` and `ascon128_decrypt` using the above.
For decrypt: XOR ciphertext block into x0 to get plaintext (x0 ^ ct), then set x0 = ct for next block.
For authentication: constant-time tag comparison (XOR all 16 bytes, check == 0).

---

### `phase2/test_vectors.c`

Cross-check against the reference C implementation at https://github.com/ascon/ascon-c
which has a `genkat_aead.c` producing the official KAT file. Use at minimum:

**Vector 1** — empty plaintext, empty AD:

```
Key   = 000102030405060708090a0b0c0d0e0f
Nonce = 000102030405060708090a0b0c0d0e0f
PT    = (empty, 0 bytes)
AD    = (empty, 0 bytes)
Expected Tag = e355a03750b7da3868e79c93a8aee5f2
               (verify against ascon-c KAT if this doesn't match)
```

**Vector 2** — 1-byte plaintext, empty AD:

```
Key   = 000102030405060708090a0b0c0d0e0f
Nonce = 000102030405060708090a0b0c0d0e0f
PT    = 00
AD    = (empty)
Derive expected CT+Tag from the ascon-c reference, then hardcode and verify.
```

**Vector 3** — encrypt then decrypt round-trip:

```
Random key, nonce, 32-byte plaintext, 8-byte AD.
Verify decrypt(encrypt(pt)) == pt and returns 0.
Also verify that flipping one tag byte causes decrypt to return -1.
```

For each vector: call the function, compare output byte-by-byte, print `[PASS]` or `[FAIL]`.
Exit 0 if all pass, 1 if any fail.

---

### `phase2/Makefile`

```makefile
CC_NATIVE  = gcc
CC_ARM     = arm-none-eabi-gcc
CFLAGS     = -O2 -Wall -Wextra -std=c99

.PHONY: native arm clean

native: ascon128.c test_vectors.c ascon128.h
	$(CC_NATIVE) $(CFLAGS) -o test_native ascon128.c test_vectors.c
	./test_native

arm: ascon128.c ascon128.h
	$(CC_ARM) $(CFLAGS) -mcpu=cortex-m3 -mthumb -c -o ascon128.o ascon128.c
	$(CC_ARM) -mcpu=cortex-m3 -mthumb -nostdlib \
	    -Wl,--entry=ascon128_encrypt -o ascon128.elf ascon128.o 2>/dev/null || \
	    mv ascon128.o ascon128.elf
	@echo "ARM ELF produced: ascon128.elf"
	@file ascon128.elf || true

clean:
	rm -f test_native ascon128.o ascon128.elf
```

Install ARM toolchain if missing: `sudo apt install gcc-arm-none-eabi`

---

## TASK 2 — Phase 3: Trace Generation

### Do not start until Phase 2 `make native` is fully passing.

Create files under `phase3/`. Install deps:

```bash
pip install rainbow h5py numpy matplotlib tqdm
```

---

### `phase3/generate_traces.py`

**Rainbow API check first:**

```python
import rainbow; print(dir(rainbow))
```

Use `from rainbow.generics import arm` if available, otherwise `from rainbow import ElfSimulator`.
Rainbow loads an ARM ELF and emulates instruction-by-instruction, logging Hamming Weight
of register values as a power trace proxy.

**Target:** `x[2]` of `ascon_state_t` immediately after the first call to `ascon_sbox()`
in round 0 of the initialization permutation. This is the standard profiling attack target
for ASCON — it is a nonlinear function of `(key XOR nonce)`.

**Leakage model:** `label = bin(x2_after_first_sbox & 0xff).count('1')` — HW of lowest byte,
range 0..8, giving 9 classification targets.

**Script structure:**

```python
import os, sys, h5py, numpy as np
from tqdm import tqdm

ELF_PATH    = "../phase2/ascon128.elf"
TRACE_LEN   = 1000
N_PROFILING = 40000
N_ATTACK    = 10000
FIXED_KEY   = bytes(range(16))
OUTPUT_DIR  = "."

def hw(v):
    return bin(int(v) & 0xff).count('1')

def downsample(trace, target_len):
    arr = np.array(trace, dtype=np.float32)
    if len(arr) <= target_len:
        return np.pad(arr, (0, target_len - len(arr)))
    factor = len(arr) // target_len
    return arr[:factor * target_len].reshape(target_len, factor).mean(axis=1)

def run_ascon_trace(sim, key_bytes, nonce_bytes):
    """
    Load key and nonce into the simulator memory, call ascon128_encrypt
    with empty plaintext and AD, capture the power trace, and extract
    the x[2] state value after the first S-box call via a memory/register hook.
    Returns (trace_array, label).
    Adapt this function to the actual Rainbow API available.
    """
    raise NotImplementedError("Implement based on installed Rainbow version")

def generate_dataset(sim, n_profiling, n_attack, fixed_key=None):
    prof_traces, prof_labels, prof_nonces = [], [], []
    atk_traces,  atk_labels,  atk_nonces = [], [], []
    prof_keys, atk_keys = [], []

    for i in tqdm(range(n_profiling + n_attack), desc="Generating traces"):
        key   = fixed_key if fixed_key else bytes(np.random.randint(0, 256, 16))
        nonce = bytes(np.random.randint(0, 256, 16))
        trace, label = run_ascon_trace(sim, key, nonce)
        trace = downsample(trace, TRACE_LEN)
        if i < n_profiling:
            prof_traces.append(trace); prof_labels.append(label); prof_nonces.append(nonce)
            if fixed_key is None: prof_keys.append(key)
        else:
            atk_traces.append(trace);  atk_labels.append(label);  atk_nonces.append(nonce)
            if fixed_key is None: atk_keys.append(key)

    return (np.array(prof_traces, dtype=np.float32),
            np.array(prof_labels,  dtype=np.uint8),
            np.array(prof_nonces,  dtype=np.uint8),
            np.array(atk_traces,  dtype=np.float32),
            np.array(atk_labels,   dtype=np.uint8),
            np.array(atk_nonces,   dtype=np.uint8),
            np.array(prof_keys,    dtype=np.uint8) if not fixed_key else None,
            np.array(atk_keys,     dtype=np.uint8) if not fixed_key else None)

# --- Fixed-key dataset ---
# sim = load Rainbow simulator here
# pt, pl, pn, at, al, an, _, _ = generate_dataset(sim, N_PROFILING, N_ATTACK, FIXED_KEY)
# with h5py.File("fixed_key_traces.h5", "w") as f:
#   f.create_dataset("profiling/traces", data=pt)
#   f.create_dataset("profiling/labels", data=pl)
#   f.create_dataset("profiling/plaintexts", data=pn)
#   f.create_dataset("attack/traces", data=at)
#   f.create_dataset("attack/labels", data=al)
#   f.create_dataset("attack/plaintexts", data=an)
#   f["metadata/key"] = np.frombuffer(FIXED_KEY, dtype=np.uint8)
#   f["metadata/leakage_model"] = "HammingWeight_x2_sbox_round0_byte0"

# --- Variable-key dataset ---
# Ensure attack keys do not appear in profiling keys (generate disjoint key pools)
# Add /profiling/keys and /attack/keys datasets
```

Fill in `run_ascon_trace` based on the actual Rainbow API. Print dataset shapes and label
distribution after writing each file.

---

### `phase3/visualize_traces.py`

Load `fixed_key_traces.h5`. Save all output to `phase3/plots/` (create if needed).

1. `trace_overlay.png` — 10 profiling traces overlaid, distinct colors, labeled by HW value
2. `mean_trace.png` — mean of all profiling traces with std band shaded
3. `label_dist.png` — bar chart of HW label distribution (x=0..8, y=count)

Print full paths of saved files.

---

## TASK 3 — Phase 4: DL Attack Scripts (Colab-ready)

Create files under `phase4_colab/`. Add this header to each:

```python
# ============================================================
# Run on Google Colab with GPU runtime (T4 or better)
# Upload fixed_key_traces.h5 / variable_key_traces.h5 to Colab
# Runtime -> Change runtime type -> T4 GPU
# ============================================================
```

---

### `phase4_colab/attack_fixed_key.py`

Sections: load dataset → normalize → build model → train → evaluate → key rank analysis → plots.

**Model:**

```python
from tensorflow.keras import layers, models

def build_model(trace_len):
    inp = layers.Input(shape=(trace_len, 1))
    x = layers.Conv1D(64, 11, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(128, 11, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(9, activation='softmax')(x)
    return models.Model(inp, out)
```

**Training:** Adam(lr=1e-3), sparse_categorical_crossentropy, 50 epochs, batch_size=256.
Save: `model_fixed_key.h5`.

**Key rank analysis** (for key byte 0):

```
true_key_byte = fixed_key[0]   # = 0x00
For each candidate k in 0..255:
  score[k] = 0
  For each of N attack traces (trace_i, nonce_i):
    predicted_hw_class = hw(sbox_output(k ^ nonce_i[0]))  # compute expected HW
    score[k] += log(model.predict(trace_i)[predicted_hw_class] + 1e-36)
Rank of true_key_byte = number of candidates with score > score[true_key_byte]
Repeat for N = 1..1000, plot rank vs N.
```

Note: `sbox_output(v)` here means the HW of the intermediate after the ASCON S-box given
input `v`. Since the S-box is fixed, precompute a lookup table:

```python
def ascon_sbox_hw(v):
    # Apply ASCON S-box to a single 5-bit value and return HW of result byte
    # Use the same logic as in ascon128.c but for scalar values
    ...
```

Save figures: `training_curves_fixed.png`, `key_rank_fixed.png`.

---

### `phase4_colab/attack_variable_key.py`

Same structure. Load `variable_key_traces.h5`. After training:

- Report accuracy vs fixed-key model
- Run key rank analysis (use the per-trace key stored in `/attack/keys`)
- Add a printed analysis block comparing generalization

Save: `model_variable_key.h5`, `training_curves_variable.png`, `key_rank_variable.png`.

---

## TASK 4 — Submission Word Document

Run after all other tasks are done (or at minimum after Phase 2 passes).

```bash
cd submission
npm install
node generate_doc.js
# Open Lab11_Submission.docx, replace [FILL] sections, insert screenshots
```

---

### `submission/package.json`

```json
{
  "name": "lab11-submission",
  "version": "1.0.0",
  "dependencies": {
    "docx": "^8.5.0"
  }
}
```

---

### `submission/generate_doc.js`

Generate `Lab11_Submission.docx`. The script must be complete and runnable — no TODOs.

**Imports needed:**

```javascript
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  Footer,
  AlignmentType,
  HeadingLevel,
  BorderStyle,
  WidthType,
  ShadingType,
  PageNumber,
  TabStopType,
  TabStopPosition,
} = require("docx");
const fs = require("fs");
```

**Page setup:** US Letter (12240 x 15840 DXA), 1-inch margins (1440 DXA each side).

**Footer** (on every page):

- Left: "CS-360 Cyber Security | Lab 11 | Spring 2026"
- Right: page number
- Use tab stop RIGHT at max position: `tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }]`

**Styles:**

- Body: Arial 12pt
- Heading 1: Arial 16pt bold (phase headings)
- Heading 2: Arial 13pt bold (sub-sections)
- Heading 3: Arial 12pt bold italic (deliverable mapping labels)

**Placeholder text helper:**

```javascript
function fillText(text) {
  return new TextRun({
    text: "[FILL] " + text,
    italics: true,
    color: "888888",
    font: "Arial",
    size: 24,
  });
}
```

**Screenshot box helper** — a paragraph that looks like a placeholder box:

```javascript
function screenshotBox(description) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "999999" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "999999" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "999999" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "999999" },
    },
    shading: { type: ShadingType.CLEAR, fill: "F5F5F5" },
    spacing: { before: 200, after: 200 },
    children: [
      new TextRun({
        text: "[ Screenshot: " + description + " ]",
        italics: true,
        color: "999999",
        font: "Arial",
        size: 22,
      }),
    ],
  });
}
```

**Section divider helper:**

```javascript
function divider() {
  return new Paragraph({
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC", space: 1 },
    },
    spacing: { before: 200, after: 200 },
    children: [],
  });
}
```

**Info table** (2 columns, widths 2500 + 6860 = 9360 DXA):

```javascript
function infoRow(label, value, isPlaceholder = false) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "DDDDDD" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new TableRow({
    children: [
      new TableCell({
        width: { size: 2500, type: WidthType.DXA },
        borders,
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        shading: { type: ShadingType.CLEAR, fill: "F0F0F0" },
        children: [
          new Paragraph({
            children: [
              new TextRun({ text: label, bold: true, font: "Arial", size: 22 }),
            ],
          }),
        ],
      }),
      new TableCell({
        width: { size: 6860, type: WidthType.DXA },
        borders,
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [
          new Paragraph({
            children: [
              isPlaceholder
                ? new TextRun({
                    text: value,
                    italics: true,
                    color: "888888",
                    font: "Arial",
                    size: 22,
                  })
                : new TextRun({ text: value, font: "Arial", size: 22 }),
            ],
          }),
        ],
      }),
    ],
  });
}
```

**Full document content — implement all sections:**

```
Title block:
  "Lab 11: Open Ended Lab"              Heading 1 centered
  "Side-Channel Analysis of ASCON-128"  Heading 2 centered
  [blank paragraph]
  Info table with rows:
    Course           | CS-360 Cyber Security
    Class            | BSCS-2K23
    Student Name     | [FILL] Your Full Name          (placeholder)
    Registration No  | [FILL] Your Registration No    (placeholder)
    Lab Instructor   | Ms. Hadia Tahir
    Class Instructor | Dr. Madiha Khalid
    Date             | 14th April 2026

divider()

"Phase 1: Research and Understanding"     Heading 1
"[Maps to Lab Deliverable: ASCON Study Report]"   Heading 3

"1.1 Overview of ASCON-128 Cipher"       Heading 2
Paragraph with fillText("Write approximately 300 words covering: what ASCON is, selected
as NIST Lightweight Cryptography Standard in 2023, winner of CAESAR competition (2019),
designed for resource-constrained environments (IoT/embedded), 128-bit security level,
family variants (ASCON-128, ASCON-128a, ASCON-Hash, ASCON-XOF), sponge-based construction,
320-bit internal state as 5x64-bit words.")

"1.2 AEAD Mode Explanation"              Heading 2
Paragraph with fillText("Write approximately 200 words covering: what Authenticated Encryption
with Associated Data (AEAD) means, how ASCON implements it through four phases (initialization,
associated data absorption, encryption, finalization), the role of the nonce, how the 128-bit
authentication tag is produced, why associated data is authenticated but not encrypted.")

"1.3 Permutation and Internal Structure" Heading 2
Paragraph with fillText("Write approximately 300 words covering: the 320-bit state as five
64-bit words x0..x4, the three-step permutation (AddConstant: XOR round constant into x2;
SubBytes: nonlinear S-box applied column-wise via bitsliced operations on all five words;
LinearDiffusion: word-specific rotations — x0 uses 19/28, x1 uses 61/39, x2 uses 1/6, x3
uses 10/17, x4 uses 7/41). p12 used in initialization and finalization. p6 used in the
encryption body. Round constants from 0xf0 down to 0x4b.")

"1.4 Potential SCA Leakage Points"       Heading 2
Paragraph with fillText("Write approximately 200 words identifying: (1) x2 after S-box in
round 0 of init permutation — depends on key XOR nonce, primary profiling attack target used
in this lab; (2) Key XOR during finalization — x1 and x2 are XORed with key before p12;
(3) Tag generation — x3 and x4 directly contain key material after the final XOR; (4) All
intermediate state words across permutation rounds. Explain why the Hamming Weight model
applies: on unprotected SW implementations, power is proportional to popcount(value) due to
switching activity on the data bus and register file.")

divider()

"Phase 2: Implementation"                Heading 1
"[Maps to Lab Deliverable: C Source Code + Technical Write-up]"  Heading 3

"2.1 Implementation Approach"            Heading 2
Paragraph with fillText("Write approximately 200 words describing: how you structured
ascon128.c, state representation as uint64_t[5], big-endian byte handling on the little-endian
host using manual shift-based load64/store64 helpers, modular breakdown into ascon_sbox /
ascon_linear / ascon_permutation / init / encrypt / finalize, padding handling for the last
plaintext block (0x80 byte then zero fill), constant-time tag comparison in decrypt.")

"2.2 Challenges Faced and Solutions"     Heading 2
Paragraph with fillText("Write approximately 150 words describing challenges encountered and
how you resolved them. Likely issues: endianness confusion when loading key/nonce bytes into
64-bit state words, off-by-one in round constant indexing for p(a) vs p12, correct handling
of the last plaintext block with no p6 after it, ARM cross-compilation flags and linker
behavior with -nostdlib.")

"2.3 Testing Methodology"                Heading 2
Paragraph with fillText("Write approximately 100 words describing: which test vectors you
used (official ASCON KAT file from ascon.iaik.tugraz.at), how you ran them via make native,
what the PASS/FAIL terminal output looks like, and whether you cross-checked against the
reference ascon-c implementation on GitHub.")

screenshotBox("Terminal output of `make native` showing all test vectors [PASS]")
screenshotBox("Terminal output of `make arm` showing ascon128.elf produced with file type confirmation")

divider()

"Phase 3: Trace Generation"              Heading 1
"[Maps to Lab Deliverable: generate_traces.py + Dataset Documentation]"  Heading 3

"3.1 Trace Generation Process"           Heading 2
Paragraph with fillText("Write approximately 200 words describing: the Rainbow framework
as an ARM ELF emulator that simulates instruction-level Hamming Weight leakage, how
ascon128.elf is loaded into the simulator, how a memory or register hook captures x[2]
after the first S-box call, total traces generated (40,000 profiling + 10,000 attack for
fixed-key; same split for variable-key), HDF5 storage format with profiling/attack/metadata
groups, and downsampling to 1,000 points when raw traces are longer.")

"3.2 Leakage Model Explanation"          Heading 2
Paragraph with fillText("Write approximately 150 words: the Hamming Weight (HW) model
defines power consumption as proportional to the number of 1-bits in the value being
processed: HW(v) = popcount(v). This applies to unprotected software on ARM Cortex-M3
because register writes, bus transfers, and memory accesses all exhibit switching activity
proportional to the data value. The label for each trace is the HW of the lowest byte of
x[2] after the first S-box, ranging from 0 to 8, giving 9 classification targets.")

"3.3 Target Point Selection"             Heading 2
Paragraph with fillText("Write approximately 150 words justifying the choice of x[2] after
S-box in round 0 of the initialization permutation: this value is a nonlinear function of
(key XOR nonce byte), making it sensitive to the secret key while remaining observable early
in execution. Earlier measurement means less accumulated noise. This is the standard target
in published ASCON profiling attacks. The S-box's nonlinearity ensures the intermediate
value is not a simple XOR of key and nonce, forcing an attacker to enumerate key candidates.")

"3.4 Sample Trace Plots"                 Heading 2
Paragraph with fillText("Write approximately 80 words describing what the plots show: the
power profile shape across the 1,000-sample trace window, any visible structure corresponding
to the permutation rounds, and whether the label distribution across HW classes 0 to 8 is
approximately binomial (as expected for random inputs), or skewed.")

screenshotBox("trace_overlay.png — 10 overlaid power traces from profiling set, colored by HW label")
screenshotBox("mean_trace.png — mean power trace across all 40,000 profiling traces with standard deviation band")
screenshotBox("label_dist.png — Hamming Weight label distribution bar chart (classes 0 to 8)")

divider()

"Phase 4: Deep Learning Attack"          Heading 1
"[Maps to Lab Deliverable: Attack Scripts + Results and Analysis Report]"  Heading 3

"4.1 Model Architecture and Training Details"  Heading 2
Paragraph with fillText("Write approximately 200 words describing: the CNN architecture
(Input → Conv1D 64 filters kernel 11 → BatchNorm → AveragePooling2 → Conv1D 128 filters
kernel 11 → BatchNorm → AveragePooling2 → Flatten → Dense 256 ReLU → Dropout 0.5 →
Dense 128 ReLU → Dense 9 Softmax), input shape (TRACE_LEN, 1), optimizer Adam with
learning rate 1e-3, loss sparse categorical crossentropy, 50 epochs, batch size 256,
trace normalization via StandardScaler fit on profiling set, training performed on Google
Colab T4 GPU.")

"4.2 Attack Performance — Fixed vs Variable Key"  Heading 2
Paragraph with fillText("Write approximately 200 words reporting: final attack-set accuracy
for the fixed-key model, final attack-set accuracy for the variable-key model, how quickly
each converged (epoch at best validation accuracy), whether overfitting was observed (train
vs val accuracy gap), and the generalization gap between the two models. Include specific
numeric values from your Colab runs.")

screenshotBox("Training loss and accuracy curves — Fixed-Key model (from Colab)")
screenshotBox("Training loss and accuracy curves — Variable-Key model (from Colab)")

"4.3 Key Recovery Results"               Heading 2
Paragraph with fillText("Write approximately 150 words reporting: key rank of the correct
key byte at N = 100, 500, and 1,000 attack traces for both the fixed-key and variable-key
models. State whether the fixed-key model successfully recovers the correct key (rank reaches
0) and approximately how many traces are required. State whether the variable-key model
generalizes and achieves key recovery.")

screenshotBox("Key rank evolution plot — Fixed-Key attack (key rank vs number of attack traces, 1 to 1,000)")
screenshotBox("Key rank evolution plot — Variable-Key attack (key rank vs number of attack traces, 1 to 1,000)")

"4.4 Critical Analysis and Observations" Heading 2
Paragraph with fillText("Write approximately 250 words analyzing: why profiling attacks on
a fixed-key device succeed (the model learns a stable trace-to-label mapping for one device
and key); why generalizing to variable keys is harder (key-dependent trace features shift,
requiring the model to learn key-agnostic representations); what countermeasures would
defeat this attack (Boolean masking randomizes intermediate values, instruction shuffling
breaks the trace-to-operation alignment, noise injection degrades SNR); how this maps to
real IoT threats (unprotected ASCON firmware on ARM Cortex-M microcontrollers is practically
attackable using an oscilloscope or EM probe with thousands of traces); and the limitations
of this simulation (Rainbow's HW model is idealized — real traces include measurement noise,
EM coupling, clock jitter, and device-specific variation not captured in software emulation).")
```

After building all content, write to file:

```javascript
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Lab11_Submission.docx", buf);
  console.log("Written: Lab11_Submission.docx");
});
```

---

## Execution Order

```bash
# Step 1 — Phase 2 (must PASS before anything else)
cd phase2
sudo apt install gcc-arm-none-eabi   # if not installed
make native        # all [PASS] required
make arm           # produces ascon128.elf

# Step 2 — Phase 3
pip install rainbow h5py numpy matplotlib tqdm
cd ../phase3
python generate_traces.py     # ~30-60 min, produces .h5 files
python visualize_traces.py    # produces plots/

# Step 3 — Phase 4 (on Colab)
# Upload phase3/fixed_key_traces.h5 and variable_key_traces.h5 to Colab
# Run phase4_colab/attack_fixed_key.py on Colab
# Run phase4_colab/attack_variable_key.py on Colab
# Download screenshots and model .h5 files

# Step 4 — Word doc
cd ../submission
npm install
node generate_doc.js
# Open Lab11_Submission.docx
# Replace every [FILL] section with actual content
# Insert screenshots into the placeholder boxes
```
