"""Generate all subsection Mermaid (.mmd) sources for the Kimi K3-in-Rust blog.

Adapted from the C project's generator. Diagrams that describe the MODEL or the
CHECKPOINT are inherited verbatim, because they are properties of Kimi K3 and of
the released weights. The four that describe this implementation, build-flow,
kernel-contract, verification-ladder and recap, were written against measurements
from the Rust port. README.md in this directory lists the split.

Locked style: hand-drawn look on warm paper, soft colored fills + medium warm
borders on leaf boxes, dark warm ink for text. Each diagram is simple (3-8 nodes)
so it never overlaps and renders wide + short.
"""

import os

# Soft hand-drawn palette on warm paper: light fills, medium warm-leaning strokes,
# cohesive across all diagrams. The cold-to-hot spine reads blue -> amber -> coral.
INK = "#3B372F"          # warm fine-liner ink for text
COLORS = {
    "blue":   ("#E4ECF3", "#5C7C9B"),   # cold  (disk, inputs)
    "amber":  ("#F2E9D4", "#C0913E"),   # warm  (mid tier)
    "pink":   ("#F3E3EB", "#B06885"),   # soft rose
    "green":  ("#E7ECDD", "#6E9160"),   # sage  (result, correct)
    "purple": ("#EAE4F1", "#7E6E9C"),   # muted violet (decisions)
    "red":    ("#F5E2DB", "#B75845"),   # hot   (misses, the wall)
    "teal":   ("#DDEBE7", "#4E8E86"),   # soft teal
    "orange": ("#F4E7D3", "#C07A42"),   # sand  (warm-mid)
    "gray":   ("#ECE8E0", "#8C8477"),   # warm neutral (plain nodes)
}

DIAGRAMS = {}


def add(name, code, fills, containers=()):
    DIAGRAMS[name] = (code.strip(), fills, list(containers))


def render_style(fills, containers):
    lines = []
    for cid in containers:
        lines.append(f"    style {cid} fill:#F4EFE6,stroke:#C9C1B2,color:{INK}")
    for nid, ck in fills.items():
        f, s = COLORS[ck]
        lines.append(f"    style {nid} fill:{f},stroke:{s},color:{INK}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the big picture

add("big-picture", """
flowchart LR
    D["NVMe<br/>1.45 TB experts"] --> T["trunk.bin<br/>108.81 GB packed"]
    T --> PIN["pinned layers<br/>in RAM"]
    D --> C["expert cache<br/>17.56 MB slots"]
    PIN --> TOK["next token<br/>8.24 GB peak RSS"]
    C --> TOK
""", {"D": "gray", "T": "blue", "PIN": "purple", "C": "amber", "TOK": "green"})

add("moe-sparsity", """
flowchart LR
    X["one token"] --> R["router<br/>896 scores"]
    R --> TOP["top 16"]
    TOP --> RUN["16 experts run<br/>17.55 MB each"]
    R -.->|"880 stay asleep"| SLEEP["never read<br/>never resident"]
    RUN --> Y["layer output"]
""", {"X": "blue", "R": "purple", "TOP": "amber", "RUN": "green", "SLEEP": "gray", "Y": "teal"})

add("machine-model", """
flowchart LR
    L["laptop<br/>8 GB"] --> D2["desktop<br/>32 GB"]
    D2 --> W["workstation<br/>96 GB"]
    W --> S["our box<br/>228 GB"]
    L -.->|"same 8 tokens"| OUT["identical ids<br/>17374 20829 10 427"]
    S -.-> OUT
""", {"L": "red", "D2": "orange", "W": "amber", "S": "blue", "OUT": "green"})

# ---------------------------------------------------------------- build and load

add("build-flow", """
flowchart LR
    SRC["14 Rust modules"] --> CC["cargo build --release<br/>lto = thin"]
    CC --> BIN["target/release/k3<br/>1,159,344 bytes"]
    CC --> T["9 test binaries"]
    T --> G["43 passed<br/>0 failed"]
    BIN --> RUN["runs a 1.56 TB model"]
""", {"SRC": "gray", "CC": "blue", "BIN": "green", "T": "amber", "G": "teal", "RUN": "purple"})

add("st-load", """
flowchart LR
    S["96 shards<br/>1.56 TB"] --> H["read 8-byte length<br/>then JSON header"]
    H --> IDX["FNV-1a index<br/>497,220 tensors"]
    IDX --> Q{"need this<br/>tensor?"}
    Q -->|"yes"| P["pread exact bytes<br/>O_DIRECT"]
    Q -->|"no"| SKIP["never touched"]
""", {"S": "gray", "H": "blue", "IDX": "amber", "Q": "purple", "P": "green", "SKIP": "teal"})

add("config-guard", """
flowchart LR
    J["config.json"] --> R["k3_cfg_load"]
    R --> Q{"every field<br/>present?"}
    Q -->|"yes"| OK["93 layers<br/>24 MLA + 69 KDA"]
    Q -->|"no"| STOP["refuse to start<br/>name the field"]
    STOP -.->|"the old reader<br/>guessed here"| WRONG["all 93 layers KDA<br/>fluent and wrong"]
""", {"J": "gray", "R": "blue", "Q": "purple", "OK": "green", "STOP": "amber", "WRONG": "red"})

add("tok-flow", """
flowchart LR
    F["prompt from a FILE<br/>never argv"] --> B["raw UTF-8 bytes"]
    B --> BPE["byte-level BPE<br/>163,584 ranks"]
    BPE --> IDS["token ids"]
    IDS --> CHK{"decode back<br/>byte-exact?"}
    CHK -->|"45 of 45"| PASS["matches tiktoken"]
""", {"F": "gray", "B": "blue", "BPE": "amber", "IDS": "green", "CHK": "purple", "PASS": "teal"})

# ---------------------------------------------------------------- mxfp4 + kernels

add("mxfp4-decode", """
flowchart LR
    BY["one byte<br/>two nibbles"] --> LO["low nibble<br/>EVEN element"]
    BY --> HI["high nibble<br/>ODD element"]
    LO --> LUT["E2M1 lookup<br/>16 values"]
    HI --> LUT
    SC["E8M0 byte<br/>1 per 32 weights"] --> MUL["value x 2^(s-127)"]
    LUT --> MUL
""", {"BY": "gray", "LO": "blue", "HI": "amber", "LUT": "purple", "SC": "orange", "MUL": "green"})

add("kernel-contract", """
flowchart LR
    W["same weights"] --> SC["portable path"]
    W --> NEON["NEON path"]
    W --> AVX["AVX2 path"]
    SC --> H{"hash of<br/>the output"}
    NEON --> H
    AVX --> H
    H -->|"identical"| OK["bit for bit"]
""", {"W": "gray", "SC": "blue", "NEON": "amber", "AVX": "orange", "H": "purple", "OK": "green"})

# ---------------------------------------------------------------- attention

add("kda-flow", """
flowchart LR
    X["token in<br/>q, k, v projected"] --> CV["ShortConv k=4<br/>fused SiLU"]
    CV --> N["L2Norm<br/>q and k only"]
    N --> ST["decay, then<br/>write the delta"]
    ST --> RD["read the<br/>UPDATED state"]
    RD --> G["RMSNorm<br/>then gate"]
    ST -.->|"carried forward"| S["one state per layer<br/>626 MB in total"]
""", {"X": "blue", "CV": "amber", "N": "orange", "ST": "purple", "RD": "green", "G": "teal", "S": "gray"})

add("kda-state", """
flowchart LR
    T1["token 1"] --> S["one state<br/>96 x 128 x 128"]
    T2["token 1000"] --> S
    T3["token 100000"] --> S
    S --> SZ["626.25 MB<br/>always"]
""", {"T1": "blue", "T2": "amber", "T3": "orange", "S": "purple", "SZ": "green"})

add("kda-nine-steps", """
flowchart LR
    P["1. project<br/>q k v beta z"] --> C["2. ShortConv<br/>+ SiLU"]
    C --> N["3. L2Norm<br/>q and k only"]
    N --> B["4/5. beta<br/>and decay"]
    B --> R["6. recurrence<br/>per head"]
    R --> G["7/8/9. norm,<br/>gate, project"]
    R -.->|"state carried"| S["one matrix<br/>per head"]
""", {"P": "gray", "C": "blue", "N": "amber", "B": "orange", "R": "purple", "G": "green", "S": "teal"})

add("decoder-layer", """
flowchart LR
    H["h in"] --> A1["aggregate over<br/>earlier blocks"]
    A1 --> ATT["attention<br/>KDA or MLA"]
    ATT --> A2["aggregate again<br/>no guard"]
    A2 --> M["MoE<br/>or dense FFN"]
    M --> HO["h out"]
    A1 -.->|"every 12 layers"| SNAP["snapshot<br/>and clear"]
""", {"H": "blue", "A1": "purple", "ATT": "amber", "A2": "purple", "M": "orange", "HO": "green", "SNAP": "teal"})

add("forward-pass", """
flowchart LR
    ID["token ids"] --> EM["embed rows"]
    EM --> LOOP["93 layers<br/>bind, run, prefetch next"]
    LOOP --> AGG["model aggregator<br/>over all snapshots"]
    AGG --> NR["final RMSNorm<br/>last position only"]
    NR --> LG["lm_head<br/>163,840 logits"]
""", {"ID": "gray", "EM": "blue", "LOOP": "amber", "AGG": "purple", "NR": "orange", "LG": "green"})

add("mla-latent", """
flowchart LR
    X["token"] --> C["latent<br/>512 + 64 = 576"]
    C --> CACHE["this is what we cache<br/>2.37 MB per position"]
    C --> K["rebuild k and v<br/>on use"]
    K --> A["96 heads<br/>width 192"]
    A --> G["gate then o_proj"]
""", {"X": "blue", "C": "purple", "CACHE": "green", "K": "amber", "A": "orange", "G": "teal"})

add("attn-res", """
flowchart LR
    E["embedding<br/>always source 0"] --> SM{"softmax over<br/>every source"}
    B1["block 1<br/>output"] --> SM
    B2["block 2<br/>output"] --> SM
    NOW["running prefix<br/>this block"] --> SM
    SM --> H["new h<br/>replaces the old"]
    NOW -.->|"every 12 layers:<br/>snapshot and clear"| B2
""", {"E": "gray", "B1": "blue", "B2": "amber", "NOW": "orange", "SM": "purple", "H": "green"})

add("moe-dispatch", """
flowchart LR
    H["h 7168"] --> R["router<br/>sigmoid + bias"]
    R --> DN["down to latent<br/>3584"]
    DN --> E16["16 experts<br/>SiTU-GLU"]
    E16 --> NRM["RMSNorm the<br/>AGGREGATE"]
    NRM --> UP["up to 7168"]
    H -.->|"unweighted"| SH["2 shared experts"]
    SH --> UP
""", {"H": "blue", "R": "purple", "DN": "amber", "E16": "orange", "NRM": "pink", "UP": "green", "SH": "teal"})

# ---------------------------------------------------------------- trunk + cache

add("pack-trunk", """
flowchart LR
    S["shard N"] --> Q{"layer bytes<br/>contiguous?"}
    Q -->|"no"| STOP["refuse<br/>would copy experts"]
    Q -->|"yes"| CP["one range copy<br/>pad to 4096"]
    CP --> TB["trunk.bin<br/>108.81 GB, 93 runs"]
""", {"S": "gray", "Q": "purple", "STOP": "red", "CP": "blue", "TB": "green"})

add("trunk-stream", """
flowchart LR
    TB["trunk.bin"] --> Q{"layer L<br/>pinned?"}
    Q -->|"yes"| MEM["already in RAM<br/>0 bytes read"]
    Q -->|"no"| RING["ring slot<br/>one pread"]
    MEM --> BIND["fix pointers<br/>widen norms"]
    RING --> BIND
    BIND --> RUN["run layer L"]
""", {"TB": "gray", "Q": "purple", "MEM": "green", "RING": "red", "BIND": "amber", "RUN": "teal"})

add("cache-3phase", """
flowchart LR
    R["16 expert ids<br/>from the router"] --> P1["phase 1<br/>reserve slots serially"]
    P1 --> P2["phase 2<br/>pread in disk order"]
    P2 --> P3["phase 3<br/>publish what arrived"]
    P3 --> USE["matmul straight<br/>from the nibbles"]
    P2 -.->|"queue depth 16,<br/>not 1"| DEV["the device<br/>stays busy"]
""", {"R": "blue", "P1": "amber", "P2": "orange", "P3": "green", "USE": "teal", "DEV": "gray"})

add("cache-slot-states", """
flowchart LR
    E["EMPTY"] --> I["INFLIGHT<br/>reserved, not readable"]
    I --> RDY["ready<br/>holds expert e"]
    RDY -->|"evicted"| E
    I -.->|"read failed"| E
    RDY --> HIT["hit: reuse"]
""", {"E": "gray", "I": "amber", "RDY": "green", "HIT": "teal"})

add("trace-replay", """
flowchart LR
    RUN["one real run"] --> TR["expert_trace.bin<br/>100,096 requests"]
    TR --> SIM["replay at every<br/>cache size"]
    SIM --> LRU["LRU: what we do"]
    SIM --> BEL["Belady: the ceiling"]
    LRU --> GAP["25.5 points apart<br/>at the same RAM"]
    BEL --> GAP
""", {"RUN": "gray", "TR": "blue", "SIM": "purple", "LRU": "amber", "BEL": "green", "GAP": "red"})

# ---------------------------------------------------------------- validation

add("oracle-ladder", """
flowchart LR
    OP["op fixtures<br/>22 kernels, no weights"] --> TOY["13-layer oracle<br/>3 gates, exact ids"]
    TOY --> REAL["93 real layers<br/>stages A, B and C"]
    REAL --> LG["163,840 logits<br/>against torch"]
    LG --> V["max diff<br/>7.9e-06"]
    TOY -.->|"a toy model,<br/>not the 2.8T one"| WARN["proves the code,<br/>not the checkpoint"]
""", {"OP": "gray", "TOY": "blue", "REAL": "amber", "LG": "orange", "V": "green", "WARN": "red"})

add("parity-flow", """
flowchart LR
    IDS["same 5 ids"] --> C["our C engine<br/>169.73 s"]
    IDS --> T["torch reference<br/>3608.5 s"]
    C --> CMP["compare all<br/>163,840 logits"]
    T --> CMP
    CMP --> R["argmax 2494 both<br/>correlation 1.000000000"]
""", {"IDS": "gray", "C": "blue", "T": "amber", "CMP": "purple", "R": "green"})

add("verification-ladder", """
flowchart LR
    OPS["op fixtures<br/>tolerances from MANIFEST"] --> EX["exact equality<br/>bf16 = f32, all 3 paths"]
    EX --> OR["tiny-model oracle<br/>same weights as the C build"]
    OR --> G1["GATE 1 teacher forcing<br/>32/32 and 20/20"]
    OR --> G2["GATE 2 greedy decode<br/>20/20"]
    OR --> G3["GATE 3 incremental<br/>KV + KDA state, 20/20"]
    G1 --> CMP["final-position logits<br/>1024 bytes, byte for byte<br/>equal to the C binary"]
    G2 --> CMP
    G3 --> CMP
""", {"OPS": "gray", "EX": "blue", "OR": "purple", "G1": "amber", "G2": "orange", "G3": "teal", "CMP": "green"})

add("first-token", """
flowchart LR
    P["prompt<br/>The capital of France is"] --> E["tokenize<br/>5 ids"]
    E --> F["93 layers<br/>69 KDA + 24 MLA"]
    F --> L["163,840 logits<br/>one per token"]
    L --> A["argmax<br/>17374"]
    A --> D["decodes to<br/>Paris"]
    L -.->|"greedy only,<br/>no sampling"| X["the other 163,839<br/>are discarded"]
""", {"P": "gray", "E": "blue", "F": "amber", "L": "orange", "A": "purple", "D": "green", "X": "teal"})

add("gen-loop", """
flowchart LR
    P["prompt"] --> PF["step 0: prefill<br/>53 s, 89 GB read"]
    PF --> ST["step n: one token<br/>8.9 s, 25.83 GB"]
    ST --> TOK["append id"]
    TOK --> ST
    ST --> TXT["decode at the end<br/>never mid-token"]
""", {"P": "gray", "PF": "red", "ST": "blue", "TOK": "amber", "TXT": "green"})

# ---------------------------------------------------------------- measurement

add("ladder-harness", """
flowchart LR
    B["budget N GB"] --> CG["systemd-run<br/>MemoryMax=NG"]
    CG --> SW["MemorySwapMax=0"]
    SW --> RUN["8 tokens"]
    RUN --> CHK{"same ids as<br/>every other rung?"}
    CHK -->|"yes"| REC["record s/token"]
    CHK -->|"no"| BUG["that is a bug"]
""", {"B": "gray", "CG": "blue", "SW": "amber", "RUN": "orange", "CHK": "purple", "REC": "green", "BUG": "red"})

add("metric-bug", """
flowchart LR
    REQ["1472 requests"] --> NAIVE["naive hit<br/>100.0%"]
    REQ --> TRUE["TRUE resident<br/>0.00% always"]
    REQ --> DER["1 - evictions/requests"]
    NAIVE --> BAD["counts disk reads<br/>as hits"]
    TRUE --> BAD2["counter never reset"]
    DER --> GOOD["agrees with the bytes"]
""", {"REQ": "gray", "NAIVE": "red", "TRUE": "red", "DER": "blue", "BAD": "orange", "BAD2": "orange", "GOOD": "green"})

add("split-rule", """
flowchart LR
    GB["one spare GB"] --> Q{"trunk or<br/>expert cache?"}
    Q -->|"trunk"| T["pins a layer<br/>-1.17 GB per token"]
    Q -->|"cache"| C["below 36 GB arena<br/>removes nothing"]
    T --> FAST["16.80 s/token"]
    C --> SLOW["28.38 s/token"]
""", {"GB": "gray", "Q": "purple", "T": "green", "C": "red", "FAST": "teal", "SLOW": "orange"})

add("noise-floor", """
flowchart LR
    SAME["same binary<br/>same minute"] --> R1["14.78 s/tok"]
    SAME --> R2["14.67 s/tok"]
    SAME --> R3["20.14 s/tok"]
    R3 --> SP["spread 33.1%"]
    SP --> RULE["under 33% is<br/>not an effect"]
""", {"SAME": "gray", "R1": "blue", "R2": "blue", "R3": "red", "SP": "amber", "RULE": "green"})

add("hosts-confound", """
flowchart LR
    A["run A<br/>31.71 s/token"] --> DA["fast NVMe<br/>3,684 MB/s"]
    B["run B<br/>70.62 s/token"] --> DB["slow volume<br/>733 MB/s"]
    DA --> Q{"is the gap<br/>the code?"}
    DB --> Q
    Q -->|"no"| DISK["it is the disk"]
""", {"A": "blue", "B": "red", "DA": "green", "DB": "orange", "Q": "purple", "DISK": "amber"})

add("recap", """
flowchart LR
    HF["1.56 TB checkpoint<br/>96 shards"] --> PK["pack the trunk<br/>108.81 GB"]
    PK --> ENG["./target/release/k3<br/>a 1.16 MB binary"]
    ENG --> MEM["8.24 GB<br/>peak RSS"]
    MEM --> OUT["Paris.<br/>the right answer"]
    ENG -.->|"1.45 TB of experts<br/>stay on disk"| D["streamed,<br/>never resident"]
""", {"HF": "gray", "PK": "blue", "ENG": "amber", "MEM": "purple", "OUT": "green", "D": "teal"})


here = os.path.dirname(os.path.abspath(__file__))
for name, (code, fills, containers) in DIAGRAMS.items():
    content = code + "\n" + render_style(fills, containers) + "\n"
    with open(os.path.join(here, name + ".mmd"), "w", encoding="utf-8") as fh:
        fh.write(content)
print(f"wrote {len(DIAGRAMS)} .mmd files")
