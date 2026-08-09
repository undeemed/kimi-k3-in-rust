# Kimi-K3 in Rust - Flow

## 1. What this program even is

```mermaid
flowchart TB
    subgraph DISK["Hard drive: 1,560 GB model file - the brain"]
        MAIN["Main weights<br/>109 GB, used every step"]
        EXPERTS["896 experts per layer<br/>1,450 GB, only 16 needed per step"]
    end

    subgraph MEMORY["Your RAM: as little as 8 GB"]
        WORK["Small working area"]
    end

    MAIN -->|"Read in a fixed loop<br/>like a conveyor belt"| WORK
    EXPERTS -->|"Fetch only the 16 needed<br/>keep recent ones on a shelf"| WORK
    WORK --> TOKEN["Next word of text"]
```

Trick: never hold the whole brain in RAM.
More RAM makes it faster, never different.
The same prompt produces the same answer on an 8 GB laptop and a 224 GB server, byte for byte.
That promise is the whole project.

## 2. The parts (C file to Rust module, plain names)

```mermaid
flowchart TB
    subgraph MATH["Math side"]
        MAIN["Front door<br/>(main.rs) the k3 command"]
        OPS["Math core<br/>(ops/) all arithmetic"]
        TOK["Word chopper<br/>(tok/) text to numbers"]
        CFG["Settings reader<br/>(cfg.rs) refuses to guess"]
        ST["File index<br/>(st.rs) where every piece lives"]
        TRUNK["Conveyor belt<br/>(trunk.rs) main weights cycle through RAM"]
        CACHE["Hot shelf<br/>(cache.rs) recently used experts"]
        LOAD["Expert fetcher<br/>(load.rs) one read per expert"]
        BIND["Assembler<br/>(bind.rs) points math at the right bytes"]

        MAIN --> OPS
        MAIN --> TOK
        MAIN --> CFG
        MAIN --> ST
        MAIN --> TRUNK
        ST --> LOAD
        LOAD --> CACHE
        LOAD --> BIND
        TRUNK --> BIND
        CACHE --> BIND
        BIND --> OPS
    end
```

## 3. One word generated, start to finish

```mermaid
sequenceDiagram
    actor U as You
    participant E as Engine
    participant D as Disk

    U->>E: Prompt text
    E->>E: Chop text into numbers

    loop 93 layers in fixed order
        D-->>E: Deliver layer weights while the next layer preloads
        E->>E: Run attention math
        E->>E: Pick 16 of 896 experts

        alt Expert is on the hot shelf
            E->>E: Use cached expert
        else Expert is not on the shelf
            E->>D: Request expert
            D-->>E: Read 17.5 MB, shelve it, evict the oldest
        end
    end

    E-->>U: Next word
    Note over U,E: Repeat for the next word
```

## 4. Build order (the 8 steps in the plan)

```mermaid
flowchart TB
    S1["1. Skeleton and math core<br/>Translate the arithmetic exactly"]
    S2["2. Big math blocks, settings reader,<br/>and mini-model exam"]
    S3["3. File index and expert fetcher"]
    S4["4. Hot shelf"]
    S5["5. Conveyor belt and assembler"]
    S6["6. Word chopper"]
    S7["7. Front door - CLI"]
    S8["8. Speed measurements"]

    S1 --> S2
    S1 --> S8
    S2 --> S3
    S2 --> S4
    S2 --> S5
    S2 --> S6
    S3 --> S7
    S4 --> S7
    S5 --> S7
    S6 --> S7
```

Steps 3 through 6 are independent and can happen in any order.
Each step ships with its own tests passing.

## 5. Why exactly the same answer is hard, and the rule that fixes it

```mermaid
flowchart LR
    C["C code adds numbers<br/>in a specific order"]
    R["Rust copies that order<br/>line by line, with no improvements"]
    SAME["Same bits out"]

    REORDER["Reorder the additions<br/>compilers love to do this"]
    DRIFT["Tiny drift<br/>different word 500 tokens later"]

    C --> R --> SAME
    REORDER -.-> DRIFT
```

Floating-point math means `(a + b) + c` is not always equal to `a + (b + c)` in the last digit.
The C code pins the order.
The Rust translation table pins the identical order, line by line.
Fast SIMD paths must preserve that order too.

## 6. How we know it worked (proof ladder)

```mermaid
flowchart TB
    OPS["Each math piece compared with<br/>15 recorded question and answer files"]
    G1["G1"]
    G2["G2"]
    G3["G3"]

    OPS --> G1 --> G2 --> G3
```
