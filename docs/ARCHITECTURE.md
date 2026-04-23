# Architecture

```mermaid
flowchart LR
    A["Raw voice recording"] --> B["prepare-reference"]
    B --> C["24 kHz mono WAV"]
    D["Input text or .txt file"] --> E["normalize + chunk"]
    C --> F["XTTS synthesis"]
    E --> F
    G["Local XTTS model"] --> F
    F --> H["WAV chunks"]
    H --> I["concat with pauses"]
    I --> J["Final narrated audio"]
```

## Optional Fine-Tuning Data Flow

```mermaid
flowchart LR
    A["Consented speech recordings"] --> B["Manual/ASR transcripts"]
    B --> C["manifest.csv"]
    A --> C
    C --> D["build-dataset"]
    D --> E["LJSpeech-style dataset"]
    E --> F["dataset-stats"]
    E --> G["XTTS fine-tuning tools"]
    G --> H["Fine-tuned checkpoint"]
    H --> I["synthesize"]
```

## Pipeline

1. `prepare-reference` loads the source recording with `librosa`, resamples to 24 kHz, converts to mono, optionally trims the useful segment, and writes PCM WAV.
2. `synthesize` reads inline text or a text file, normalizes whitespace and punctuation, then splits long prose into smaller chunks.
3. `XttsVoiceCloner` loads XTTS either from a local model directory or the default Coqui model name.
4. Each text chunk is synthesized with the same `speaker_wav`.
5. Chunk WAV files are concatenated with configurable paragraph and sentence pauses.

## Why Chunking Matters

Long text synthesis is more stable when split into smaller pieces. It reduces hallucinated repetitions, gives better pause control, and makes failed chunks easier to rerun.
