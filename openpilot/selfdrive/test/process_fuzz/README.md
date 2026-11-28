# process fuzz

Runs each process's real `main()` through process replay on synthetic messages covering its whole input space, and asserts
properties that must hold for every input.

## Running

```sh
python -m openpilot.selfdrive.test.process_fuzz.fuzz test                          # CI: all processes (or name some)
python -m openpilot.selfdrive.test.process_fuzz.fuzz run plannerd --budget 60      # fresh examples, properties only
python -m openpilot.selfdrive.test.process_fuzz.fuzz update plannerd --budget 600  # add corpus genomes for new coverage
python -m openpilot.selfdrive.test.process_fuzz.fuzz repro <genome.json>           # rerun a failure
```

`FUZZ_SEED` seeds CI's fresh examples, `PROCESS_FUZZ_WORKERS` sets the worker count (default: usable CPUs),
`PROCESS_FUZZ_OUT` is where failing genomes go (default `/tmp/process_fuzz`), `PROCESS_FUZZ_DEEP=1` also fails on `KNOWN` bugs.

## Adding a process

1. Write `procs/<name>.py` with `SPEC = H.Spec(proc=..., scope=..., ledger=LEDGER, checks=[...])`, starting from a similar
   process. The name must be a process replay `CONFIGS` entry.
2. `fuzz.py update <name>` seeds the corpus and writes `procs/<name>_reads.json`.
3. Give every read the generator can't fuzz freely a `LEDGER` entry (`test_ledger` lists what's missing).
4. Explain every unreached branch it prints in `UNREACHED = {reason: [ids or '*' patterns]}`; fix the generator for gaps.
5. Triage every failure: an upstream guarantee becomes a `LEDGER` entry plus a property on the producer; otherwise it's a
   bug, so fix it or file it in `KNOWN` with its genome.
