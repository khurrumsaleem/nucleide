# utils — Synthetic merge/extract golden pairs for the MCPL utilities

Hand-framed particle lists authored for Nucleide (no license needed): input
files written with `write_mcpl` from explicit header/particle dicts, golden
outputs produced by the `merge_mcpl`/`extract_mcpl` facade over those inputs.
All particles are synthetic axis-vector records; no upstream files are read.

| File | Role |
| --- | --- |
| `merge_a.mcpl` / `merge_b.mcpl` | Merge inputs: single precision, user flags, two records each |
| `merge_ab.mcpl` | Golden `merge_mcpl([merge_a, merge_b])`: concat order, first-file header plus provenance comment |
| `merge_c.mcpl` | Double-precision merge input (one record) |
| `merge_ac.mcpl` | Golden `merge_mcpl([merge_a, merge_c])`: precision promoted to double |
| `extract_src.mcpl` | Extract input: double precision, four records, one header blob |
| `extract_range.mcpl` | Golden `extract_mcpl(extract_src, start=1, stop=3)` |
| `extract_pdg.mcpl` | Golden `extract_mcpl(extract_src, predicate=pdg==2112)` |
