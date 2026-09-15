# fispact — Synthetic FISPACT-II-style inventory authored for Nucleide (no license needed)

`inventory.fis` (3 cooling steps x 4 rows x 3 variables).

`clearance.out` (2 time steps x 7 rows): synthetic values in the real FISPACT-II
wide inventory grammar printed with the `HAZARDS` + `CLEAR` keywords; the layout
is cross-checked against the Apache-2.0 `fispact/workshops` reference outputs
(`2020/exercises/files/basic/compressxs/ref/inventorywithcompress_ref.out`) and
all numbers are hand-picked round values, not copied output.
