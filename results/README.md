# Results

Published bake-off results: one directory per Monday, plus the files the site and API serve.
`experiments/` holds runs from explorations outside the weekly rules, such as the multi-GPU
runs behind `/topology/`; they never change a pick.
From the week of 2026-11-30, each bundle keeps its small files here and its larger files (quality
outputs and samples) at `https://data.bakeoff.si/sha256/<aa>/<hash>`, named by their SHA-256 hash;
`layout.toml` says which files stay in git for which weeks, and `bakeoff store fetch` downloads and
checks them. `withdrawn.toml` lists any file taken down, with the date and the reason.

The data in this directory is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
reuse it freely, with credit to bakeoff.si. The code in the rest of the repository is Apache-2.0.
