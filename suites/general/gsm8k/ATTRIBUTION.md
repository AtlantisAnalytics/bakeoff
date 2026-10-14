# Attribution: GSM8K slice

- Dataset: GSM8K (grade-school math word problems), test split.
- Authors: Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
  Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse and John
  Schulman (OpenAI).
- Paper: Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems",
  https://arxiv.org/abs/2110.14168
- License: MIT. License text: https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE
- Source file (pinned): https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl
- Source SHA-256: 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14

This slice is unmodified apart from sampling (100 of 1319 eligible rows, seeded, after
dropping 0 rows that failed the scrub scan) and the field mapping: `question` becomes
`prompt`, and `expected` is the number after `####` with thousands commas removed.
