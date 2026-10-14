# Attribution: IFEval slice

- Dataset: IFEval (Instruction-Following Eval), `ifeval_input_data.jsonl`.
- Authors: Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan,
  Denny Zhou and Le Hou (Google).
- Paper: Zhou et al. 2023, "Instruction-Following Evaluation for Large Language Models",
  https://arxiv.org/abs/2311.07911
- License: Apache-2.0 (dataset card `license: apache-2.0`). Card: https://huggingface.co/datasets/google/IFEval/blob/966cd89545d6b6acfd7638bc708b98261ca58e84/README.md
- Source file (pinned): https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl
- Source SHA-256: 6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2

This slice is unmodified apart from sampling (100 of 308 eligible rows, seeded, where eligible
means every instruction is implemented by our checker subset, after dropping 0 rows that
failed the scrub scan) and the field mapping: `prompt` is kept, `expected` holds
`instruction_id_list` and `kwargs` with null-valued keys dropped.
