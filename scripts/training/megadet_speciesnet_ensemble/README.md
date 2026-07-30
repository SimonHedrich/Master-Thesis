# `megadet_speciesnet_ensemble` — teacher-baseline outputs

This directory holds **outputs only** — a MegaDetector v5 + SpeciesNet ensemble has no
code of its own here. The code that produces and scores these artifacts lives in
`scripts/training/yolov5s/eval_suite/` (`predict_ensemble.py`, `run_evaluation.py`) and
is shared with the `yolov5s`/`yolo26n` detector pipelines; see that package's
`README.md` for the actual commands.

## Layout

```
model_exports/
├── pretrained/              off-the-shelf MegaDetector v5 + SpeciesNet, no fine-tuning
│   ├── predictions_real.json
│   ├── predictions_synth.json
│   └── eval/
└── finetuned-<run_name>/    ensemble with a scripts/training/teacher_finetune
    │                        checkpoint substituted for SpeciesNet's classifier;
    │                        <run_name> matches the teacher_finetune run directory
    │                        the checkpoint came from (e.g. teacher-finetune-20260815-101500)
    ├── predictions_real.json
    ├── predictions_synth.json
    └── eval/
```

`predict_ensemble.py` picks the right subdirectory automatically from whether
`--checkpoint` is passed, so the pretrained baseline and any fine-tuned variant can
never collide or overwrite each other by omission — only an explicit `--output-dir`
override can do that.

Fine-tuning is currently scoped to the SpeciesNet classifier head only (MegaDetector
is unmodified in every variant here); see `scripts/training/teacher_finetune/README.md`
for how a fine-tuned checkpoint is produced.
