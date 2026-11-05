# VisAnomBench Data

Download the dataset files from Hugging Face. Run this from the repository root (not from inside `VisAnomBench/`), so the files land in this folder:

```bash
hf download PLAN-Lab/VisAnomBench \
  --repo-type dataset \
  --include "VisAnomBench_train.json" \
  --include "VisAnomBench_test.json" \
  --include "VisAnomBench_plot_train/*" \
  --include "VisAnomBench_plot_test/*" \
  --include "VisAnomBench_train_ts/*" \
  --include "VisAnomBench_test_ts/*" \
  --local-dir VisAnomBench
```

After download, this folder should contain:

```text
VisAnomBench_train.json
VisAnomBench_test.json
VisAnomBench_plot_train/
VisAnomBench_plot_test/
VisAnomBench_train_ts/
VisAnomBench_test_ts/
```
