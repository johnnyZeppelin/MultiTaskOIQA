# Merge patch for full OIQA project

This patch adds **single-image inference** to the full project.

## Files to copy into the full project

- `src/oiqa_bpr_vmamba/cli/infer_single_image.py`
- `configs/infer_default.yaml`
- `tests/test_infer_single_image.py`

## Required entry-point update

Add this to your `pyproject.toml` under `[project.scripts]`:

```toml
oiqa-infer-single-image = "oiqa_bpr_vmamba.cli.infer_single_image:main"
```

## What this CLI expects

Inputs:
- one distorted ERP image
- one restored ERP image
- 20 distorted viewport images
- 20 restored viewport images
- one trained checkpoint

It will:
- auto-synthesize degraded viewport pseudo-references
- build the model batch
- load the checkpoint
- run forward inference
- print and save the quality score

## Example command

```bash
oiqa-infer-single-image \
  --config configs/infer_default.yaml \
  --checkpoint /path/to/best.pt \
  --image /path/to/326.png \
  --restored-image /path/to/326_r.png \
  --viewport-root /path/to/viewports/326 \
  --restored-viewport-root /path/to/viewports_restored/326
```

## Assumptions

This patch assumes the full project already contains:
- `oiqa_bpr_vmamba.models.network.OIQABPRVMamba`
- `oiqa_bpr_vmamba.data.degradation`
- `oiqa_bpr_vmamba.utils.config`
- `oiqa_bpr_vmamba.utils.hashing`
- `oiqa_bpr_vmamba.utils.io`

Those modules existed in the earlier full project versions we discussed.
