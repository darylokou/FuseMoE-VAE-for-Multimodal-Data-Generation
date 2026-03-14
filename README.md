# FuseMoE-VAE-for-Multimodal-Data-Generation
Proposed architecture that aim to generate precise clinical data from multimodal inputs exploiting the work of FuseMoE for a more precise ...

```
fusemoe_gen/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/
│  ├─ data/
│  │  └─ toy_ts_tab.yaml
│  ├─ model/
│  │  ├─ moe_vae.yaml
│  │  └─ baseline_concat_vae.yaml
│  └─ train/
│     ├─ debug.yaml
│     └─ default.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ notebooks/
├─ scripts/
│  ├─ run_toy.sh
│  ├─ run_missing_modality.sh
├─ src/
│  └─ fusemoe_gen/
│     ├─ __init__.py
│     ├─ data/
│     │  ├─ datasets.py
│     │  ├─ collate.py
│     │  ├─ preprocess.py
│     │  └─ synthetic.py
│     ├─ models/
│     │  ├─ encoders/
│     │  │  ├─ base.py
│     │  │  ├─ ts_irregular.py
│     │  │  └─ tabular.py
│     │  ├─ fusion/
│     │  │  ├─ sparse_moe.py
│     │  │  ├─ hierarchical_moe.py
│     │  │  ├─ transformer_cross.py
│     │  │  └─ router_utils.py
│     │  ├─ latent/
│     │  │  ├─ posterior.py
│     │  │  └─ prior.py
│     │  ├─ decoders/
│     │  │  ├─ ts_decoder.py
│     │  │  └─ tabular_decoder.py
│     │  ├─ multimodal_vae.py
│     │  └─ baselines.py
│     ├─ losses/
│     │  ├─ reconstruction.py
│     │  ├─ kl.py
│     │  ├─ balance.py
│     │  └─ total.py
│     ├─ training/
│     │  ├─ engine.py
│     │  ├─ evaluator.py
│     │  ├─ callbacks.py
│     │  └─ utils.py
│     ├─ metrics/
│     │  ├─ generation.py
│     │  ├─ utility.py
│     │  └─ missingness.py
│     └─ utils/
│        ├─ seed.py
│        ├─ io.py
│        └─ logging.py
└─ tests/
   ├─ test_router.py
   ├─ test_shapes.py
   └─ test_forward.py
```
