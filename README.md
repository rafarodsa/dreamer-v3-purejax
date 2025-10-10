# Single(-ish) file reimplementation of DreamerV3 – Pure JAX (Readable, Minimal)

A **from-scratch, readable DreamerV3** implementation in **pure JAX** for easy reproduction and adaptation.


- **Pure JAX**: no heavy frameworks; modules live in `jaxmodels_nnx.py` and simple utilities in `utils/`.
- **Educational structure**: world model + actor + critic kept separate; training loop is easy to trace in `dreamer.py`.
- **Gymnax-ready**: light wrappers for classic control / pixel tasks via `gymnax_wrappers.py`.
- **Config-first**: tweak hyperparams in `dreamer.yaml` instead of chasing constants across files.
- **Small & hackable**: intended as a starter codebase for students and practitioners to learn or extend Dreamer.

---

## Getting started

### Environment

Install JAX appropriate for your machine (CPU/CUDA) following the JAX docs, then install Python deps:

```bash
# Create env (optional)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```


## Example run 

```
python dreamer.py --config dreamer.yaml --env CartPole-v1
```

## Caveats / TODOs

- Currently it only works with [Gymnax](https://github.com/RobertTLange/gymnax) environments;
- DreamerWrapper was added to remove the autoreset feature of Gymnax.
- The main script assumes discrete actions. TODO: add continuous action policies.

## References

- [DreamerV3 paper](https://arxiv.org/abs/2301.04104)
- [DreamerV3 Official Implementation](https://github.com/danijar/dreamerv3) 

## Citation

```
@article{hafner2025mastering,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

Citing this implementation
```
@software{dreamer_v3_purejax,
  author       = {Rafael Rodriguez-Sanchez},
  title        = {DreamerV3 — Pure JAX (Readable, Minimal)},
  year         = {2025},
  url          = {https://github.com/rafarodsa/dreamer-v3-purejax}
}
```