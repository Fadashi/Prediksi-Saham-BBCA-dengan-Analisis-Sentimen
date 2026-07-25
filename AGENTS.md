# Rules and Guidelines for Agent Execution

## Code Conventions & Architecture
1. **Modular Code**: All core logic must reside under `src/`. Keep functions modular, documented in Indonesian docstrings, and type-annotated where possible.
2. **No Hardcoded Paths**: Load all parameters and file paths dynamically via `src/config.py` reading `config.yaml`.
3. **Reproducibility**: Global seed (random, numpy, tensorflow) must be initialized using `set_seed()` in `src/config.py`.
4. **No Data Leakage**: Scalers must only be fitted on the training split (`fit_transform` on train, `transform` on val/test). Chronological sequence splitting must be maintained without look-ahead bias.
5. **Quality & Testing**: Write unit tests for preprocessing, technical indicators, and dataset assembly in `tests/`.

## Workflow & Safety
- Secrets (e.g. `STOCKBIT_BEARER_TOKEN`) are loaded from `.env` and must never be committed.
- Large datasets (`data/raw/`, `data/interim/`, `data/processed/`) and binary model artifacts (`models/*.keras`) should be ignored by Git.
