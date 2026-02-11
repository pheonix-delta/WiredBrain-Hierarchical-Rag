# Changelog

All notable changes to the WiredBrain project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Research Proofs**: Added `trm_proofs/` directory containing JSON logs and PNG evidence of the 693K node audit.
    - `trm_proofs/trm_wiredbrain_proof.json`: Full reasoning trace for the "Deep Audit".
    - `trm_proofs/math_trm_wiredbrain_proof.json`: Mathematical derivation log.
    - `trm_proofs/fig10_honesty_moat.png`: Visual proof of hallucination mitigation.
    - `trm_proofs/fig11_resilience_moat.png`: Visual proof of noise filtering.
- **Verification Scripts**: Added `scripts/wiredbrain_proof_of_worth.py` to allow users to verify the system performance locally.
- **Sample Data**: Included `data/samples/sample_data.json` for reproducibility.

### Changed
- **License**: Migrated project license from **MIT** to **GNU AGPLv3**.
    - This change ensures that any future modifications and network deployments remain open source.
    - Added "Legacy Note" in README to clarify status for previous forks.
- **Documentation**: Updated `README.md` to reflect the new licensing model and emphasize the "Data Sovereignty" mission.
- **Git Hygiene**: Updated `.gitignore` to properly track research artifacts while excluding large datasets.

### Security
- **Sovereign Isolation**: Implemented stricter data handling protocols for sensitive graph discovery logic (moved to private development branch).
