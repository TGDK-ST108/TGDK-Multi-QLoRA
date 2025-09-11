# TGDK-QLoRA with Duo

🚀 **TGDK-QLoRA** is the world’s first **multi-model symbolic QLoRA variant**, extending the Hugging Face QLoRA technique into TGDK’s sovereign, license-bound ecosystem.

This release demonstrates how **BERT** (understanding) and **Mistral** (generation) can be fine-tuned together under **TGDK’s Duo arbitration layer**, with differentiated optimizers (**AdamW** and **Lion**) applied for maximum efficiency.

---

## ✨ Key Features
- **Multi-Model Training** → run **BERT + Mistral** in a single fine-tuning pipeline.  
- **Optimizer Differentiation** → AdamW for stability in BERT; Lion for fast convergence in Mistral.  
- **Duo Orchestration** → a TGDK interface that dynamically routes optimizer updates across models.  
- **Hybrid Licensing** → this package is released under the TGDK-BFE Research License (see [LICENSE](LICENSE.txt)).

---

## 🧩 Project Structure
```
├── qlora.py # Training wrapper with multi-model support
├── Duo.py # AdamW + Lion selection logic
├── duo_interface.py # Public Duo arbitration interface (no internals)
├── train_config.yaml # Example training configuration
├── README.md # You are here
└── examples/
├── run_mistral.sh # Fine-tune Mistral with Lion
└── run_bert.sh # Fine-tune BERT with AdamW
```

## ⚙️ Usage


AdamW → BERT (encoder stability, clause fidelity).

Lion → Mistral (generative speed, entropy-aware loops).

⚠️ Note: This repo exposes only the public Duo interface. Full Duo internals, VaultLedger hooks, and TGDK sealing remain proprietary.

## 📜 License
This release is provided under the TGDK-BFE Research License.

Free for research, academic, and non-commercial use.

Commercial deployment requires a TGDK license and VaultLedger binding.

📡 Credits
Developed by Sean Tichenor (TGDK LLC) with OliviaAI orchestration.
Part of the TGDK Sovereign AI Ecosystem.
