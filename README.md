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
## Duo Hybrid vs. Baseline (QLoRA, 7B)

| Metric | Baseline (AdamW/Lion std) | Duo Hybrid (Duo + Jade + MMT) |
|--------|----------------------------|-------------------------------|
| Train loss (final) | 4.06 | 4.06 |
| Eval loss (final)  | 4.47 | 4.47 |
| Train runtime      | 6021.5 s | **4262.1 s** |
| Train samples/sec  | 1.36 | **1.92** |
| Train steps/sec    | 0.042 | **0.059** |
| Eval runtime       | 265.15 s | **172.01 s** |
| Eval steps/sec     | 1.003 | **1.546** |
| Grad norm (peak → final) | ~15.6 → ~12.3 | ~13.9 → **~11.0** |
| LR schedule (end)  | linear (2e-5) | linear (2e-5) + Jade/Plateau escape |


Takeaways

~30% faster end-to-end training (1.36 → 1.92 samples/s) with Duo Hybrid.

Smoother optimization: lower final grad norm (~12.3 → ~11.0).

No loss regression: train/eval losses match baseline parity.

Eval pass faster (265s → 172s), same batch/seq settings.

Representative logs

Baseline (final):
train_loss=4.0628 | eval_loss=4.4685 | train_runtime=6021.5s | samples/s=1.36 | steps/s=0.042 | eval_steps/s=1.003

Duo Hybrid (final):
train_loss=4.0628 | eval_loss=4.4685 | train_runtime=4262.1s | samples/s=1.92 | steps/s=0.059 | eval_steps/s=1.546

Repro (Duo Hybrid)
```
python qlora.py \
  --optimizer duo \
  --duo-mode hybrid \
  --epochs 6 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --scheduler_type linear \
  --use-mmt \
  --use-jade \
  --plateau-patience 150 \
  --plateau-delta 5e-4
```

## 📜 License
This release is provided under the TGDK-BFE Research License.

Free for research, academic, and non-commercial use.

Commercial deployment requires a TGDK license and VaultLedger binding.

📡 Credits
Developed by Sean Tichenor (TGDK LLC) with OliviaAI orchestration.
Part of the TGDK Sovereign AI Ecosystem.
