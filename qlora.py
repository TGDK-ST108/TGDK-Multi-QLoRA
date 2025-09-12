# qlora.py – TGDK Magic + Duo + MMT + JadeCodewright + Seals + Rituals
import os, sys, torch, hashlib, json, subprocess, datetime, glob, argparse, lzma, shutil
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig, get_scheduler
)
import lzma
import base64
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from torch.optim import AdamW
import sqlite3
from lion_pytorch import Lion
from scipy.spatial import Delaunay
import logging
import torch.nn as nn


# External TGDK geometry modules
from Mahadevi import Mahadevi
from Maharaga import Maharaga
from Trinity import Trinity
from Duo import make_duo_optimizer, Duo
import json
from device import load_model
from accelerate import Accelerator
from peft import get_peft_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainerCallback



# resolve OUT directory from env or default to ./out
OUT = os.environ.get("OUT", "./out")
os.makedirs(OUT, exist_ok=True)


# Disable mixed precision globally
accelerator = Accelerator(mixed_precision="no")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_config(path="models.config"):
    with open(path, "r") as f:
        cfg = json.load(f)
    default_id = cfg.get("default")
    iterations = {m["id"]: m for m in cfg.get("iterations", [])}
    if default_id not in iterations:
        raise ValueError(f"Default model {default_id} not in iterations")
    return iterations[default_id]

model_cfg = load_model_config()
BASE     = model_cfg["base_model"]
HF_TOKEN = os.environ.get("HF_TOKEN")



# ------------------------------------------------------------------
# CLI Arguments (parsed first so we can fall back if config is missing keys)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", type=str, default="adamw",
                    choices=["adamw", "lion", "adafactor", "duo"],
                    help="Optimizer choice")
parser.add_argument("--scheduler_type", type=str, default="linear",
                    choices=["linear", "cosine", "polynomial"],
                    help="LR scheduler type")
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--use-mmt", action="store_true")
parser.add_argument("--use-jade", action="store_true")
parser.add_argument(
    "--duo-mode",
    type=str,
    default="hybrid",
    choices=["symbolic", "amp", "hybrid"],
    help="Which Duo optimizer mode to use"
)
parser.add_argument("--plateau-patience", type=int, default=200,
                    help="Steps to wait before triggering escape mechanisms")
parser.add_argument("--plateau-delta", type=float, default=1e-3,
                    help="Minimum loss improvement threshold to reset patience")

cli_args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
# Config overlay (config > CLI > defaults)
# ------------------------------------------------------------------

# After you load model_cfg and CLI args:
BASE = model_cfg.get("base_model") or os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.1")
epochs    = model_cfg.get("epochs")    or cli_args.epochs
opt_choice = model_cfg.get("optimizer") or cli_args.optimizer
use_mmt   = model_cfg.get("mmt")       or cli_args.use_mmt
use_jade  = model_cfg.get("jade")      or cli_args.use_jade
outdir   = os.path.join(OUT, "olivia-12ob-dapt-lora")
offload_dir = os.path.join(outdir, "offload")


PRIVATE_KEY_PATH = os.environ.get("TGDK_PRIVATE_KEY", "tgdk_private.pem")
PUBLIC_KEY_PATH  = os.environ.get("TGDK_PUBLIC_KEY", "tgdk_public.pem")
model = load_model(BASE, HF_TOKEN, outdir, gpu_idx=0)
OUT = os.environ.get("OUT", ".")

use_cuda = torch.cuda.is_available()
supports_bf16 = use_cuda and torch.cuda.is_bf16_supported()

fp16_flag = use_cuda and not supports_bf16
bf16_flag = supports_bf16

model_cfg = load_model_config()
HF_TOKEN = os.environ.get("HF_TOKEN")




if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        fp16_flag = False
        bf16_flag = True
    else:
        fp16_flag = True
        bf16_flag = False
else:
    fp16_flag = False
    bf16_flag = False



# Accelerate dummy fallbacks (API moved across versions)
try:
    from accelerate.state import DummyOptim, DummyScheduler
except ImportError:
    try:
        from accelerate.utils.dataclasses import DummyOptim, DummyScheduler
    except ImportError:
        from compat_dummy import DummyOptim, DummyScheduler


bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def olivia_pipeline(text):
    # Step 1: BERT encodes the text
    bert_inputs = bert_tok(text, return_tensors="pt")
    bert_outputs = bert_model(**bert_inputs).last_hidden_state

    # Maybe take [CLS] embedding
    cls_embedding = bert_outputs[:, 0, :]

    # Step 2: Feed CLS embedding into Mistral prompt
    prompt = f"[BERT-CLS: {cls_embedding.tolist()}]\n{text}\nResponse:"
    mistral_inputs = mistral_tok(prompt, return_tensors="pt")
    gen = mistral_model.generate(**mistral_inputs, max_new_tokens=100)

    return mistral_tok.decode(gen[0], skip_special_tokens=True)

class PlateauEscapeCallback(TrainerCallback):
    def __init__(self, trainer, patience, min_delta):
        self.trainer = trainer
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_steps = 0
        self.jade_triggered = False
        self.cosine_triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        loss = logs["loss"]

        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        if self.bad_steps >= self.patience:
            if not self.cosine_triggered:
                from transformers import get_scheduler
                new_sched = get_scheduler(
                    "cosine",
                    optimizer=self.trainer.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=state.max_steps
                )
                self.trainer.lr_scheduler = new_sched
                print(f"⚡ [PlateauEscape] Switched LR scheduler → cosine (patience={self.patience})")
                self.cosine_triggered = True

            if hasattr(self.trainer, "jade_lex") and not self.jade_triggered:
                self.trainer.use_jade_reweighting = True
                print(f"⚡ [PlateauEscape] Jade reweighting enabled (min_delta={self.min_delta})")
                self.jade_triggered = True

            self.bad_steps = 0


class BertMistralFusion(nn.Module):
    def __init__(self, bert, mistral):
        super().__init__()
        self.bert = bert
        self.mistral = mistral
        self.proj = nn.Linear(
            bert.config.hidden_size + mistral.config.hidden_size,
            mistral.config.hidden_size
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        bert_out = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        mistral_out = self.mistral.model(input_ids, attention_mask=attention_mask).last_hidden_state
        fused = torch.cat([bert_out.unsqueeze(1).expand(-1, mistral_out.size(1), -1), mistral_out], dim=-1)
        fused = self.proj(fused)
        return self.mistral(inputs_embeds=fused, labels=labels)


class TGDKMemoryDB:
    def __init__(self, db_path="tgdk_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER,
            step INTEGER,
            loss REAL,
            eval_loss REAL,
            pillar TEXT,
            sliver TEXT,
            matrix BLOB,
            timestamp TEXT
        )
        """)
        self.conn.commit()

    def save_entry(self, epoch, step, loss, eval_loss, pillar, matrix, sliver):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO memory (epoch, step, loss, eval_loss, pillar, sliver, matrix, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (epoch, step, loss, eval_loss, pillar, sliver,
              matrix.tobytes(), datetime.datetime.now(datetime.timezone.utc).isoformat()))

        self.conn.commit()

    def recall(self, limit=5, query=None):
        cur = self.conn.cursor()
        if query is None:
            query = f"SELECT * FROM memory ORDER BY id DESC LIMIT {limit}"
        cur.execute(query)
        return cur.fetchall()

class PlateauEscapeCallback(TrainerCallback):
    def __init__(self, trainer, patience=200, min_delta=1e-3):
        self.trainer = trainer
        self.patience = patience       # steps to wait before escape
        self.min_delta = min_delta     # minimum improvement in loss
        self.best_loss = float("inf")
        self.bad_steps = 0
        self.jade_triggered = False
        self.cosine_triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        loss = logs["loss"]

        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        if self.bad_steps >= self.patience:
            # Switch LR scheduler to cosine if not already
            if not self.cosine_triggered:
                from transformers import get_scheduler
                new_sched = get_scheduler(
                    "cosine",
                    optimizer=self.trainer.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=state.max_steps
                )
                self.trainer.lr_scheduler = new_sched
                print("⚡ [PlateauEscape] Switched LR scheduler to cosine")
                self.cosine_triggered = True

            # Apply Jade reweighting dynamically
            if hasattr(self.trainer, "jade_lex") and not self.jade_triggered:
                self.trainer.use_jade_reweighting = True
                print("⚡ [PlateauEscape] Jade reweighting enabled")
                self.jade_triggered = True

            # Reset counter so it doesn’t trigger every log
            self.bad_steps = 0


bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

def run_pipeline(input_text: str, task: str = "generation"):
    if task in ("classification", "embedding"):
        return bert_model(**bert_tok(input_text, return_tensors="pt"))
    else:
        return mistral_model.generate(
            **mistral_tok(input_text, return_tensors="pt"),
            max_new_tokens=100
        )


class MMTController:
    """
    Multi-Modal TGDK Controller
    - volumetric_infinitizer: infinite volumetric expansion field
    - pyramid: directional pyramid routing
    - figure8flow: oscillatory 8-flow vector harmonics
    """

    def __init__(self, dim: int = 128, mahadevi=None, maharaga=None, trinity=None):
        self.mahadevi = mahadevi
        self.maharaga = maharaga
        self.trinity = trinity
        self.dim = dim
        self.state = {
            "volumetric": np.zeros(dim, dtype=float),
            "pyramid": np.zeros(dim, dtype=float),
            "figure8": np.zeros((2, dim), dtype=float),
        }
        logging.info(f"[MMTController] initialized with dim={dim}")

    def log_state(self, outdir, epoch):
        state = {
            "epoch": epoch,
            "mahadevi_vectors": self.mahadevi.vector_field,
            "maharaga_centroid": (
                self.maharaga.calculate_centroid().tolist()
                if self.maharaga.data_points else None
            ),
            "trinity_seq": float(
                np.mean(self.trinity.expand_data(np.random.rand(5)))
            )
        }
        path = os.path.join(outdir, f"mmt_state_epoch{epoch}.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        return path
        
    # --- core generators ---
    def volumetric_infinitizer(self, step: int = 1):
        vec = np.sin(np.linspace(0, np.pi * step, self.dim))
        self.state["volumetric"] = vec
        return vec

    def pyramid(self, height: int = 4):
        # Create a simple pyramid ramp vector
        ramp = np.linspace(-1, 1, self.dim)
        vec = np.abs(ramp) ** height
        self.state["pyramid"] = vec
        return vec

    def figure8flow(self, t: float = 0.0):
        # Parametric figure-8 (lemniscate) curves mapped into vector slots
        theta = np.linspace(0, 2 * np.pi, self.dim)
        x = np.sin(theta + t)
        y = np.sin(theta + t) * np.cos(theta + t)
        self.state["figure8"] = np.vstack([x, y])
        return x, y

    # --- stepper ---
    def step(self, step_id: int = 0):
        v = self.volumetric_infinitizer(step=step_id + 1)
        p = self.pyramid(height=4)
        f = self.figure8flow(t=step_id * 0.1)
        logging.info(f"[MMTController] step {step_id} updated state")
        return {"volumetric": v, "pyramid": p, "figure8": f}


# ------------------------------------------------------------------
# Initialize MMT + Jade systems
# ------------------------------------------------------------------
if cli_args.use_mmt:
    # --- Initialize Mahadevi (vector field) ---
    mahadevi = Mahadevi()
    mahadevi.set_vector_field([np.array([1, 0]), np.array([0, 1])])  # orthogonal seed vectors
    print("Vector field set successfully.")

    # --- Initialize Maharaga (centroid data) ---
    maharaga = Maharaga()
    maharaga.add_data_point([0.5, 0.5])  # starter centroid
    print("Data point [0.5, 0.5] added.")

    # --- Initialize Trinity (AQVP balance) ---
    trinity = Trinity(0.8, 1.2, 0.95)

    # --- Build MMT Controller with all three ---
    mmt_controller = MMTController(
        dim=1440,
        mahadevi=mahadevi,
        maharaga=maharaga,
        trinity=trinity
    )
    print("[MMT] Mahadevi/Maharaga/Trinity initialized and bound")

else:
    mmt_controller = None



# --------- Build optimizer (with mode from CLI) ---------
#duo_optim, duo_sched = make_duo_optimizer(
#   model,
#    mmt_controller=mmt_controller,
#    lr=cli_args.learning_rate,
#    weight_decay=cli_args.weight_decay,
#    mode=cli_args.duo_mode  # 👈 symbolic | amp | hybrid
#)
#use_optimizers = (duo_optim, duo_sched)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
WORK     = os.environ.get("WORK", ".")
# Collect training files
train_files = sorted(glob.glob(os.path.join(WORK, "packs", "train*.jsonl")))
val_files   = sorted(glob.glob(os.path.join(WORK, "packs", "val*.jsonl")))

# ---- Training set ----
if train_files:
    print(f"[INFO] Found training files: {train_files}")
    raw_train = load_dataset(
        "json",
        data_files={"train": train_files},
        split="train"
    )
else:
    raise FileNotFoundError("No train*.jsonl files found under packs/")
print(raw_train[0])

# ---- Validation set ----
if val_files:
    print(f"[INFO] Found validation files: {val_files}")
    raw_val = load_dataset(
        "json",
        data_files={"validation": val_files},
        split="validation"
    )
else:
    # fallback: take slice of training
    raw_val = raw_train.select(range(min(20, len(raw_train))))
    print("[WARN] No val*.jsonl found — using slice of train")


try:
    from accelerate.utils import DummyOptim
except ImportError:
    class DummyOptim(torch.optim.Optimizer):
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.param_groups = optimizer.param_groups
        def step(self, closure=None):
            return self.optimizer.step(closure)
        def zero_grad(self, set_to_none=False):
            return self.optimizer.zero_grad(set_to_none=set_to_none)
        def state_dict(self):
            return self.optimizer.state_dict()
        def load_state_dict(self, state_dict):
            return self.optimizer.load_state_dict(state_dict)


# ------------------------------------------------------------------
# TGDK Key Management
# ------------------------------------------------------------------
def ensure_keys(priv_path, pub_path):
    if os.path.exists(priv_path) and os.path.exists(pub_path):
        return
    print("🔑 TGDK Keys not found — generating new RSA keypair...")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(priv_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    public_key = private_key.public_key()
    with open(pub_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    print(f"✅ TGDK Keys generated → {priv_path}, {pub_path}")

ensure_keys(PRIVATE_KEY_PATH, PUBLIC_KEY_PATH)


class TGDKFoldExpansion:
    def __init__(self, r=16, alpha=32, dropout=0.05, targets=None):
        self.peft_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=targets or ["q_proj","k_proj","v_proj","o_proj"],
            bias="none", task_type="CAUSAL_LM"
        )
    def apply(self, trainer):
        trainer.add_callback(self._seal_callback)

    def _seal_callback(self, state, control, **kwargs):
        # Insert TGDK clause hooks or gradient folding rituals here
        pass

class EntropyFoldConfig:
    def __init__(self, precision="nf4", dtype="bfloat16"):
        self.cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=precision,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

# ------------------------------------------------------------------
# JadeCodewright Lexicon
# ------------------------------------------------------------------
class JadeCodewrightLexicon:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}

    def bind_metrics(self, metrics):
        jade = {}
        jade["Δ_entropy"] = "JADE-Σ" if metrics["loss"] < 1.0 else "JADE-Ω"
        jade["Δ_jovian"] = f"JX-{int(metrics['TGDK::JovianScalar']*100)}"
        jade["Δ_rigpa"]  = f"HUM-{int(metrics['TGDK::Rigpa']['rigpa_scalar']*100)}"
        return jade

    def emit_clause(self, jade, outdir, epoch):
        path = os.path.join(outdir, f"jade_lex_epoch{epoch}.txt")
        with open(path, "w") as f:
            for k, v in jade.items():
                f.write(f"{k} :: {v}\n")
        print(f"[JadeCodewright] Lexicon emitted → {path}")
        return path

    @staticmethod
    def jade_loss_reweight(loss, jade):
        weight = 1.0
        if jade["Δ_entropy"] == "JADE-Σ":  # low loss → reinforce
            weight *= 0.9
        else:  # high loss → encourage correction
            weight *= 1.1
        if "HUM" in jade["Δ_rigpa"]:
            weight *= 0.95
        return loss * weight



class TGDKMemoryDB:
    def __init__(self, db_path="tgdk_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER,
            step INTEGER,
            loss REAL,
            eval_loss REAL,
            pillar TEXT,
            sliver TEXT,
            matrix BLOB,
            timestamp TEXT
        )
        """)
        self.conn.commit()

    def save_entry(self, epoch, step, loss, eval_loss, pillar, matrix, sliver):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO memory (epoch, step, loss, eval_loss, pillar, sliver, matrix, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (epoch, step, loss, eval_loss, pillar, sliver,
              matrix.tobytes(), datetime.datetime.now(datetime.timezone.utc).isoformat()))

        self.conn.commit()

    def recall(self, query="SELECT * FROM memory ORDER BY id DESC LIMIT 5"):
        cur = self.conn.cursor()
        cur.execute(query)
        return cur.fetchall()


# ------------------------------------------------------------------
# Tokenizer + Dataset
# ------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, token=HF_TOKEN)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

def fmt(example):
    instruction = example.get("instruction", "")
    input_text  = example.get("input", "")
    output_text = example.get("output", "")

    if input_text:
        prompt = f"{instruction}\n\nInput:\n{input_text}\n\nResponse:"
    else:
        prompt = f"{instruction}\n\nResponse:"

    # Build the full sequence
    text = (prompt.strip() + " " + output_text).strip()

    return {
        "text": text,
        # Tokenizer will convert these into input_ids and labels
        "labels": text
    }




# Build text column first
ds_train = raw_train.map(fmt, remove_columns=raw_train.column_names)
ds_val   = raw_val.map(fmt, remove_columns=raw_val.column_names)
ds_train = ds_train.filter(lambda ex: len(ex["text"].strip()) > 0)
ds_val   = ds_val.filter(lambda ex: len(ex["text"].strip()) > 0)

memory_db = TGDKMemoryDB(os.path.join(outdir, "tgdk_memory.db"))

# Try recalling
try:
    recalls = memory_db.recall("SELECT * FROM memory ORDER BY id DESC LIMIT 144")
    if recalls:
        print(f"[TGDK-MEMORY] Warming start with {len(recalls)} fold matrices")
        warm_matrix = np.mean(
            [np.frombuffer(r[7], dtype=np.float64).reshape(3,3) for r in recalls],
            axis=0
        )
        # Optionally inject into adapters or optimizer states
except Exception as e:
    print("[TGDK-MEMORY] Recall failed:", e)
    recalls = []
# ------------------------------------------------------------------
# Model + LoRA
# ------------------------------------------------------------------

# Training-time memory savers
model.config.use_cache = False              # important for training
model.gradient_checkpointing_enable()       # big saver
try:
    model.enable_input_require_grads()
except Exception:
    pass

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      bias="none", task_type="CAUSAL_LM")

# ------------------------------------------------------------------
# Training Args
# ------------------------------------------------------------------
# Auto-detect precision based on hardware
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
fp16 = torch.cuda.is_available() and not bf16

train_args = TrainingArguments(
    output_dir=outdir,
    per_device_train_batch_size=4,     # scale with your GPU VRAM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,     # effective batch = 32
    learning_rate=cli_args.learning_rate,
    num_train_epochs=cli_args.epochs,  # e.g. 3–5
    fp16=False,
    bf16=False,
    fp16_full_eval=False,
    bf16_full_eval=False,
    logging_steps=25,
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=5,                # keep 5 checkpoints
    dataloader_num_workers=2,
    report_to=["tensorboard"],         # or ["wandb"] if you use wandb
    warmup_ratio=0.03,                 # ~3% of steps for warmup
)



# ------------------------------------------------------------------
# Initialize MMT + Jade systems
# ------------------------------------------------------------------
if cli_args.use_mmt:
    mahadevi = Mahadevi()
    mahadevi.set_vector_field([np.array([1, 0]), np.array([0, 1])])  # orthogonal seed vectors

    maharaga = Maharaga()
    maharaga.add_data_point([0.5, 0.5])  # centroid starter point

    trinity = Trinity(0.8, 1.2, 0.95)  # balanced initial AQVP

    mmt_controller = MMTController(
        dim=256,
        mahadevi=mahadevi,
        maharaga=maharaga,
        trinity=trinity
    )

    print("[MMT] Mahadevi/Maharaga/Trinity initialized and bound")
else:
    mmt_controller = None

if cli_args.use_jade:
    jade_vocab = {
        "Σ": "low entropy fold",
        "Ω": "high entropy fold",
        "JX": "jovian scalar expansion",
        "HUM": "rigpa seed binding"
    }
    jade_lex = JadeCodewrightLexicon(vocab=jade_vocab)
    print("[JadeCodewright] Lexicon initialized with symbolic vocab")
else:
    jade_lex = None

# ------------------------------------------------------------------
# Optimizer Factory
# ------------------------------------------------------------------
def make_optimizer_scheduler(model, cli_args, total_steps):
    if cli_args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay
        )

    elif cli_args.optimizer == "lion":
        optimizer = Lion(
            model.parameters(),
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay
        )

    elif cli_args.optimizer == "adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None
        )

    elif cli_args.optimizer == "duo":
        # --- Initialize TGDK vector systems ---
        mahadevi = Mahadevi()
        mahadevi.set_vector_field([np.array([1, 0]), np.array([0, 1])])

        maharaga = Maharaga()
        maharaga.add_data_point([0.5, 0.5])

        trinity = Trinity(0.8, 1.2, 0.95)

        # --- collect only trainable params (LoRA adapters, unfrozen layers, etc.) ---
        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        if not trainable_params:
            logging.warning("[Duo] No trainable parameters found — inserting dummy param")
            trainable_params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
        else:
            logging.info(f"[Duo] Found {len(trainable_params)} trainable params.")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    logging.debug(f"[Duo] Trainable → {n}: {tuple(p.shape)}")

        optimizer = Duo(
            trainable_params,
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay,
            mahadevi=mahadevi,
            maharaga=maharaga,
            trinity=trinity
        )

    else:
        raise ValueError(f"Unknown optimizer {cli_args.optimizer}")

    scheduler = get_scheduler(
        cli_args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cli_args.warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler


total_steps = len(ds_train) * cli_args.epochs
optimizers = make_optimizer_scheduler(model, cli_args, total_steps)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM"
)


# Wrap base model with LoRA adapters before optimizer creation
model = get_peft_model(model, peft_cfg)


# If you prefer standard optimizers instead, comment the two lines above and use:
# total_steps = len(ds_train) * cli_args.epochs
# use_optimizers = make_optimizer_scheduler(model, cli_args, total_steps)

sft_config = SFTConfig(
    output_dir=outdir,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=cli_args.learning_rate,
    num_train_epochs=cli_args.epochs,
    warmup_ratio=0.1,                # better than fixed warmup_steps
    weight_decay=cli_args.weight_decay,
    logging_steps=25,
    eval_strategy="steps",     # ✅ correct arg
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=5,
    dataloader_num_workers=2,
    report_to=["tensorboard"],
    max_seq_length=2048,
    dataset_text_field="text",       # ✅ here only
    packing=False,
    max_grad_norm=1.0,
    lr_scheduler_type=cli_args.scheduler_type,
    optim="adamw_torch",             # fallback
)


# --------- Build SFTTrainer (no Accelerate wrappers, no AMP) ---------
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    peft_config=peft_cfg,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    args=train_args,              # contains batch size, lr, steps, etc.
    dataset_text_field="text",    # ✅ expect pre-mapped "text"
    packing=False,
    optimizers=optimizers
)
trainer.jade_lex = jade_lex if cli_args.use_jade else None
trainer.use_jade_reweighting = False

trainer.add_callback(
    PlateauEscapeCallback(trainer,
                          patience=cli_args.plateau_patience,
                          min_delta=cli_args.plateau_delta)
)


# --------- Train once ---------
train_output = trainer.train()

current_epoch = int(trainer.state.epoch) if trainer.state.epoch is not None else 0
current_step  = int(trainer.state.global_step) if trainer.state.global_step is not None else 0

memory_db.save_entry(
    epoch=current_epoch,
    step=current_step,
    loss=last.get("loss", 0.0),
    eval_loss=last.get("eval_loss", 0.0),
    pillar=pillar_sig,
    matrix=matrix,
    sliver=sliver
)

try:
    recalls = memory_db.recall("SELECT * FROM memory ORDER BY id DESC LIMIT 10")
    if recalls:
        print(f"[TGDK-MEMORY] Warm-started with {len(recalls)} past fold entries")
except Exception as e:
    print("[TGDK-MEMORY] Recall failed:", e)

# ------------------------------------------------------------------
# TGDK Ritual Functions
# ------------------------------------------------------------------
def tgdk_log_metrics(step, loss, eval_loss=None, extra=None):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),

        "step": int(step),
        "loss": float(loss) if loss else None,
        "eval_loss": float(eval_loss) if eval_loss else None,
        "entropy_signature": hashlib.sha256(f"{step}-{loss}".encode()).hexdigest(),
        "culmex": "active"
    }
    if extra: entry.update(extra)
    with open(os.path.join(outdir, "tgdk_metrics.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("TGDK::Metrics", entry)
    return entry

def tgdk_pillar_wrap(outdir):
    model_bin = os.path.join(outdir, "pytorch_model.bin")
    sig = None
    if os.path.exists(model_bin):
        sig = hashlib.sha256(open(model_bin, "rb").read()).hexdigest()
        with open(os.path.join(outdir, "tgdk_pillar.json"), "w") as f:
            json.dump({"pillar_sig": sig, "culmex": "bound"}, f)
        with open(os.path.join(outdir, "tgdk.pillar"), "w") as f:
            f.write(sig)
        print("TGDK::Pillar sealed:", sig[:16])
    return sig

def tgdk_vault_sync(path):
    try:
        subprocess.run(["quomo_uplink", "vault_sync", path], check=True)
        print(f"[Vault] Synced {path} → QuomoSatNet uplink")
    except FileNotFoundError:
        print("[Vault] quomo_uplink not installed, skipping sync")
    except subprocess.CalledProcessError as e:
        print(f"[Vault] Sync failed: {e}")

def olivia_clause_echo(epoch, outdir):
    clausefile = os.path.join(outdir, f"epoch_{epoch}.clause")
    with open(clausefile, "w") as f:
        f.write(f"OliviaAI Clause Echo :: Epoch {epoch} sealed\n")
    print(f"[OliviaAI] :: Epoch {epoch} :: clausewalk sealed → {clausefile}")
    return clausefile

def hexidex(seal_sig, loss):
    return int(seal_sig, 16) % 997 ^ int(loss * 1e6)

def compute_trideotaxis_metrics(loss):
    safe = str(loss) if loss is not None else ""
    encoded = safe.encode("utf-8")
    s_scalar = (1.0 / (1.0 + loss))
    trideo = s_scalar * 3.14159
    quaitrideo = trideo ** 0.5
    return {"TGDK::S_Scalar": s_scalar, "TGDK::Trideotaxis": trideo, "TGDK::Quaitrideodynamics": quaitrideo}

def compute_jovian_metrics(loss):
    base = 11.86
    jovian_scalar = (1.0 / (1.0 + loss)) * base
    linguistics = f"JovianExpansion({jovian_scalar:.6f})"
    return {"TGDK::JovianScalar": jovian_scalar, "TGDK::JovianLinguistics": linguistics}

def compute_rigpa_hum(loss):
    emptiness = 1.0 / (1.0 + loss)
    return {"TGDK::Rigpa": {"seed": "HUM", "definition": "that which is empty and powerful", "rigpa_scalar": emptiness}}


def tgdk_seal_packet(outdir, pillar_sig, last_metrics):
    clauses = sorted(glob.glob(os.path.join(outdir, "*.clause")))

    seal = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pillar": pillar_sig,
        "last_metrics": last_metrics,
        "clauses": [os.path.basename(c) for c in clauses],
        "culmex": "sealed"
    }

    seal_path = os.path.join(outdir, "tgdk_seal.json")
    with open(seal_path, "w") as f:
        json.dump(seal, f, indent=2)

    # Sign the seal packet
    with open(PRIVATE_KEY_PATH, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )

    with open(seal_path, "rb") as f:
        data = f.read()

    signature = private_key.sign(
        data,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    sig_path = os.path.join(outdir, "tgdk_seal.sig")
    with open(sig_path, "wb") as f:
        f.write(signature)

    print(f"TGDK::Seal packet signed → {sig_path}")
    return seal_path, sig_path

def tgdk_verify_seal(seal_path, sig_path, pubkey_path):
    with open(pubkey_path, "rb") as f: public_key = serialization.load_pem_public_key(f.read(), backend=default_backend())
    with open(seal_path, "rb") as f: seal_data = f.read()
    with open(sig_path, "rb") as f: signature = f.read()
    try:
        public_key.verify(signature, seal_data, padding.PKCS1v15(), hashes.SHA256())
        print("✅ TGDK Seal verification PASSED")
        return True
    except Exception as e:
        print("❌ TGDK Seal verification FAILED:", e); sys.exit(1)

def build_charted_matrix(metrics):
    # Example: 3x3 fold expansion from loss, jovian, rigpa
    arr = np.array([
        [metrics.get("loss", 0.0), metrics.get("eval_loss", 0.0), metrics.get("hexidex", 0)],
        [metrics.get("TGDK::S_Scalar", 0.0), metrics.get("TGDK::Trideotaxis", 0.0), metrics.get("TGDK::Quaitrideodynamics", 0.0)],
        [metrics.get("TGDK::JovianScalar", 0.0), metrics["TGDK::Rigpa"]["rigpa_scalar"], 1.0]
    ])
    return arr


def ouija_sliver(matrix, pillar_sig):
    if not pillar_sig:
        pillar_sig = ""  # fallback empty string
    h = hashlib.sha256(matrix.tobytes() + pillar_sig.encode()).hexdigest()
    sliver = base64.urlsafe_b64encode(h.encode()).decode()[:64]
    return sliver
import os, lzma, shutil

def save_slivered_checkpoint(outdir: str, adapter_path: str, sliver: str) -> str | None:
    if not os.path.exists(adapter_path):
        print(f"[WARN] Adapter file not found: {adapter_path}")
        return None

    chkpt_path = os.path.join(outdir, f"checkpoint_{sliver}.xz")
    with open(adapter_path, "rb") as src, lzma.open(chkpt_path, "wb", preset=6) as dst:
        shutil.copyfileobj(src, dst, length=1024*1024)
    print(f"[TGDK] Slivered checkpoint saved → {chkpt_path}")
    return chkpt_path


def build_vectorized_planar(metrics):
    # Map metrics into 2D "GIS" vectors
    pts = np.array([
        [metrics.get("loss", 0.0), metrics.get("eval_loss", 0.0)],
        [metrics.get("TGDK::S_Scalar", 0.0), metrics.get("TGDK::Trideotaxis", 0.0)],
        [metrics.get("TGDK::JovianScalar", 0.0), metrics["TGDK::Rigpa"]["rigpa_scalar"]],
    ])
    return pts


def triangulate_points(points):
    tri = Delaunay(points)
    return tri.simplices  # indices of triangles

def quaitriangulate(points, simplices):
    new_simplices = []
    for tri in simplices:
        a, b, c = points[tri]
        ab = (a+b)/2
        bc = (b+c)/2
        ca = (c+a)/2
        center = (a+b+c)/3
        # Add 4 new triangles
        new_simplices.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c, ca, bc],
            [ab, bc, ca]
        ])
    return np.array(new_simplices)

def save_geometry(memory_db, epoch, step, pillar, metrics):
    pts = build_vectorized_planar(metrics)
    simplices = triangulate_points(pts)
    quads = quaitriangulate(pts, simplices)

    entry = {
        "epoch": epoch,
        "pillar": pillar,
        "planar_trace": pts.tolist(),
        "triangles": simplices.tolist(),
        "quaitriangles": quads.tolist(),
    }
    with open(os.path.join(outdir, f"geometry_epoch{epoch}.json"), "w") as f:
        json.dump(entry, f, indent=2)

    print(f"[TGDK-GIS] Epoch {epoch} geometry traced with {len(simplices)} tris / {len(quads)} quads")
    return entry

def next_outdir(base_name="olivia"):
    # Look for existing olivia-v1, olivia-v2, ...
    i = 1
    while True:
        candidate = f"{base_name}-v{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

os.makedirs(outdir, exist_ok=True)
print(f"[INFO] Output dir set → {outdir}")

MODELS_CONFIG = "models.config"

# --- Output directory versioning (define this early!) ---
MODELS_CONFIG = "models.config"

def next_model_version(base_name="olivia") -> tuple[str, int]:
    models = {}
    if os.path.exists(MODELS_CONFIG):
        with open(MODELS_CONFIG, "r") as f:
            try:
                models = json.load(f)
            except json.JSONDecodeError:
                print("[WARN] models.config is not valid JSON, starting fresh.")

    max_v = 0
    for key in models.keys():
        if key.startswith(base_name + "-v"):
            try:
                vnum = int(key.split("-v")[-1])
                max_v = max(max_v, vnum)
            except ValueError:
                continue

    version = max_v + 1
    out_dir = f"./{base_name}-v{version}"
    return out_dir, version


# --------- Post-process once ---------
last = trainer.state.log_history[-1] if trainer.state.log_history else {}

pillar_sig = tgdk_pillar_wrap(outdir)
s_metrics  = compute_trideotaxis_metrics(last.get("loss", 0.0))
jovian     = compute_jovian_metrics(last.get("loss", 0.0))
rigpa      = compute_rigpa_hum(last.get("loss", 0.0))
h_score    = hexidex(pillar_sig or "0", last.get("loss", 0.0))

metrics = {
    "loss": last.get("loss", 0.0),
    "eval_loss": last.get("eval_loss", 0.0),
    "hexidex": h_score, **s_metrics, **jovian, **rigpa
}

if cli_args.use_jade:
    jade = jade_lex.bind_metrics(metrics)
    metrics["loss"] = JadeCodewrightLexicon.jade_loss_reweight(metrics["loss"], jade)
    jade_lex.emit_clause(jade, outdir, epoch=0)

# Save geometry + memory once
geom_entry = save_geometry(memory_db, epoch=0, step=0, pillar=pillar_sig, metrics=metrics)
matrix = build_charted_matrix(metrics)
sliver = ouija_sliver(matrix, pillar_sig)


trainer.save_model(outdir)
tok.save_pretrained(outdir)
save_slivered_checkpoint(outdir, os.path.join(outdir, "adapter_model.bin"), sliver)
olivia_clause_echo(0, outdir)

seal_path, sig_path = tgdk_seal_packet(outdir, pillar_sig, metrics)
tgdk_verify_seal(seal_path, sig_path, PUBLIC_KEY_PATH)
tgdk_vault_sync(outdir)
# versioning
out_dir, version = next_model_version("olivia")
print(f"[INFO] Using output dir → {out_dir}")

# run training with output_dir=out_dir ...

# ensure LoRA adapter is written to top level
trainer.save_model(out_dir)

def update_models_config(model_name, path):
    if os.path.exists(MODELS_CONFIG):
        with open(MODELS_CONFIG, "r") as f:
            try:
                models = json.load(f)
            except json.JSONDecodeError:
                models = {}
    else:
        models = {}

    models[model_name] = {"path": path}

    with open(MODELS_CONFIG, "w") as f:
        json.dump(models, f, indent=2)
    print(f"[INFO] models.config updated with {model_name} → {path}")

# --------- Build optimizer (choose one path) ---------
# A) Duo optimizer from Duo.py
print("Model:", type(model))
print("Params count:", sum(p.numel() for p in model.parameters()))
print("Trainable params count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

duo_optim, duo_sched = make_duo_optimizer(model, mmt_controller)
use_optimizers = (duo_optim, duo_sched)
if hasattr(duo_optim, "ghost_gate"):
    trainer.add_callback(GhostGateCallback(duo_optim.ghost_gate))


if getattr(trainer, "use_jade_reweighting", False) and trainer.jade_lex:
    jade = trainer.jade_lex.bind_metrics(metrics)
    metrics["loss"] = JadeCodewrightLexicon.jade_loss_reweight(metrics["loss"], jade)



# now update config ONCE
update_models_config(f"olivia-v{version}", out_dir)

model = get_peft_model(model, peft_cfg)

# --- Output directory versioning ---
outdir, version = next_model_version("olivia")
os.makedirs(outdir, exist_ok=True)

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)
print(f"[INFO] Training Olivia version v{version} → {outdir}") 