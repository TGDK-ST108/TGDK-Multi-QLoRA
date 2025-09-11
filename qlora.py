# qlora.py – TGDK Magic + Duo + MMT + JadeCodewright + Seals + Rituals
import os, sys, torch, hashlib, json, subprocess, datetime, glob, argparse
import numpy as np
from datasets import load_dataset
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

# External TGDK geometry modules
from Mahadevi import Mahadevi
from Maharaga import Maharaga
from Trinity import Trinity
from Duo import make_duo_optimizer 
import json
from device import load_model
from accelerate import Accelerator

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
outdir   = model_cfg["outdir"]
HF_TOKEN = os.environ.get("HF_TOKEN")

os.makedirs(outdir, exist_ok=True)
model = load_model(BASE, HF_TOKEN, outdir, gpu_idx=0)

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

OUT = os.environ.get("OUT", ".")




# Accelerate dummy fallbacks (API moved across versions)
try:
    from accelerate.state import DummyOptim, DummyScheduler
except ImportError:
    try:
        from accelerate.utils.dataclasses import DummyOptim, DummyScheduler
    except ImportError:
        from compat_dummy import DummyOptim, DummyScheduler


# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)
offload_dir = os.path.join(outdir, "offload")
os.makedirs(offload_dir, exist_ok=True)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

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
              matrix.tobytes(), datetime.datetime.utcnow().isoformat()))
        self.conn.commit()

    def recall(self, limit=5, query=None):
        cur = self.conn.cursor()
        if query is None:
            query = f"SELECT * FROM memory ORDER BY id DESC LIMIT {limit}"
        cur.execute(query)
        return cur.fetchall()


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


# Instantiate
mmt_controller = MMTController(dim=256)

# Attach to Duo optimizer
duo_optim, duo_sched = make_duo_optimizer(model, mmt_controller)



# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
WORK     = os.environ.get("WORK", ".")
trainp   = os.path.join(WORK, "packs", "train.jsonl")
valp     = os.path.join(WORK, "packs", "val.jsonl")
outdir   = os.path.join(OUT, "olivia-12ob-dapt-lora")
offload_dir = os.path.join(outdir, "offload")


PRIVATE_KEY_PATH = os.environ.get("TGDK_PRIVATE_KEY", "tgdk_private.pem")
PUBLIC_KEY_PATH  = os.environ.get("TGDK_PUBLIC_KEY", "tgdk_public.pem")

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

# ------------------------------------------------------------------
# Duo Optimizer
# ------------------------------------------------------------------
class Duo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 mahadevi=None, maharaga=None, trinity=None):
        self.adamw = AdamW(params, lr=lr, weight_decay=weight_decay)
        self.lion = Lion(params, lr=lr, weight_decay=weight_decay)
        self.mahadevi = mahadevi
        self.maharaga = maharaga
        self.trinity = trinity
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Capture param snapshots
        adamw_before = {id(p): p.clone().detach()
                        for g in self.adamw.param_groups
                        for p in g['params'] if p.grad is not None}
        self.adamw.step()
        adamw_update = {pid: (p.detach() - adamw_before[pid])
                        for g in self.adamw.param_groups
                        for p in g['params'] if p.grad is not None
                        for pid in [id(p)]}

        lion_before = {id(p): p.clone().detach()
                       for g in self.lion.param_groups
                       for p in g['params'] if p.grad is not None}
        self.lion.step()
        lion_update = {pid: (p.detach() - lion_before[pid])
                       for g in self.lion.param_groups
                       for p in g['params'] if p.grad is not None
                       for pid in [id(p)]}

        alpha = self.compute_balance_factor()

        # Interweave updates
        for g in self.adamw.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                pid = id(p)
                blended = alpha * adamw_update[pid] + (1 - alpha) * lion_update[pid]
                p.data = lion_before[pid] + blended
        return loss

    def compute_balance_factor(self, epoch=None, loss=None):
        alpha = 0.5
        if self.mahadevi and self.maharaga and self.trinity:
            try:
                v1 = np.array(self.mahadevi.vector_field[0])
                v2 = np.array(self.mahadevi.vector_field[1])
                angle = self.mahadevi.angle_between_vectors(v1, v2) / 180.0
                centroid = (np.mean(self.maharaga.data_points, axis=0)
                            if self.maharaga.data_points else np.array([0.5]))
                centroid_norm = np.linalg.norm(centroid) % 1.0
                trinity_seq = np.mean(self.trinity.expand_data(np.random.rand(5)))
                alpha = float((angle + centroid_norm + trinity_seq) / 3.0)
                if epoch is not None:
                    alpha *= (1 - epoch * 0.05)
                if loss is not None:
                    alpha *= (1.0 / (1.0 + loss))
                return max(0.0, min(1.0, alpha))
            except Exception:
                return 0.5
        return alpha

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
              matrix.tobytes(), datetime.datetime.utcnow().isoformat()))
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

# Load datasets
raw_train = load_dataset("json", data_files=trainp, split="train")

# If you have a validation JSONL, load it here
if os.path.exists(valp):
    raw_val = load_dataset("json", data_files=valp, split="train")
else:
    # fallback: take a slice of train as validation
    raw_val = raw_train.select(range(min(100, len(raw_train))))
    print("[WARN] val.jsonl not found — using small slice of train as validation set")

def fmt(ex): return {"text": ex["text"]}
ds_train = raw_train.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"]) > 0).map(fmt)
ds_val   = raw_val.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"]) > 0).map(fmt)

memory_db = TGDKMemoryDB(os.path.join(outdir, "tgdk_memory.db"))

# Try recalling
try:
    recalls = memory_db.recall("SELECT * FROM memory ORDER BY id DESC LIMIT 3")
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
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=cli_args.learning_rate,
    num_train_epochs=cli_args.epochs,
    fp16=False,              # ensure disabled
    bf16=False,              # ensure disabled
    fp16_full_eval=False,
    bf16_full_eval=False,
    logging_steps=1,
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=1,
    save_steps=10,
    save_total_limit=1,
    dataloader_num_workers=0,
    report_to="none",
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

    mmt_controller = MMTController(mahadevi, maharaga, trinity)
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
        optimizer = AdamW(model.parameters(), lr=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
    elif cli_args.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
    elif cli_args.optimizer == "adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    elif cli_args.optimizer == "duo":
        mahadevi = Mahadevi()
        mahadevi.set_vector_field([np.array([1,0]), np.array([0,1])])
        maharaga = Maharaga()
        maharaga.add_data_point([0.5, 0.5])
        trinity = Trinity(0.8, 1.2, 0.95)
        optimizer = Duo(model.parameters(), lr=cli_args.learning_rate, weight_decay=cli_args.weight_decay,
                        mahadevi=mahadevi, maharaga=maharaga, trinity=trinity)
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


# --------- Build optimizer (choose one path) ---------
# A) Duo optimizer from Duo.py
duo_optim, duo_sched = make_duo_optimizer(model, mmt_controller)
use_optimizers = (duo_optim, duo_sched)

# If you prefer standard optimizers instead, comment the two lines above and use:
# total_steps = len(ds_train) * cli_args.epochs
# use_optimizers = make_optimizer_scheduler(model, cli_args, total_steps)

sft_config = SFTConfig(
    output_dir=outdir,
    overwrite_output_dir=True,         # overwrite previous runs
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,     # scale if you want effective larger batch
    learning_rate=cli_args.learning_rate,
    num_train_epochs=cli_args.epochs,
    warmup_steps=cli_args.warmup_steps,
    weight_decay=cli_args.weight_decay,
    logging_steps=10,
    eval_strategy="steps",             # replaces deprecated evaluation_strategy
    eval_steps=50,                     # eval every N steps
    save_strategy="steps",             # save on step boundaries
    save_steps=200,
    save_total_limit=2,                # keep last 2 checkpoints
    dataloader_num_workers=2,
    report_to="none",                  # disable wandb/hf tracking
    max_seq_length=4096,               # moved here (valid in SFTConfig)
    dataset_text_field="text",         # moved here (valid in SFTConfig)
    packing=False,
    fp16=False,                        # we forced off AMP earlier
    bf16=False,
    max_grad_norm=1.0,                 # ✨ stabilizes Duo spikes
    lr_scheduler_type=cli_args.scheduler_type,  # linear/cosine/polynomial
    optim="adamw_torch",               # fallback if Duo is disabled
)

# --------- Build SFTTrainer (no Accelerate wrappers, no AMP) ---------
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    peft_config=peft_cfg,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    args=train_args,          # fp16=False, bf16=False here
    packing=False,
    optimizers=use_optimizers # <- pass optimizers directly
)

# --------- Train once ---------
train_output = trainer.train()





try:
    recalls = memory_db.recall("SELECT * FROM memory ORDER BY id DESC LIMIT 3")
    if recalls:
        print(f"[TGDK-MEMORY] Warm-started with {len(recalls)} past fold entries")
except Exception as e:
    print("[TGDK-MEMORY] Recall failed:", e)

# ------------------------------------------------------------------
# TGDK Ritual Functions
# ------------------------------------------------------------------
def tgdk_log_metrics(step, loss, eval_loss=None, extra=None):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
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
    seal = {"timestamp": datetime.datetime.utcnow().isoformat(), "pillar": pillar_sig,
            "last_metrics": last_metrics, "clauses": [os.path.basename(c) for c in clauses], "culmex": "sealed"}
    seal_path = os.path.join(outdir, "tgdk_seal.json")
    with open(seal_path, "w") as f: json.dump(seal, f, indent=2)

    with open(PRIVATE_KEY_PATH, "rb") as key_file:
        private_key = serialization.load_pem_private_key(key_file.read(), password=None, backend=default_backend())
    with open(seal_path, "rb") as f: data = f.read()
    signature = private_key.sign(data, padding.PKCS1v15(), hashes.SHA256())
    sig_path = os.path.join(outdir, "tgdk_seal.sig")
    with open(sig_path, "wb") as f: f.write(signature)
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
    # Hash + fold + base64 = "slivered" imprint
    h = hashlib.sha256(matrix.tobytes() + pillar_sig.encode()).hexdigest()
    sliver = base64.urlsafe_b64encode(h.encode()).decode()[:64]  # short sliver code
    return sliver


def save_slivered_checkpoint(outdir, adapter, sliver):
    chkpt_path = os.path.join(outdir, f"checkpoint_{sliver}.xz")
    with lzma.open(chkpt_path, "wb") as f:
        f.write(open(adapter, "rb").read())
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
memory_db.save_entry(
    epoch=0, step=0,
    loss=metrics["loss"], eval_loss=metrics["eval_loss"],
    pillar=pillar_sig, matrix=matrix, sliver=sliver
)

trainer.save_model(outdir)
tok.save_pretrained(outdir)
save_slivered_checkpoint(outdir, os.path.join(outdir, "adapter_model.bin"), sliver)
olivia_clause_echo(0, outdir)

seal_path, sig_path = tgdk_seal_packet(outdir, pillar_sig, metrics)
tgdk_verify_seal(seal_path, sig_path, PUBLIC_KEY_PATH)
tgdk_vault_sync(outdir)

