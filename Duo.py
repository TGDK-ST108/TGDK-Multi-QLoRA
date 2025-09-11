# Duo.py – TGDK Legacy + QLoRA Optimizer Integration (Qiskit replaced with Scoring)

import logging
import numpy as np
import torch
from torch.optim import Optimizer, AdamW
from lion_pytorch import Lion
from torch.optim.lr_scheduler import LambdaLR

# TGDK scoring utilities
from scoring import simulate_quantum_with_scorer
# visceptor module (TGDK Sensors)
from tgdk_sensors import visceptor
from accelerate import Accelerator
from tgdk_accelerator import GhostGateAccelerator
from accelerate.optimizer import AcceleratedOptimizer

accelerator = Accelerator(mixed_precision="no")  # no AMP at all



# -----------------------------
# DataSectorDuoqiadratilizer
# -----------------------------
class DataSectorDuoqiadratilizer:
    def __init__(self, sector_count=8):
        logging.info("Initializing Data Sector Duoqiadratilizer (Scoring-powered)")
        self.sector_count = sector_count
        self.sympathizers = self._initialize_sympathizers(sector_count)
        self.indicators = self._initialize_indicators(sector_count)
        self.vector_sequences = self._initialize_vector_sequences(sector_count)

    def _initialize_sympathizers(self, sector_count):
        return [np.random.rand(sector_count) for _ in range(sector_count)]

    def _initialize_indicators(self, sector_count):
        return np.random.rand(sector_count)

    def _initialize_vector_sequences(self, sector_count):
        return [np.sin(np.linspace(0, 2 * np.pi, sector_count))
                for _ in range(sector_count)]

    def _apply_duoquadratic_modifications(self, data):
        modified_data = []
        for d in data:
            vector = np.array([ord(c) for c in d])
            modified_vector = vector + np.random.choice(self.sympathizers)
            modified_data.append(modified_vector)
        return modified_data

    def duoqiadratilize(self, data):
        """
        Perform secure duoqiadratilization using TGDK scoring instead of Qiskit.
        Returns: dict with S, amplitudes, probabilities, peak_state.
        """
        modified_data = self._apply_duoquadratic_modifications(data)
        flat = np.concatenate(modified_data)
        F = float(np.mean(flat))
        L = float(np.std(flat))
        M = float(np.median(flat))
        x = float(np.sum(flat) % 1000) / 1000.0

        result = simulate_quantum_with_scorer(F, L, M, x, size=self.sector_count)
        logging.info(f"[Duo] Duoqiadratilization result: {result}")
        return result

    def CoordVal(
        x,
        packet,
        DivValue,
        Metscore,
        Situation,
        Logistics,
        Location,
        Overfold,
        visceptor,
        disceptor,
        sublimationMetric,
        MatrixClause,
        PayloadRelease,
    ):
        term1 = circumferentialize_degree_field(packet + x) / DivValue
        term2 = (Metscore / Situation) * Logistics * Location / Overfold
        term3 = sublimationMetric / MatrixClause * PayloadRelease
        return term1 - term2 - disceptor + term3


class GentuoGuide:
    """
    Gentuo layer: guides the blend of AdamW and Lion updates
    by orbiting codepoints in a controlled timestep.
    """
    def __init__(self, warp_factor=0.5):
        self.warp_factor = warp_factor

    def warp(self, adamw_update, lion_update):
        # delta between the two optimizers
        delta = lion_update - adamw_update
        # warp it slightly, keep orbit stable
        return adamw_update + self.warp_factor * delta

class AMPProxyOptimizer(torch.optim.Optimizer):
    def __init__(self, duo_optim):
        self.duo_optim = duo_optim
        self.param_groups = duo_optim.param_groups

    def step(self, closure=None):
        return self.duo_optim.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        return self.duo_optim.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.duo_optim.state_dict()

    def load_state_dict(self, state_dict):
        return self.duo_optim.load_state_dict(state_dict)


# -----------------------------
# AMP-safe Duo Optimizer
# -----------------------------
class DuoOptimizer(Optimizer):
    _step_supports_amp_scaling = True  # tells AMP we can handle scaled steps

    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 collator=None, mmt_controller=None):
        params = [p for p in params if getattr(p, "requires_grad", False)]
        if not params:
            # safety fallback to avoid "empty parameter list"
            params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # sub-optimizers using *self.param_groups*
        self.adamw = torch.optim.AdamW(self.param_groups, lr=lr, weight_decay=weight_decay)
        self.lion  = Lion(self.param_groups, lr=lr, weight_decay=weight_decay)

        self.collator = collator
        self.mmt_controller = mmt_controller
        self._step_count = 0

    def step(self, closure=None, grad_scaler=None):
        """
        Performs a single optimization step (AdamW + Lion blended).
        AMP-friendly: supports optional grad_scaler.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if grad_scaler is not None:
            grad_scaler.step(self.adamw)
            grad_scaler.step(self.lion)
        else:
            self.adamw.step()
            self.lion.step()

        if self.collator:
            self.collator.coalesce(self.param_groups[0]["params"])

        if self.mmt_controller:
            alpha = self.mmt_controller.step().get("volumetric", [0.5])[0]
            for g in self.param_groups:
                g["lr"] *= max(0.5, min(1.0, float(alpha)))

        self._step_count += 1
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """
        Clears gradients of all optimized parameters.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach_()
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        """
        return {
            "adamw": self.adamw.state_dict(),
            "lion": self.lion.state_dict(),
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        self.adamw.load_state_dict(state_dict["adamw"])
        self.lion.load_state_dict(state_dict["lion"])
        self._step_count = state_dict.get("_step_count", 0)

    @property
    def duo_param_groups(self):
        return self.adamw.param_groups





# -----------------------------
# Duo-bound Optimizer Factory
# -----------------------------

def make_duo_optimizer(model, mmt_controller=None, lr=2e-5, weight_decay=0.01):
    collator = ForkedCoalescingMatrixCollator()
    ghost_gate = GhostGateAccelerator(mixed_precision="no")

    if model is None:
        logging.warning("[Duo] No model provided, returning Dummy optimizer.")
        return DummyOptim([]), DummyScheduler()

    duo = DuoOptimizer(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        collator=collator,
        mmt_controller=mmt_controller,
    )

    sched = LambdaLR(duo, lr_lambda=lambda step: 1.0)

    logging.info("[Duo] DuoOptimizer initialized with AdamW+Lion+GhostGate+MMT.")
    return duo, sched


# -----------------------------
# Gradient Collator
# -----------------------------
class ForkedCoalescingMatrixCollator:
    """
    TGDK collator: coalesces gradients and applies Duo duoqiadratilization (via scoring).
    """

    def __init__(self):
        self.duoq = DataSectorDuoqiadratilizer()

    def coalesce(self, params):
        grads_as_text = [str(p.grad.shape) for p in params if getattr(p, "grad", None) is not None]
        if grads_as_text:
            try:
                result = self.duoq.duoqiadratilize(grads_as_text)
                logging.info(f"[Collator] Quantum merge result: {result}")
            except Exception as e:
                logging.warning(f"[Collator] Duoqiadratilization failed: {e}")


# -----------------------------
# Dummy Fallbacks
# -----------------------------
class DummyOptim:
    def __init__(self, params=None, collator=None):
        self.params = list(params) if params is not None else []
        self.collator = collator or ForkedCoalescingMatrixCollator()

    def step(self, *args, **kwargs):
        if self.collator:
            self.collator.coalesce(self.params)

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self): return {}
    def load_state_dict(self, state_dict): pass


class DummyScheduler:
    def __init__(self, optimizer=None): self.optimizer = optimizer
    def step(self, *args, **kwargs): pass
    def state_dict(self): return {}
    def load_state_dict(self, state_dict): pass


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Dummy model for testing
    dummy_model = torch.nn.Linear(10, 2)

    # Run optimizer factory with dummy
    optimizer, scheduler = make_duo_optimizer(dummy_model, mmt_controller=None)
    optimizer = accelerator.prepare_optimizer(optimizer)
    print("Optimizer:", optimizer)
    print("Scheduler:", scheduler)
