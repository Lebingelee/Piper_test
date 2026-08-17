import copy
from typing import Any, Dict

import torch

from agent_factory.agents.mixins.critic.base_eval import CriticEvalMixinBase
from agent_factory.agents.mixins.critic.tdqc_common import TDQCBatchMixin, TDQCRNNConfig
from agent_factory.modules.risk_predictors.tdqc import TDQCRNN, masked_bce_loss, masked_td0_loss


class TDQCRNNMixin(TDQCBatchMixin, CriticEvalMixinBase):
    CONFIG_CLASS = TDQCRNNConfig
    CONFIG_KEY = "critic"

    def _build_tdqc_predictor(self):
        cfg = self.cfg.critic
        self._build_tdqc_encoder_override()
        self.tdqc_predictor = TDQCRNN(
            input_dim=int(getattr(cfg, "input_dim", 0)),
            rnn_type=str(getattr(cfg, "rnn_type", "lstm")),
            hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
            num_layers=int(getattr(cfg, "num_layers", 1)),
            dropout=float(getattr(cfg, "dropout", 0.0)),
            head_hidden_dims=list(getattr(cfg, "head_hidden_dims", [256])),
        )
        self.tdqc_target_predictor = copy.deepcopy(self.tdqc_predictor)
        self.tdqc_optimizer = torch.optim.AdamW(self.tdqc_predictor.parameters(), lr=float(cfg.lr))
        self._tdqc_target_synced_once = False

    def _init_tdqc_optimizers(self):
        if not hasattr(self, "tdqc_optimizer"):
            self.tdqc_optimizer = torch.optim.AdamW(self.tdqc_predictor.parameters(), lr=float(self.cfg.critic.lr))

    @torch.no_grad()
    def _sync_tdqc_target(self):
        self.tdqc_target_predictor.load_state_dict(self.tdqc_predictor.state_dict())
        self.tdqc_target_predictor.requires_grad_(False)

    def update_tdqc(self, batch: Dict[str, Any]) -> Dict[str, float]:
        prepared = self._prepare_tdqc_batch(batch)
        step_features = prepared["step_features"]
        valid_mask = prepared["valid_mask"]
        success_label = prepared["success_label"]

        logits = self.tdqc_predictor(step_features, valid_mask=valid_mask)
        success_prob = torch.sigmoid(logits)
        loss_type = str(getattr(self.cfg.critic, "loss_type", "td0")).strip().lower()

        if loss_type in {"bce", "mc", "monte_carlo"}:
            loss_parts = masked_bce_loss(logits, valid_mask, success_label)
        elif loss_type in {"td0", "td-0", "td"}:
            with torch.no_grad():
                target_logits = self.tdqc_target_predictor(step_features, valid_mask=valid_mask)
                if not self._tdqc_target_synced_once:
                    self._sync_tdqc_target()
                    self._tdqc_target_synced_once = True
                    target_logits = self.tdqc_target_predictor(step_features, valid_mask=valid_mask)
                target_prob = torch.sigmoid(target_logits)
            loss_parts = masked_td0_loss(success_prob, target_prob, valid_mask, success_label)
        else:
            raise ValueError(f"Unsupported TDQC loss_type={loss_type!r}. Expected 'td0' or 'bce'.")

        loss = loss_parts["loss"]
        self.tdqc_optimizer.zero_grad()
        loss.backward()
        grad_clip = float(getattr(self.cfg.critic, "grad_clip_norm", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.tdqc_predictor.parameters(), grad_clip)
        self.tdqc_optimizer.step()

        self.step += 1
        target_interval = max(int(getattr(self.cfg.critic, "target_update_interval", 100)), 1)
        if self.step % target_interval == 0:
            self._sync_tdqc_target()

        with torch.no_grad():
            eval_success = self._select_eval_timestep(success_prob, valid_mask)
            valid = valid_mask.float()
            if valid.ndim == 3 and valid.shape[-1] == 1:
                valid = valid.squeeze(-1)
            success_mean = (success_prob.squeeze(-1) * valid).sum() / valid.sum().clamp_min(1.0)

        metrics = {
            key: float(value.detach().cpu().item())
            for key, value in loss_parts.items()
            if torch.is_tensor(value) and value.ndim == 0
        }
        metrics.update(
            {
                "loss_tdqc": float(loss.detach().cpu().item()),
                "success_prob_mean": float(success_mean.detach().cpu().item()),
                "eval_success_prob_mean": float(eval_success.mean().detach().cpu().item()),
                "success_label_mean": float(success_label.float().mean().detach().cpu().item()),
            }
        )
        return metrics

    def train_tdqc_loop(self, dataloader, num_steps: int, save_dir: str = ""):
        import os
        from tqdm import tqdm

        self.train()
        if hasattr(dataloader, "__len__") and len(dataloader) <= 0:
            raise ValueError("TDQC-RNN dataloader is empty; cannot train.")
        log_interval = max(int(getattr(self.cfg.critic, "log_interval", 100)), 1)
        save_interval = max(min(int(getattr(self.cfg.train, "save_interval", num_steps // 2)), max(num_steps // 4, 1)), 1)

        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        iterator = infinite_iterator(dataloader)
        running: Dict[str, float] = {}
        pbar = tqdm(range(num_steps), desc="Train TDQC RNN", leave=True)
        for step_idx in pbar:
            batch = self._batch_to_device(next(iterator)) if hasattr(self, "_batch_to_device") else next(iterator)
            metrics = self.update_tdqc(batch)
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)

            if (step_idx + 1) % log_interval == 0:
                shown = {
                    "loss": running.get("loss_tdqc", 0.0) / log_interval,
                    "p": running.get("success_prob_mean", 0.0) / log_interval,
                    "y": running.get("success_label_mean", 0.0) / log_interval,
                }
                pbar.set_postfix(shown)
                running = {}

            if save_dir and (step_idx + 1) % save_interval == 0:
                os.makedirs(save_dir, exist_ok=True)
                self.save(
                    os.path.join(save_dir, f"tdqc_rnn_step_{step_idx + 1}.pth"),
                    meta={"mode": "tdqc_rnn"},
                )

    @torch.no_grad()
    def predict_success_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        prepared = self._prepare_tdqc_batch(batch)
        logits = self.tdqc_predictor(prepared["step_features"], valid_mask=prepared["valid_mask"])
        success_prob = torch.sigmoid(logits)
        return self._select_eval_timestep(success_prob, prepared["valid_mask"]).detach().cpu()

    @torch.no_grad()
    def eval_risk_batch(self, batch: Dict[str, Any], only_obs: bool = True) -> Dict[str, Any]:
        del only_obs
        prepared = self._prepare_tdqc_batch(batch)
        valid_mask = prepared["valid_mask"]
        logits = self.tdqc_predictor(prepared["step_features"], valid_mask=valid_mask)
        success_prob_all = torch.sigmoid(logits)
        success_prob = self._select_eval_timestep(success_prob_all, valid_mask).detach().cpu()
        failure_prob = 1.0 - success_prob
        success_prob_all_cpu = success_prob_all.squeeze(-1).detach().cpu()
        failure_prob_all_cpu = 1.0 - success_prob_all_cpu
        valid_mask_cpu = valid_mask.detach().cpu()

        outputs: Dict[str, Any] = {
            "risk_score": failure_prob,
            "risk_signal/raw_success_prob": success_prob,
            "risk_signal/raw_failure_prob": failure_prob,
            "risk_signal/raw_success_prob_all": success_prob_all_cpu,
            "risk_signal/raw_failure_prob_all": failure_prob_all_cpu,
            "risk_signal/valid_mask": valid_mask_cpu,
            "risk_signal/method_id": "tdqc_rnn",
            "risk_signal/display_name": "TDQC RNN",
            "risk_signal/score_formula": "one_minus_success_prob",
        }
        if "frame" in batch:
            frame = batch["frame"]
            if torch.is_tensor(frame):
                frame_all = frame.detach().cpu().long()
                frame = self._select_eval_timestep(frame, valid_mask).detach().cpu().long()
                outputs["frame_all"] = frame_all
                outputs["risk_signal/frame_all"] = frame_all
            outputs["frame"] = frame
        return outputs

    @torch.no_grad()
    def eval_risk_sequence_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.eval_risk_batch(batch, only_obs=True)
