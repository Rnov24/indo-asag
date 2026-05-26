"""IndoBERT fine-tuning for regression-based ASAG with robust training support."""

import numpy as np

# Torch imports are deferred to allow safe imports when torch is not installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class BertRegressorModule(nn.Module):
        """Standard PyTorch Module for IndoBERT Regression with Multi-Sample Dropout."""
        
        def __init__(self, model_name, dropout=0.1, multi_dropout=None):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.multi_dropout = multi_dropout
            
            if multi_dropout and multi_dropout > 1:
                self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(multi_dropout)])
            else:
                self.drop = nn.Dropout(dropout)
                
            self.head = nn.Linear(self.bert.config.hidden_size, 1)

        def forward(self, ids, mask):
            out = self.bert(input_ids=ids, attention_mask=mask)
            cls = out.last_hidden_state[:, 0, :]
            
            if self.training and self.multi_dropout and self.multi_dropout > 1:
                # Average predictions across multiple dropout masks
                preds = torch.stack([self.head(d(cls)).squeeze(-1) for d in self.dropouts])
                return preds.mean(dim=0), cls
            else:
                drop_fn = getattr(self, 'drop', None)
                cls_dropped = drop_fn(cls) if drop_fn else cls
                return self.head(cls_dropped).squeeze(-1), cls
else:
    class BertRegressorModule:
        pass


class PairDataset:
    """PyTorch Dataset for sentence pair regression.
    
    Tokenizes pairs of (text_a, text_b) with scores for regression.
    """
    
    def __init__(self, text_a, text_b, scores, tokenizer, max_len=128):
        import torch
        self.a = text_a
        self.b = text_b
        self.s = scores
        self.tok = tokenizer
        self.ml = max_len
        self._torch = torch
    
    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, i):
        enc = self.tok(
            self.a[i], self.b[i],
            truncation=True, padding="max_length",
            max_length=self.ml, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "score": self._torch.tensor(self.s[i], dtype=self._torch.float),
        }


class BertRegressor:
    """IndoBERT-based regression model with robust training extensions.
    
    Features:
        - Multi-Sample Dropout
        - Layer-wise Learning Rate Decay (LLRD)
        - Gradual Unfreezing (Bottom Layer Freezing)
        - R-Drop (Consistency Regularization)
    """
    
    def __init__(self, model_name="indobenchmark/indobert-base-p2", dropout=0.1, multi_dropout=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch and/or Transformers must be installed to use BertRegressor.")
        self.model_name = model_name
        self.dropout = dropout
        self.multi_dropout = multi_dropout
        
    def _get_llrd_param_groups(self, model, lr, weight_decay, decay=0.92):
        """Build parameter groups with LLRD."""
        no_decay = ["bias", "LayerNorm.weight"]
        
        # 1. Regression Head
        head_params = []
        for n, p in model.named_parameters():
            if "head" in n or "drop" in n:
                head_params.append(p)
                
        groups = [
            {"params": head_params, "lr": lr * 2, "weight_decay": weight_decay}
        ]
        
        # 2. Encoder layers (11 down to 0)
        for i in range(11, -1, -1):
            layer_lr = lr * (decay ** (11 - i))
            
            decay_group = []
            nodecay_group = []
            for n, p in model.bert.encoder.layer[i].named_parameters():
                if any(nd in n for nd in no_decay):
                    nodecay_group.append(p)
                else:
                    decay_group.append(p)
                    
            groups.extend([
                {"params": decay_group, "lr": layer_lr, "weight_decay": weight_decay},
                {"params": nodecay_group, "lr": layer_lr, "weight_decay": 0.0}
            ])
            
        # 3. Embeddings
        embed_lr = lr * (decay ** 12)
        decay_group = []
        nodecay_group = []
        for n, p in model.bert.embeddings.named_parameters():
            if any(nd in n for nd in no_decay):
                nodecay_group.append(p)
            else:
                decay_group.append(p)
                
        groups.extend([
            {"params": decay_group, "lr": embed_lr, "weight_decay": weight_decay},
            {"params": nodecay_group, "lr": embed_lr, "weight_decay": 0.0}
        ])
        
        return groups

    def _set_freezing(self, model, n_freeze, freeze=True):
        """Freeze or unfreeze the first n_freeze layers."""
        for param in model.bert.embeddings.parameters():
            param.requires_grad = not freeze
            
        for i in range(min(n_freeze, len(model.bert.encoder.layer))):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = not freeze

    def train_fold(self, train_df, val_df, fold,
                   text_a_col, text_b_col,
                   epochs=4, batch_size=16, lr=2e-5, save_path=None,
                   weight_decay=0.01, llrd_decay=None,
                   n_freeze_layers=0, unfreeze_epoch=0,
                   rdrop_alpha=None):
        """Fine-tune on one fold and return OOF predictions + embeddings.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            fold: Fold index (for logging).
            text_a_col: Column for first text (e.g., "reference_answer").
            text_b_col: Column for second text (e.g., "student_answer").
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Base learning rate.
            save_path: Path to save the best model weights.
            weight_decay: Weight decay coefficient.
            llrd_decay: LLRD decay rate (e.g., 0.92), None to disable.
            n_freeze_layers: Number of bottom BERT layers to freeze initially.
            unfreeze_epoch: Epoch index at which to unfreeze bottom layers.
            rdrop_alpha: R-Drop regularization coefficient (e.g., 0.5), None to disable.
            
        Returns:
            Tuple of (predictions, cls_embeddings) for validation set.
        """
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        train_ds = PairDataset(
            train_df[text_a_col].tolist(), train_df[text_b_col].tolist(),
            train_df["score_norm"].tolist(), tokenizer
        )
        val_ds = PairDataset(
            val_df[text_a_col].tolist(), val_df[text_b_col].tolist(),
            val_df["score_norm"].tolist(), tokenizer
        )
        
        tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        vl = DataLoader(val_ds, batch_size=batch_size * 2)
        
        model = BertRegressorModule(self.model_name, self.dropout, self.multi_dropout)
        model = model.to(device)
        
        # Freezing setup
        if n_freeze_layers > 0:
            self._set_freezing(model, n_freeze_layers, freeze=True)
            
        # Param groups & Optimizer setup
        if llrd_decay is not None:
            param_groups = self._get_llrd_param_groups(model, lr, weight_decay, llrd_decay)
            opt = torch.optim.AdamW(param_groups)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            
        total_steps = len(tl) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
        loss_fn = torch.nn.MSELoss()
        
        best_loss = float("inf")
        best_preds = None
        best_embs = None
        
        for ep in range(epochs):
            model.train()
            
            # Gradual Unfreezing check
            if n_freeze_layers > 0 and ep >= unfreeze_epoch:
                self._set_freezing(model, n_freeze_layers, freeze=False)
                # Re-setup optimizer parameters with active grads, if LLRD is enabled we keep decay
                if llrd_decay is not None:
                    # Refresh param groups with active requires_grad
                    param_groups = self._get_llrd_param_groups(model, lr, weight_decay, llrd_decay)
                    # We preserve state but update param groups
                    opt.param_groups.clear()
                    for group in param_groups:
                        opt.add_param_group(group)
            
            for b in tl:
                opt.zero_grad()
                ids, mask, scores = b["input_ids"].to(device), b["attention_mask"].to(device), b["score"].to(device)
                
                if rdrop_alpha is not None and rdrop_alpha > 0:
                    # Forward pass 1
                    p1, _ = model(ids, mask)
                    # Forward pass 2 (different dropout masks)
                    p2, _ = model(ids, mask)
                    
                    task_loss = (loss_fn(p1, scores) + loss_fn(p2, scores)) / 2.0
                    consistency_loss = F.mse_loss(p1, p2)
                    loss = task_loss + rdrop_alpha * consistency_loss
                else:
                    p, _ = model(ids, mask)
                    loss = loss_fn(p, scores)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            
            model.eval()
            vp, ve, vl_sum = [], [], 0
            with torch.no_grad():
                for b in vl:
                    p, e = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                    vl_sum += loss_fn(p, b["score"].to(device)).item()
                    vp.extend(p.cpu().numpy())
                    ve.extend(e.cpu().numpy())
            
            avg_loss = vl_sum / len(vl)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_preds = np.array(vp)
                best_embs = np.array(ve)
                if save_path:
                    torch.save(model.state_dict(), save_path)
            
            print(f"  Fold {fold} Ep {ep+1}/{epochs} val_loss={avg_loss:.4f}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_preds, best_embs
