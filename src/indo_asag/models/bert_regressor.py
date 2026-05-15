"""IndoBERT fine-tuning for regression-based ASAG."""

import numpy as np

# Torch imports are deferred to avoid errors when torch is not installed


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
    """IndoBERT-based regression model.
    
    Architecture: [CLS] text_a [SEP] text_b [SEP] → Dropout → Linear(768, 1)
    
    Also returns [CLS] embeddings for downstream Late Fusion / Naive Concat.
    """
    
    def __init__(self, model_name="indobenchmark/indobert-base-p2", dropout=0.1):
        import torch
        from transformers import AutoModel
        
        self.torch = torch
        self._nn_module = type(
            "BertRegressorModule",
            (torch.nn.Module,),
            {
                "__init__": lambda self_m, name, dp: (
                    super(type(self_m), self_m).__init__(),
                    setattr(self_m, "bert", AutoModel.from_pretrained(name)),
                    setattr(self_m, "drop", torch.nn.Dropout(dp)),
                    setattr(self_m, "head", torch.nn.Linear(
                        self_m.bert.config.hidden_size, 1
                    )),
                )[-1],
                "forward": lambda self_m, ids, mask: (
                    lambda out: (
                        self_m.head(self_m.drop(out.last_hidden_state[:, 0, :])).squeeze(-1),
                        out.last_hidden_state[:, 0, :]
                    )
                )(self_m.bert(input_ids=ids, attention_mask=mask)),
            }
        )
        self.model_name = model_name
        self.dropout = dropout
    
    def train_fold(self, train_df, val_df, fold,
                   text_a_col, text_b_col,
                   epochs=4, batch_size=16, lr=2e-5, save_path=None):
        """Fine-tune on one fold and return OOF predictions + embeddings.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            fold: Fold index (for logging).
            text_a_col: Column for first text (e.g., "reference_answer").
            text_b_col: Column for second text (e.g., "student_answer").
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Learning rate.
            
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
        
        model = self._nn_module(self.model_name, self.dropout)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(tl) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
        loss_fn = torch.nn.MSELoss()
        
        best_loss = float("inf")
        best_preds = None
        best_embs = None
        
        for ep in range(epochs):
            model.train()
            for b in tl:
                opt.zero_grad()
                p, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                loss = loss_fn(p, b["score"].to(device))
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
