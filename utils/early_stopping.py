# Autora: Giulia Katherine Romas Tomazeli

import torch

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode="min", save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            return

        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

    def _is_improvement(self, score):
        if self.mode == "max":
            return score > self.best_score + self.min_delta

        return score < self.best_score - self.min_delta

    def _save_checkpoint(self, model):
        if self.save_path is None:
            return

        torch.save(model.state_dict(), self.save_path)