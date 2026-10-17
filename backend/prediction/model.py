"""Temperature forecaster (Step 12): a recurrent network over the last minutes of one rack's signals.

    input   (batch, sequence_length, n_features), standardised with the training set's mean and std
    network LSTM or GRU (num_layers x hidden_size, dropout between layers) -> Linear(hidden_size, n_horizons)
    output  temperature CHANGE from now to each horizon, in units of the training set's per-horizon std
    forecast = current temperature + predicted change

Predicting the change, not the absolute temperature, makes "no change" the zero output, so the model
starts from the persistence forecast and only has to learn the departure from it.

``ForecastProvider`` adapts a trained forecaster to the environment's ``PredictionProvider`` interface: it keeps
the last minutes of features itself and returns a (num_racks, n_horizons) array of forecast temperatures in °C.

    python -m backend.prediction.model                      # train the configured architecture -> models/temperature_predictor/
    python -m backend.prediction.model --architecture gru   # comparison model -> models/temperature_predictor_gru/
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.config import Config, get_config
from backend.prediction.dataset import EXTRA_FEATURES, History, Windows, make_windows, rack_features, split_days
from backend.rl.dqn import configure_torch
from backend.simulation.datacentre import DataCentre
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed, make_rng, set_global_seed

logger = get_logger("prediction.model")

WEIGHTS = "forecaster.pt"
META_FILE = "forecaster.json"
CSV_FIELDS = ("epoch", "train_loss", "val_loss", "val_mae_c", "best")


class RecurrentRegressor(nn.Module):
    def __init__(self, n_features: int, n_horizons: int, architecture: str, hidden_size: int, num_layers: int,
                 dropout: float) -> None:
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[architecture]
        self.rnn = rnn(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, n_horizons)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.n_features:
            raise ValueError(f"expected (batch, steps, {self.n_features}), got {tuple(x.shape)}")
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class TemperatureForecaster:
    """A trained (or trainable) network plus the scaling and metadata needed to use it."""

    def __init__(self, feature_names: list[str], horizons: list[int], sequence_length: int, architecture: str,
                 hidden_size: int, num_layers: int, dropout: float, seed: int = 42) -> None:
        self.feature_names, self.horizons, self.sequence_length = list(feature_names), list(horizons), sequence_length
        self.architecture, self.hidden_size, self.num_layers, self.dropout = architecture, hidden_size, num_layers, dropout
        self.seed = seed
        self.temp_index = self.feature_names.index("temperature")
        torch.manual_seed(derive_seed(seed, "prediction", "init"))
        self.net = RecurrentRegressor(len(feature_names), len(horizons), architecture, hidden_size, num_layers, dropout)
        self.x_mean = np.zeros(len(feature_names), dtype=np.float32)
        self.x_std = np.ones(len(feature_names), dtype=np.float32)
        self.y_std = np.ones(len(horizons), dtype=np.float32)
        self.metadata: dict = {}

    @classmethod
    def from_config(cls, config: Config, architecture: str | None = None,
                    features: list[str] | None = None) -> "TemperatureForecaster":
        p = config.prediction
        return cls(features or p.features, p.horizons_minutes, p.sequence_length, architecture or p.architecture, p.hidden_size,
                   p.num_layers, p.dropout, p.seed)

    # ------------------------------------------------------------- scaling
    def fit_scalers(self, train: Windows) -> None:
        flat = train.x.reshape(-1, train.x.shape[-1])
        self.x_mean = flat.mean(axis=0).astype(np.float32)
        self.x_std = np.maximum(flat.std(axis=0), 1e-6).astype(np.float32)
        self.y_std = np.maximum(train.y.std(axis=0), 1e-6).astype(np.float32)

    def _scale_x(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_std)

    # ------------------------------------------------------------ inference
    def predict_change(self, x: np.ndarray) -> np.ndarray:
        """Temperature change in °C to each horizon for raw windows (N, L, F)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (self.sequence_length, len(self.feature_names)):
            raise ValueError(f"windows must be (N, {self.sequence_length}, {len(self.feature_names)}), got {x.shape}")
        self.net.eval()
        with torch.no_grad():
            out = torch.cat([self.net(self._scale_x(x[i:i + 4096])) for i in range(0, len(x), 4096)])
        return out.numpy() * self.y_std

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forecast temperature in °C at each horizon: last observed temperature + predicted change."""
        x = np.asarray(x, dtype=np.float32)
        return x[:, -1, self.temp_index][:, None] + self.predict_change(x)

    # ------------------------------------------------------------- training
    def fit(self, train: Windows, val: Windows, epochs: int, batch_size: int, learning_rate: float) -> list[dict]:
        """Adam on the MSE of the standardised change. Keeps the weights of the epoch with the lowest validation loss."""
        self.fit_scalers(train)
        x_tr, y_tr = self._scale_x(train.x), torch.as_tensor(train.y / self.y_std)
        x_va, y_va = self._scale_x(val.x), torch.as_tensor(val.y / self.y_std)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        shuffle = make_rng(derive_seed(self.seed, "prediction", "shuffle"))
        torch.manual_seed(derive_seed(self.seed, "prediction", "dropout"))
        best_val, best_state, rows = float("inf"), None, []
        for epoch in range(1, epochs + 1):
            self.net.train()
            order = shuffle.permutation(len(x_tr))
            total = 0.0
            for i in range(0, len(order), batch_size):
                idx = torch.as_tensor(order[i:i + batch_size])
                loss = loss_fn(self.net(x_tr[idx]), y_tr[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                pred = torch.cat([self.net(x_va[i:i + 4096]) for i in range(0, len(x_va), 4096)])
                val_loss = loss_fn(pred, y_va).item()
                val_mae = float(((pred - y_va).abs().numpy() * self.y_std).mean())
            if not np.isfinite(val_loss):
                raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
            improved = val_loss < best_val
            if improved:
                best_val, best_state = val_loss, {k: v.clone() for k, v in self.net.state_dict().items()}
            rows.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss,
                         "val_mae_c": val_mae, "best": int(improved)})
            logger.info("%s epoch %d/%d  train %.4f  val %.4f  val MAE %.3f °C%s", self.architecture, epoch, epochs,
                        rows[-1]["train_loss"], val_loss, val_mae, "  *" if improved else "")
        self.net.load_state_dict(best_state)
        self.net.eval()
        return rows

    # ---------------------------------------------------------- persistence
    def save(self, directory: Path, metadata: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / WEIGHTS)
        payload = {"feature_names": self.feature_names, "horizons_minutes": self.horizons,
                   "sequence_length": self.sequence_length, "architecture": self.architecture,
                   "hidden_size": self.hidden_size, "num_layers": self.num_layers, "dropout": self.dropout,
                   "seed": self.seed, "target": "temperature change from the last input frame, °C",
                   "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(), "y_std": self.y_std.tolist(),
                   "parameters": sum(p.numel() for p in self.net.parameters()), "weights": WEIGHTS,
                   **(metadata or {})}
        (directory / META_FILE).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return directory / META_FILE

    @classmethod
    def load(cls, directory: Path) -> "TemperatureForecaster":
        directory = Path(directory)
        m = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        f = cls(m["feature_names"], m["horizons_minutes"], m["sequence_length"], m["architecture"], m["hidden_size"],
                m["num_layers"], m["dropout"], m["seed"])
        f.net.load_state_dict(torch.load(directory / m["weights"], map_location="cpu", weights_only=True))
        f.net.eval()
        f.x_mean, f.x_std = np.array(m["x_mean"], dtype=np.float32), np.array(m["x_std"], dtype=np.float32)
        f.y_std = np.array(m["y_std"], dtype=np.float32)
        f.metadata = m
        return f


class ForecastProvider:
    """``PredictionProvider`` for ``DataCenterEnv``: forecast temperatures (num_racks, n_horizons) in °C.

    Keeps one frame of ``rack_features`` per simulated minute. A new episode (a different simulator object, or the
    clock going backwards) clears the history. Until ``sequence_length`` frames exist, the earliest frame is
    repeated, so the first forecasts of an episode are close to "no change". Calling it twice in the same minute
    does not add a second frame.
    """

    def __init__(self, forecaster: TemperatureForecaster) -> None:
        self.forecaster = forecaster
        self.history: deque[np.ndarray] = deque(maxlen=forecaster.sequence_length)
        self._dc_id: int | None = None
        self._minute: int | None = None
        self.last_forecast: np.ndarray | None = None

    def reset(self) -> None:
        self.history.clear()
        self._dc_id = self._minute = None

    def __call__(self, dc: DataCentre) -> np.ndarray:
        if id(dc) != self._dc_id or self._minute is None or dc.minute < self._minute:
            self.reset()
            self._dc_id = id(dc)
        if dc.minute != self._minute:
            self.history.append(rack_features(dc, self.forecaster.feature_names))
            self._minute = dc.minute
        frames = list(self.history)
        frames = [frames[0]] * (self.forecaster.sequence_length - len(frames)) + frames
        windows = np.stack(frames, axis=1)                    # (racks, L, F)
        self.last_forecast = self.forecaster.predict(windows).astype(np.float64)
        return self.last_forecast


# --------------------------------------------------------------------------- #
def train_forecaster(config: Config, history: History, architecture: str | None = None,
                     features: list[str] | None = None) -> tuple[TemperatureForecaster, list[dict], dict]:
    p = config.prediction
    features = list(features or p.features)
    train_days, val_days, test_days = split_days(len(history.policies), p.split)
    train = make_windows(history, train_days, p.sequence_length, p.horizons_minutes, features)
    val = make_windows(history, val_days, p.sequence_length, p.horizons_minutes, features)
    forecaster = TemperatureForecaster.from_config(config, architecture, features)
    start = time.perf_counter()
    rows = forecaster.fit(train, val, p.epochs, p.batch_size, p.learning_rate)
    best = min(rows, key=lambda r: r["val_loss"])
    info = {"train_days": list(train_days), "val_days": list(val_days), "test_days": list(test_days),
            "train_windows": len(train), "val_windows": len(val), "epochs": p.epochs, "batch_size": p.batch_size,
            "learning_rate": p.learning_rate, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
            "seconds": time.perf_counter() - start, "torch_version": torch.__version__,
            "day_policies": history.policies, "day_seeds": history.seeds}
    return forecaster, rows, info


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", choices=["lstm", "gru"], default=cfg.prediction.architecture)
    parser.add_argument("--history", type=Path, default=cfg.paths.data / "temperature_history.npz")
    parser.add_argument("--extended-features", action="store_true",
                        help="feature ablation: configured features + " + ", ".join(EXTRA_FEATURES))
    args = parser.parse_args()
    primary = args.architecture == cfg.prediction.architecture and not args.extended_features
    tag = "temperature_predictor" if primary else (
        "temperature_predictor_extended" if args.extended_features
        else f"temperature_predictor_{args.architecture}")
    features = list(cfg.prediction.features) + (EXTRA_FEATURES if args.extended_features else [])

    set_global_seed(cfg.prediction.seed)
    configure_torch(cfg.prediction.torch_threads)
    forecaster, rows, info = train_forecaster(cfg, History.load(args.history), args.architecture, features)
    csv_path = cfg.paths.experiments / f"{tag}_training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    forecaster.save(cfg.paths.models / tag, {"training": {**info, "history_file": str(args.history),
                                                          "training_csv": str(csv_path)}})
    print(f"{args.architecture.upper()} forecaster: {info['train_windows']} training windows, best epoch "
          f"{info['best_epoch']}/{info['epochs']} (val loss {info['best_val_loss']:.4f}), {info['seconds']:.0f} s "
          f"-> models/{tag}")


if __name__ == "__main__":
    main()
