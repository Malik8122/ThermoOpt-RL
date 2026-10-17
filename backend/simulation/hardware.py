"""Individual devices inside a rack.

Resource accounting is done at rack level (jobs request rack CPU/GPU units).
Within a rack, load is assumed to be balanced evenly, so every server runs at
the rack's CPU utilisation and every GPU node at the rack's GPU utilisation.
Devices exist so power is computed per device and summed, which keeps the
model correct if per-device load balancing is added later.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.simulation.power import PowerModel


@dataclass
class Server:
    server_id: str
    power_model: PowerModel
    utilisation: float = 0.0   # CPU utilisation fraction 0-1

    def power_w(self) -> float:
        return self.power_model.server_power_w(self.utilisation)


@dataclass
class GPUNode:
    node_id: str
    power_model: PowerModel
    utilisation: float = 0.0   # GPU utilisation fraction 0-1

    def power_w(self) -> float:
        return self.power_model.gpu_power_w(self.utilisation)
