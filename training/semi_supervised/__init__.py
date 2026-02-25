"""Semi-supervised learning techniques for PesaFlow ML.

Provides SSL trainers that leverage unlabeled predictions to improve model quality:
  - Self-training (pseudo-labeling with confidence thresholds)
  - Label propagation (graph-based label spreading)
  - Consistency regularization (perturbation-based)
  - MixMatch (augmentation + sharpening + MixUp)
"""

from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer
from training.semi_supervised.consistency_regularization import ConsistencyRegularizationTrainer
from training.semi_supervised.label_propagation import LabelPropagationTrainer
from training.semi_supervised.mixmatch import MixMatchTrainer
from training.semi_supervised.self_training import SelfTrainingTrainer

__all__ = [
    "BaseSSLTrainer",
    "SelfTrainingTrainer",
    "LabelPropagationTrainer",
    "ConsistencyRegularizationTrainer",
    "MixMatchTrainer",
]
