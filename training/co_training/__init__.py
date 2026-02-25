"""Co-training techniques for PesaFlow ML.

Provides co-training methods that cross-pollinate labels between views and domains:
  - Multi-view co-training (split features into views, each view teaches others)
  - Cross-domain co-training (fraud→AML→merchant label transfer)
  - Tri-training (3-classifier agreement-based pseudo-labeling)
"""

from training.co_training.cross_domain_trainer import CrossDomainTrainer
from training.co_training.multi_view_trainer import MultiViewCoTrainer
from training.co_training.tri_training import TriTrainingTrainer

__all__ = [
    "MultiViewCoTrainer",
    "CrossDomainTrainer",
    "TriTrainingTrainer",
]
