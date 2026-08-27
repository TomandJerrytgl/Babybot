"""Replaceable neural-feature training boundary for Dreaming mode."""


class FeatureTrainer:
    configured = False

    def status(self):
        return {
            "configured": False,
            "message": "Neural network trainer is not configured; algorithm features remain available",
        }

    def add_object_sample(self, *_args, **_kwargs):
        return None

    def train(self):
        return self.status()

    def extract_feature(self, *_args, **_kwargs):
        return None

    def save_checkpoint(self, *_args, **_kwargs):
        return None

    def load_checkpoint(self, *_args, **_kwargs):
        return None
