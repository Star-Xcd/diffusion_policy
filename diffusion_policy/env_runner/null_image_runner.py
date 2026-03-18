from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NullImageRunner(BaseImageRunner):
    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir=output_dir)
        self.kwargs = kwargs

    def run(self, policy: BaseImagePolicy) -> Dict:
        # Training-only placeholder. Real robot evaluation is handled separately.
        _ = policy
        return {}
