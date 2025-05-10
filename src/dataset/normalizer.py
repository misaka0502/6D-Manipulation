from typing import Dict, Union
import torch.nn as nn
import numpy as np
from src.behavior.ibc_utils.pytorch_util import dict_apply
import torch
from ipdb import set_trace as bp

def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x

class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stats = nn.ParameterDict()

    def fit(self, data_dict):
        for key, tensor in data_dict.items():
            min_value = tensor.min(dim=0)[0]
            max_value = tensor.max(dim=0)[0]

            # Check if any column has only one value throughout
            diff = max_value - min_value
            constant_columns = diff == 0

            # Set a small range for constant columns to avoid division by zero
            min_value[constant_columns] -= 1
            max_value[constant_columns] += 1

            self.stats[key] = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_value, requires_grad=False),
                    "max": nn.Parameter(max_value, requires_grad=False),
                },
            )
        self._turn_off_gradients()

    def _normalize(self, x, key):
        stats = self.stats[key]
        x = (x - stats["min"]) / (stats["max"] - stats["min"])
        x = 2 * x - 1
        return x

    def _denormalize(self, x, key):
        stats = self.stats[key]
        x = (x + 1) / 2
        x = x * (stats["max"] - stats["min"]) + stats["min"]
        return x

    def forward(self, x, key, forward=True):
        if forward:
            return self._normalize(x, key)
        else:
            return self._denormalize(x, key)

    def _turn_off_gradients(self):
        for key in self.stats.keys():
            for stat in self.stats[key].keys():
                self.stats[key][stat].requires_grad = False

    def load_state_dict(self, state_dict):

        stats = nn.ParameterDict()
        for key, value in state_dict.items():
            if key.startswith("stats."):
                param_key = key[6:]
                keys = param_key.split(".")
                current_dict = stats
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = nn.ParameterDict()
                    current_dict = current_dict[k]
                current_dict[keys[-1]] = nn.Parameter(value)

        self.stats = stats
        self._turn_off_gradients()

        return f"<Added keys {self.stats.keys()} to the normalizer.>"

    def keys(self):
        return self.stats.keys()
    
    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)
    
    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result

if __name__ == "__main__":
    # Create the normalizers
    linear_normalizer = LinearNormalizer()

    # Print the stats
    print("Linear normalizer stats:")
    print(linear_normalizer.stats_dict)

    # Test the fit function for the LinearNormalizer
    # Create fake parts_poses data of shape (64, 35)
    data = {
        "parts_poses": np.random.randint(0, 100, (64, 35)).astype(np.float32),
    }

    # Fit the data
    linear_normalizer.fit(data)

    # Print the stats
    print("Linear normalizer stats after fitting:")
    print(linear_normalizer.stats_dict.keys())
    print(linear_normalizer.stats_dict["parts_poses"])
    print(linear_normalizer.stats_dict["parts_poses"]["min"].shape)
