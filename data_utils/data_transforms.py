import torch


class DataTransform:
    def __init__(self, is_train, use_volume=False, additional_features=[]):
        self.is_train = is_train
        self.keys = ['timestamp', 'open', 'high', 'low', 'close']
        if use_volume:
            self.keys.append('volume')
        self.keys += additional_features
        print(f"Using features: {self.keys}")


    def __call__(self, window):
        data_list = []
        output = {}
        if 'Timestamp_orig' in window.keys():
            self.keys.append('Timestamp_orig')
        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())
            if key == 'volume':
                data /= 1e9
            output[key] = data[-1]
            output[f'{key}_old'] = data[-2]
            if key == 'Timestamp_orig':
                continue
            data_list.append(data[:-1].reshape(1, -1))
        features = torch.cat(data_list, 0)
        output['features'] = features
        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)