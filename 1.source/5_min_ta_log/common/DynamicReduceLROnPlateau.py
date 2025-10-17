from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

custom_switch_factor = 0.1

class DynamicReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, switch_epoch=20, switch_factor=custom_switch_factor, **kwargs):
        super().__init__(**kwargs)
        self.switch_epoch = switch_epoch
        self.switch_factor = switch_factor

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.switch_epoch:  # epochs are 0-indexed
            self.factor = self.switch_factor
            print(f"\n>>> Switching ReduceLROnPlateau factor to {self.switch_factor}")
        super().on_epoch_end(epoch, logs)
