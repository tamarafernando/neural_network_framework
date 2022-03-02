class StandardScaler():
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

        # the standard deviation can be 0, which provokes devision-by-zero errors; let's avoid that:
        self.std[self.std == 0] = 0.00001

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x_scaled):
        return x_scaled * self.std + self.mean


class NormalScaler():
    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0) - self.min

    def transform(self, x):
        return (x - self.min) / self.max

    def inverse_transform(self, x_scaled):
        return x_scaled * self.max + self.min