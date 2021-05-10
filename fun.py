import numpy as np
from sklearn.cross_decomposition import CCA
import scipy.signal as sig
from sklearn.base import BaseEstimator
import sklearn.metrics

class bandpass_filter:

    def __init__(self, bandpass=None, fs=None, shape=None, filter_order=4):
        self.bandpass = bandpass
        self.fs = fs
        self.shape = shape
        self.filter_order = filter_order

    # def fit(self, X, y):
        # pass

    def transform(self, X):
        if X.ndim == 1:
            X = X[np.newaxis, :]
        n_samples = X.shape[0]
        X_shaped = np.reshape(X, [n_samples, self.shape[0], self.shape[1]])
        fs = self.fs
        bandpass = self.bandpass
        if bandpass:
            wn = np.array(bandpass)/(.5*fs)
            b, a = sig.iirfilter(self.filter_order, wn, btype='band', ftype='butter')
            X_filtered = sig.filtfilt(b, a, X_shaped, 2)

        return np.reshape(X_filtered, [n_samples, np.prod(self.shape)])

    def fit_transform(self, X, y):
        X = self.transform(X)
        return X

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def get_params(self, **params):
        return {'bandpass':self.bandpass, 'fs':self.fs, 'shape':self.shape}

class base_line_adjustment:
    def __init__(self, base_period=[], shape=None):
        self.base_period = base_period # [sample]
        self.shape = shape

    def fit(self, X, y):
        pass

    def transform(self, X):
        if self.base_period.ndim > 1:
            n_samples = X.shape[0]
            X_shaped = np.reshape(X, [n_samples, self.shape[0], self.shape[1]])
            X_transed = np.zeros([n_samples, self.shape[0], self.shape[1]], dtype=float)
            for ii in range(n_samples):
                epoch_bl = X_shaped[ii, :, self.base_period]
                bl = epoch_bl.mean(0)[:, np.newaxis]
                X_transed[ii, :, :] = X_shaped[ii, :, :] - np.tile(bl, (1, self.shape[1]))
            return np.reshape(X_transed, [n_samples, np.prod(self.shape)])

        return X

    def fit_transform(self, X, y):
        X = self.transform(X)
        return X

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def get_params(self, **params):
        return {"base_period":self.base_period, "shape":self.shape}

class time_window:
    def __init__(self, window, shape):
        self.window = window 
        self.shape = shape
        self.transformed_shape = list(shape[:])
        self.transformed_shape[1] = self.window.shape[0]

    def fit(self, X, y):
        pass

    def transform(self, X):
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if self.window.ndim > 0:
            n_samples = X.shape[0]
            X_shaped = np.reshape(X, [n_samples, self.shape[0], self.shape[1]])
            X_windowed = X_shaped[:, :, self.window]
            return np.reshape(X_windowed, [n_samples, self.shape[0]*self.window.shape[0]])
        return X

    def fit_transform(self, X, y):
        X = self.transform(X)
        return X

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def get_params(self, **params):
        return {'window':self.window,
                'shape':self.shape,
                'transformed_shape':self.transformed_shape,
                }

class down_sample:
    def __init__(self, down_factor=1, shape=[]):
        self.down_factor = down_factor
        self.shape = shape

    def fit(self, X, y):
        pass

    def transform(self, X):
        n_samples = X.shape[0]
        if self.down_factor > 1:
            X_shaped = np.reshape(X, [n_samples, self.shape[0], self.shape[1]])
            X_transed = X_shaped[:, :, 0:-1:self.down_factor]
            return np.reshape(X_transed, [n_samples, np.prod(X_transed.shape[1:])])
        return X

    def fit_transform(self, X, y):
        X = self.transform(X)
        return X

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def get_params(self, **params):
        return {"down_factor":self.down_factor}

class CCA_SSVEP:
    def __init__(self, Y=[], signal_shape=[], S=[]):
        self.Y = Y
        self.n_Y = len(Y)
        self.cca = CCA(n_components=1)
        self.S = S
        self.signal_shape = signal_shape

    def fit(self, X, y):
        return self

    def transform(self, X):
        if len(X.shape) > 1:
            n_samples = X.shape[0]
            Y = np.zeros([n_samples, self.n_Y])
            for ii in range(n_samples):
                Y[ii, :] = self.transform1(X[ii])
            return Y
        else:
            return self.transform1(X)

    def transform1(self, X_):
        X = np.reshape(X_, self.signal_shape)
        if len(self.S) > 0:
            X = np.dot(self.S.T, X)
        features = np.zeros([1, self.n_Y])
        self.set_x_weights_ = []
        self.set_y_weights_ = []
        for ii in range(self.n_Y):
            Xc, Yc = self.cca.fit(X.T, self.Y[ii].T).transform(X.T, self.Y[ii].T)
            features[0, ii] = np.corrcoef(Xc.ravel(), Yc.ravel())[1, 0]
            if len(self.S) > 1:
                self.set_x_weights_.append(np.dot(self.S, self.cca.x_weights_))
            else:
                self.set_x_weights_.append(self.cca.x_weights_)
            self.set_y_weights_.append(self.cca.y_weights_)
        return features.ravel()

def generate_CCA_ref(times, reference_freq, n_harmonics):
    Y = []
    for ii in range(n_harmonics+1):
        if reference_freq * (ii + 1) < 90.:
            Y += [np.sin(2*np.pi*reference_freq*times*(ii+1))]
            Y += [np.cos(2*np.pi*reference_freq*times*(ii+1))]
    return np.array(Y).squeeze()

def generate_CCA_refs(freqs, times, n_harmonics):
    Y = []
    for freq_idx, freq in enumerate(freqs):
        Y.append(generate_CCA_ref(times, freq, n_harmonics))
    return Y

class max_min_classifier(BaseEstimator):

    def __init__(self, criterion='max'):
        self.criterion=criterion

    def fit(self, X, y):
        self.class_labels = np.sort(np.unique(y))
        return self

    def predict(self, X):
        return self.class_labels[np.argmax(X, 1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return sklearn.metrics.accuracy_score(y, y_pred)

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, **params):
        return {'criterion':self.criterion}

def channel_labels2indexs(channel_labels, selected_labels):
    idxs = []
    channel_labels = list(channel_labels)
    for label in selected_labels:
        idxs.append(channel_labels.index(label))
    return idxs