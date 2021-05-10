import importlib
import numpy as np
import matplotlib.pylab as plt
import scipy.signal, scipy.stats
from sklearn.cross_decomposition import CCA
import fun
importlib.reload(fun)

class CCA_SSVEP_kernel_refs:
    def __init__(self, signal_shape, kernels=[], n_jobs=1):
        self.signal_shape = signal_shape
        self.cca = CCA(n_components=1)
        self.kernels = kernels
        self.n_features = len(kernels)
        self.n_jobs = n_jobs

    def transform(self, X):
        if X.ndim > 1:
            n_samples = X.shape[0]
            f = np.zeros([n_samples, self.n_features])
            if self.n_jobs == 1:
                n_samples = X.shape[0]
                for ii in range(n_samples):
                    _, _, f[ii, :] = self.transform1(X[ii])
            else:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=self.n_jobs, verbose=0)([delayed(self.transform1)(sig) for sig in X])
                for ii, r in enumerate(results):
                    f[ii, :] = r[2]
            return f
        else:
            return self.transform1(X)

    def transform1(self, X_):
        X = np.reshape(X_, self.signal_shape)
        f = np.zeros(self.n_features)
        Xc = np.zeros([self.signal_shape[1], len(self.kernels)])
        Yc = np.zeros([self.signal_shape[1], len(self.kernels)])
        for ii in range(self.n_features):
            xc, yc = self.cca.fit(X.T, self.kernels[ii].T).transform(X.T, self.kernels[ii].T)
            Xc[:, ii] = xc.ravel()
            Yc[:, ii] = yc.ravel()
            f[ii] = np.corrcoef(xc.ravel(), yc.ravel())[1, 0]
            # f[ii] = np.dot(yc.T, yc)
        return Xc, Yc, f

def generate_kernels(times, fs, freqs, shift=0, wl=None, kernel='rbf', scale=.01):
    n_times = times.shape[0]
    kernels = []
    if shift == 0:
        shift = 1. / fs
    for freq in freqs:
        # _kernel = np.zeros([n_kernels, n_times])
        # for time_idx1, time1 in enumerate(np.linspace(0., 1. / freq, n_kernels + 1)[:-1]):
        time_centers = np.arange(0., 1. / freq, shift)
        _kernel = np.zeros([time_centers.shape[0], n_times])
        for time_idx1, time1 in enumerate(time_centers):
            # peaks = np.arange(-times[-1] + time1, times[-1], 1. / freq)
            peaks = np.arange(times[0] + time1, times[-1], 1. / freq)
            _times = np.tile(times[:, np.newaxis], [1, len(peaks)]) - peaks[np.newaxis, :]
            if kernel == 'rbf':
                kernel_wav = scipy.stats.norm.pdf(_times, scale=scale).sum(1)
            elif kernel == 'rect':
                kernel_wav = np.zeros([1, n_times])
                for peak_idx in range(len(peaks)):
                    kernel_wav[0, np.abs(_times[:, peak_idx]) <= scale] = 1.
            if wl:
                window = scipy.stats.norm.pdf(times - time1, scale=wl)
                kernel_wav *= window
            # _kernel[time_idx1, :] = kernel_wav / np.sqrt(np.dot(kernel_wav, kernel_wav.T))
            _kernel[time_idx1, :] = kernel_wav
        kernels += [_kernel]
    return kernels

def generate_multi_scale_kernels(times, fs, freqs, scales=[.01], shifts=[0], wl=None, kernel='rbf'):
    for ii, scale in enumerate(scales):
        _kernels = generate_kernels(times, fs, freqs, shifts[ii], wl, kernel, scale)
        if ii == 0:
            kernels = _kernels
        else:
            for jj in range(len(freqs)):
                kernels[jj] = np.vstack([kernels[jj], _kernels[jj]])
    return kernels

if __name__=='__main__':

    # =============================
    # Generate SSVEP samples
    # =============================
    n_samples_per_class = 10
    n_channels = 4
    # fs = 250.
    fs = 1000.
    # freqs = np.array([8., 8.2, 8.4])
    freqs = np.array([10, 13, 15])
    # freqs = np.arange(8, 15, .2)
    duration = 1.
    times = np.arange(0., duration, 1. / fs)
    n_times = times.shape[0]
    n_freqs = len(freqs)
    channel_amplitudes = np.random.uniform(.1, 1, [n_channels, 1])
    signal = np.zeros([n_channels, n_times])
    n_classes = len(freqs)
    y = np.reshape(np.tile(np.arange(n_classes)[np.newaxis], [1, n_samples_per_class]), n_classes * n_samples_per_class)
    n_samples = len(y)
    signals = np.zeros([n_samples, n_channels, n_times])
    source = []
    for sample_idx, class_ in enumerate(y):
        a = np.sin(2 * np.pi * freqs[class_] * times)[np.newaxis, :]
        a = np.zeros([1, n_times])
        for st in np.arange(0, times[-1], 1./freqs[class_]):
            s = scipy.signal.gausspulse(times - st, fc=30, bw=.5)[np.newaxis, :]
            s[0, np.where(times < st)[0]] = 0.
            a += s
        source += [a]
        signals[sample_idx] = np.dot(channel_amplitudes, a)
    signals += .5 * np.random.standard_normal([n_samples, n_channels, n_times])
    plt.figure(1).clf()
    plt.subplot(211)
    for ff, freq in enumerate(freqs):
        plt.plot(times, source[ff].T, label=freq)
    plt.ylabel('SSVEP components')
    plt.legend()
    plt.subplot(212)
    plt.plot(times, signals[0][0].T)
    plt.plot(times, signals[1][0].T)
    plt.plot(times, signals[3][0].T)
    plt.ylabel('Observed signals')
    plt.xlabel('Time [s]')

    signal_shape = [n_channels, n_times]
    x = np.reshape(signals, [n_samples, n_channels * n_times])
    # x: vectroized signals for all trials. n_samples x (n_channels * n_times)
    # y: class labels. n_samples

    # =============================
    # Make and show reference signals for CCA-with-kernels
    # =============================
    scales = [1 / fs * .5] # The width of a kernel
    shift = [1 / fs] # The shift length between kernels
    kernels = generate_multi_scale_kernels(times, fs, freqs, scales=scales, shifts=shift, wl=0, kernel='rect')
    cck = CCA_SSVEP_kernel_refs(signal_shape, kernels, n_jobs=-1)
    plt.figure(2).clf()
    for kk, kernel in enumerate(kernels):
        plt.subplot(len(kernels), 1, kk + 1)
        plt.plot(times, kernel.T)
        plt.xlim([0, .1])
        plt.ylabel('{} Hz'.format(freqs[kk]))
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('figs/kernels.pdf', transparent=True, bbox_inches='tight')

    plt.figure(3).clf()
    plt.tight_layout()
    plt.subplot(len(kernels), 1, 1)
    plt.plot(times, kernels[0][0, :].T)
    plt.xlim([times[0], times[-1]])
    plt.xlabel('Time [s]')
    plt.savefig('figs/kernel00.pdf', transparent=True, bbox_inches='tight')

    Xc, Yc, f = cck.transform1(x[0])
    plt.figure(4).clf()
    plt.subplot(211)
    plt.plot(times, Xc)
    plt.title('Spatial-filtered signal')
    plt.subplot(212)
    plt.plot(times, Yc)
    plt.title('Combinated Reference signal')

    # =============================
    # Make and show reference signals for the standard CCA method
    # =============================
    cca_refs = fun.generate_CCA_refs(freqs, times, n_harmonics=1)
    cca = fun.CCA_SSVEP(Y=cca_refs, signal_shape=signal_shape)
    plt.figure(5).clf()
    for ff, freq in enumerate(freqs):
        plt.subplot(len(freqs), 1, ff + 1)
        plt.plot(times, cca_refs[ff].T)
        plt.xlim([0, .1])
        plt.ylabel('{} Hz'.format(freq))
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('figs/sincos.pdf', transparent=True, bbox_inches='tight')

    plt.figure(6).clf()
    plt.tight_layout()
    plt.subplot(len(freqs), 1, 1)
    plt.plot(times, cca_refs[0][0, :].T)
    plt.xlim([times[0], times[-1]])
    plt.xlabel('Time [s]')
    plt.savefig('figs/sincos00.pdf', transparent=True, bbox_inches='tight')

    # =============================
    # Classification accuracy in the artificially-generated samples
    # =============================

    Fk = cck.transform(x)
    clf = fun.max_min_classifier(criterion='max')
    clf.fit(Fk, y)
    acc = clf.score(Fk, y)
    print('CCK Acc: {:.1f}%%'.format(acc * 100))

    Fa = cca.transform(x)
    clf = fun.max_min_classifier(criterion='max')
    clf.fit(Fa, y)
    acc = clf.score(Fa, y)
    print('CCA Acc: {:.1f}%%'.format(acc * 100))

    plt.show()
