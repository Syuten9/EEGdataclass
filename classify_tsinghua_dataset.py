import importlib
import numpy as np
import scipy.io
import fun
import sklearn.model_selection as model_selection
from sklearn.svm import LinearSVC
import sklearn.linear_model
import scipy.signal


def load_tsinghua(data_dir, dataset_name, channel_idxs):
    data_all = scipy.io.loadmat('%s/%s.mat' % (data_dir, dataset_name))
    data = data_all['data'] # electrodes x time_points x targets x blocks
    y = np.reshape(np.tile(np.arange(data.shape[2])[:, np.newaxis], [1, data.shape[3]]), np.prod(data.shape[2:]))
    data = np.transpose(
            np.reshape(data, [data.shape[0], data.shape[1], np.prod(data.shape[2:])]), [2, 0, 1]) # trials x channels  x times
    # n_samples = data.shape[0]
    data = data[:, channel_idxs]
    signal_shape = data.shape[1:]
    fs = 250. # Hz
    # times = 1.*np.arange(signal_shape[1])/fs
    times = np.arange(-.5, 5.5, 1./fs)
    return data, y, fs, times, signal_shape

def select_class_samples(X, y, selected_classes, random=False):
    selected_samples = np.zeros([0], dtype=int)
    for selected_class in selected_classes:
        selected_samples = np.hstack([selected_samples, np.where(y == selected_class)[0]])
    if random:
        selected_samples = selected_samples[np.random.permutation(selected_samples.shape[0])]
    X = X[selected_samples]
    y = y[selected_samples]
    return X, y

def tsinghua_channel_labels():
    return ['FP1', 'FPz', 'FP2', 'AF3', 'AF4',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'M1',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'M2',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'CB1',
            'O1', 'Oz', 'O2',
            'CB2',
            ]


if __name__=='__main__':

    dataset_dir = r'\lab\Bciinfo\dataset\Benchmarkdataset' # change the datapath for your own env.
    dataset_names = ['S1', 'S2', 'S3', 'S4', 'S6', 'S7', 'S8', 'S9', 'S10',
                    'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                    'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
                    'S31', 'S32', 'S33', 'S34', 'S35'
                    ]
    data_all = scipy.io.loadmat(dataset_dir + '/Phase.mat')
    freqs = data_all['freqs'].squeeze()
    phases = data_all['phases'].squeeze()
    channel_labels = tsinghua_channel_labels()
    channel_idxs = fun.channel_labels2indexs(channel_labels, ['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2'])
    selected_classes = range(40)
    # selected_classes = range(8)
    print('Target SSVEP freqs:', freqs[selected_classes], 'Hz')
    band = [7, 70] # bandpass filter
    downed_fs = 250 # down sampling
    time_range = [.14, 2.14] # time window
    n_harmonics = 5 # #harmonics for CCA reference signals
    classifier = 'max_min_classifier' # max(roh), unsupervised classifier
    # classifier = 'linear-svm_leave-one-out_CV' # SVM, supervised classifier
    print('%d classes, [%d, %d] Hz, %d Hz, [%.2f, %.2f] ms, %d channels by CCA and %s' % (
        len(selected_classes), band[0], band[1], downed_fs, time_range[0], time_range[1], len(channel_idxs), classifier))

    # preprocessing functions
    X_shaped, y, fs, times, signal_shape = load_tsinghua(dataset_dir, dataset_names[0], channel_idxs)
    bpf = fun.bandpass_filter(bandpass=band, fs=fs, shape=signal_shape, filter_order=9)
    down_factor = int(fs/downed_fs)
    time_window = np.arange(
            np.max(np.where(times <= np.max([times[0], time_range[0]]))[0]),
            np.min(np.where(times >= np.min([times[-1], time_range[1]]))[0]),
            down_factor)
    trans_fs = fs/down_factor
    twf = fun.time_window(window=time_window, shape=signal_shape)

    # feature extraction by CCA
    cca_refs = fun.generate_CCA_refs(freqs[selected_classes], times[twf.window], n_harmonics=n_harmonics)
    cca = fun.CCA_SSVEP(Y=cca_refs, signal_shape=twf.transformed_shape)

    scores = []
    for dataset_idx, dataset_name in enumerate(dataset_names):
        X_shaped, y, fs, times, signal_shape = load_tsinghua(dataset_dir, dataset_name, channel_idxs)
        X = np.reshape(X_shaped, [X_shaped.shape[0], np.prod(signal_shape)])
        X, y = select_class_samples(X, y, selected_classes, random=True)
        X_ = cca.transform(twf.transform(bpf.transform(X)))

        # score evaluation
        if classifier == 'max_min_classifier':
            clf = fun.max_min_classifier(criterion='max')
            clf.fit(X_, y)
            scores += [clf.score(X_, y)]
        elif classifier == 'linear-svm_leave-one-out_CV':
            clf = LinearSVC()
            scores += [model_selection.cross_val_score(clf, X_, y, cv=model_selection.LeaveOneOut(), n_jobs=-1).mean()]
        print('%s: %03.2f' % (dataset_name, scores[-1].mean()*100))

    scores = np.array(scores)
    r_txt = '[%s] %d-class, Ave.: %.2f%%' % (model_selection, len(selected_classes), scores.mean()*100)
    print(r_txt)
