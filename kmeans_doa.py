import glob
import tqdm
import numpy as np
import librosa
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
from scipy.stats import rankdata


class KMeans_DOA:
    def __init__(self, num_degrees=10, n_init=1024, max_iter=1024, sr=8000, fmin=400, fmax=3300):
        self.num_degrees = num_degrees
        self.n_init = n_init
        self.max_iter = max_iter
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.azimuth_resol = int(180.0 / (self.num_degrees - 1))

    def fit_doa(self, power_ratio):
        model = KMeans(n_clusters=self.num_degrees, init="random", n_init=1024, max_iter=1024, random_state=0).fit(power_ratio)
        c = model.cluster_centers_.reshape(-1)
        label = rankdata(c).astype(int)
        print(c)
        print(rankdata(c).astype(int))

        pred_degrees = np.array(label[model.labels_] - 1, dtype=np.int)
        pred_degrees *= self.azimuth_resol
        print(pred_degrees.shape)

        return pred_degrees

    def compute_power_ratio(self, filepath):
        wavfiles = sorted(glob.glob(filepath))
        print('the number of wavfiles:', wavfiles)

        power_ratio = []
        for wavfile in tqdm.tqdm(wavfiles):
            audio, _ = self.read_stereo_audio(wavfile, target_fs=self.sr, sample_len=int(self.sr*6.1))

            audio_L = audio[0]
            audio_R = audio[1]
            audio_L = self.butter_bandpass_filter(audio_L, self.fmin, self.fmax, self.sr, 9)
            audio_R = self.butter_bandpass_filter(audio_R, self.fmin, self.fmax, self.sr, 9)

            PL = np.sum(np.abs(audio_L))
            PR = np.sum(np.abs(audio_R))
            if PL > PR:
                power_ratio.append(PR / (PL + 1e-4))
            else:
                power_ratio.append(2.0 - (PL / (PR + 1e-4)))

        power_ratio = np.array(power_ratio, dtype=np.float32).reshape(-1, 1)

        return power_ratio

    def display_scatter(self, power_ratio, pred_degrees):
        plt.figure(figsize=(8, 8))
        for i in range(self.num_degrees):
            plt.scatter(power_ratio[pred_degrees == int(i * self.azimuth_resol)], pred_degrees[pred_degrees == int(i * self.azimuth_resol)], marker='o', s=100)
        plt.show()

    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)

        return y

    def read_stereo_audio(self, filepath, target_fs, sample_len):
        audio, sr = librosa.load(filepath, sr=target_fs, mono=False)
        if audio.ndim <= 1:
            print('wav file is not stereo!')
        audio_len = np.size(audio, 1)

        if audio_len < sample_len:
            zero_sample = np.zeros([2, sample_len - audio_len])
            audio = np.concatenate((audio, zero_sample), axis=1)
        else:
            audio = audio[:, :sample_len]

        return audio, sr

if __name__ == "__main__":
    doa = KMeans_DOA()
    pr = doa.compute_power_ratio('../db/stereo_after_rir/val/5m/*/*/*/*.wav')

    pred_degrees = doa.fit_doa(pr)

    doa.display_scatter(pr, pred_degrees)
