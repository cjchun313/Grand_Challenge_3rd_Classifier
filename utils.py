import glob
import tqdm
import numpy as np
import librosa
import os


SAMPLING_FREQUENCY = 8000
SAMPLE_LEN = int(SAMPLING_FREQUENCY * 6.1)

def read_audio(filepath, target_fs=SAMPLING_FREQUENCY, sample_len=SAMPLE_LEN):
    audio, sr = librosa.load(filepath, sr=target_fs, mono=True)
    audio_len = len(audio)

    if audio_len < SAMPLE_LEN:
        zero_sample = np.zeros([sample_len - audio_len])
        audio = np.concatenate((audio, zero_sample), axis=0)
    else:
        audio = audio[:SAMPLE_LEN]

    return audio, sr


def gen_melspectrogram(filepath, n_mels=128, target_fs=SAMPLING_FREQUENCY, sample_len=SAMPLE_LEN):
    audio, sr = read_audio(filepath, target_fs=target_fs, sample_len=sample_len)

    # Passing through arguments to the Mel filters
    #L = librosa.feature.melspectrogram(y=audio[0], sr=sr, n_mels=n_mels, fmin=200, fmax=4000)
    #R = librosa.feature.melspectrogram(y=audio[1], sr=sr, n_mels=n_mels, fmin=200, fmax=4000)

    # mel = np.concatenate((L.reshape(1, n_mels, -1), R.reshape(1, n_mels, -1)), axis=0)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmin=200, fmax=4000)

    return mel

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

def create_npz_dataset(read_path, write_path):
    wavfiles = sorted(glob.glob(read_path))
    print('the number of wav files:', len(wavfiles))

    data = []
    for wavfile in tqdm.tqdm(wavfiles):
        mel = gen_melspectrogram(wavfile)
        data.append(mel)

    data = np.array(data, dtype=np.float32)
    print('data :', data.shape)

    np.savez_compressed(write_path, x=np.array(data, dtype=np.float32))
    print('done!')

if __name__ == "__main__":
    train_child_filepath = '../db/stereo_after_rir/train/*/*/child/*/*.wav'
    train_male_filepath = '../db/stereo_after_rir/train/*/*/male/*/*.wav'
    train_female_filepath = '../db/stereo_after_rir/train/*/*/female/*/*.wav'

    val_child_filepath = '../db/stereo_after_rir/val/*/*/child/*/*.wav'
    val_male_filepath = '../db/stereo_after_rir/val/*/*/male/*/*.wav'
    val_female_filepath = '../db/stereo_after_rir/val/*/*/female/*/*.wav'

    create_npz_dataset(train_child_filepath, '../db/stereo_after_rir/train_child_8k_mono.npz')
    create_npz_dataset(train_male_filepath, '../db/stereo_after_rir/train_male_8k_mono.npz')
    create_npz_dataset(train_female_filepath, '../db/stereo_after_rir/train_female_8k_mono.npz')

    create_npz_dataset(val_child_filepath, '../db/stereo_after_rir/val_child_8k_mono.npz')
    create_npz_dataset(val_male_filepath, '../db/stereo_after_rir/val_male_8k_mono.npz')
    create_npz_dataset(val_female_filepath, '../db/stereo_after_rir/val_female_8k_mono.npz')

    '''
    wavfiles = sorted(glob.glob(filepath))
    print(len(wavfiles))
    for wavfile in wavfiles:
        audio, _ = read_audio(wavfile)
        break
    '''