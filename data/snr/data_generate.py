import soundfile as sf
import numpy as np
import pandas as pd
import os


def cal_avg_power(signals):
    powers = []
    for signal in signals:
        power = np.sum(signal**2, axis=1)
        powers.append(power/signal.shape[1])

    return powers


def data_generate(save_dir, src_paths, noise_path, Normalize, dir_name, num_src=1, SNR=0, overlap_ratio=0.5):
    srcs = []
    # load source signals
    for src_path in src_paths:
        src, sr = sf.read(os.path.join(src_path, 'mono_minmax_no_silence/hal_in_pure_24_4ch_48k_' + str(1) + '.wav'))
        src = src[np.newaxis, :]
        for channel in range(3):
            data, _ = sf.read(os.path.join(src_path, 'mono_minmax_no_silence/hal_in_pure_24_4ch_48k_' + str(channel + 2) + '.wav'))
            src = np.concatenate([src, data[np.newaxis, :]], axis=0)
        srcs.append(src)
    
    if num_src > 1: # overlap 50%
        overlap_idx = np.abs(srcs[0].shape[1] - int((srcs[0].shape[1] + srcs[1].shape[1])/((1/overlap_ratio)+1)))
        length = srcs[1].shape[1] + overlap_idx
    else:
        _, length = srcs[0].shape

    # load noise signal
    noise, _ = sf.read(os.path.join(noise_path, 'mono_minmax_no_silence/hal_in_pure_24_4ch_48k_' + str(1) + '.wav'))
    noise = noise[np.newaxis, :]
    for channel in range(3):
        data, _ = sf.read(os.path.join(noise_path, 'mono_minmax_no_silence/hal_in_pure_24_4ch_48k_' + str(channel + 2) + '.wav'))
        noise = np.concatenate([noise, data[np.newaxis, :]], axis=0)
    # random_idx = random.randint(0, noise.shape[1]-length)
    # noise = noise[:, random_idx:random_idx+length]


    # SNR : channel 1을 기준으로 계산해주었습니다.
    # 각각의 power 구하기
    s_power = cal_avg_power(srcs)
    # source 2를 source1과 power가 같아지게 맞춰주고 overlap 약 50%로 합치기
    if num_src > 1:
        srcs[1] = srcs[1]*((s_power[0][0]/s_power[1][0])**(1/2))
        srcs[0] = np.pad(srcs[0], ((0,0),(0, length - srcs[0].shape[1]))) + np.pad(srcs[1], ((0,0),(overlap_idx, 0)))
    # source 1과 noise 사이의 비 선택 (입력된 SNR 값에 따라)해서 맞춰주고 노이즈 더하기
    if length < noise.shape[1]: # 길이에 맞추어 noise 자르기
        noise = noise[:, :length]
    else: # 짧을 경우, noise 앞 부분을 위에 붙여주었습니다...
        noise = np.concatenate([noise, noise], axis=1)[:, :length]
    n_power = cal_avg_power([noise])
    noise = noise*((s_power[0][0]/n_power[0][0]/(10**(SNR/10)))**(1/2))

    output = srcs[0] + noise
    # 전체 합쳐진 신호가 처음 source 1의 power와 같아지도록 normalize
    if Normalize:
        power = cal_avg_power([output])
        output = output*((s_power[0][0]/power[0][0])**(1/2))

    if not os.path.isdir(os.path.join(save_dir, str(num_src) + 'mix')):
        os.mkdir(os.path.join(save_dir, str(num_src) + 'mix'))
    if not os.path.isdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize))):
        os.mkdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize)))
    if not os.path.isdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'SNR_' + str(SNR))):
        os.mkdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'SNR_' + str(SNR)))
    if not os.path.isdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'SNR_' + str(SNR), dir_name)):
        os.mkdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'SNR_' + str(SNR), dir_name))
    
    if num_src > 1:
        if not os.path.isdir(os.path.join(save_dir, '2mix', 'Normalize_' + str(Normalize), 'Clean')):
            os.mkdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'Clean'))
        if not os.path.isdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'Clean', dir_name)):
            os.mkdir(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'Clean', dir_name))
            for channel in range(4):
                sf.write(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'Clean', dir_name, 'hal_in_pure_24_4ch_48k_' + str(channel + 1) +'.wav'), srcs[0][channel], sr)
    
    
    for channel in range(4):
        sf.write(os.path.join(save_dir, str(num_src) + 'mix', 'Normalize_' + str(Normalize), 'SNR_' + str(SNR), dir_name, 'hal_in_pure_24_4ch_48k_' + str(channel + 1) + '+'
                               + noise_path.split('/')[-1] +'.wav'), output[channel], sr)



def make_1mix(meta, dataset_dir, save_dir, noise, SNR, Normalize):
    
    for dir in meta['name']:
        src_path = [os.path.join(dataset_dir, dir)]
        noise_path = 'Z:/Database/삼성과제/noise' + str(noise)

        data_generate(save_dir, src_path, noise_path, Normalize, dir, 1, SNR)



def make_Nmix(meta, dataset_dir, save_dir, noise, SNR, Normalize):
    noise_path = 'Z:/Database/삼성과제/noise' + str(noise)

    for i in range(np.max(df['pair'])):
        paths = df['name'][df['pair']==i+1].reset_index(drop=True).to_list()
        src_paths = [os.path.join(dataset_dir, dir) for dir in paths]
        num_srcs = len(src_paths)

        dir_name = paths[0]
        for dir in paths[1:]:
            dir_name = dir_name + '+' + dir
        data_generate(save_dir, src_paths, noise_path, Normalize, dir_name, num_srcs, SNR)
    


if __name__ == '__main__':

    file_name = 'meta_240625.xlsx'
    df = pd.read_excel(file_name)

    save_dir = 'Z:/Database/삼성과제/_Mixture' # 저장 위치
    dataset_dir = 'Z:/Database/삼성과제' # 공유폴더 삼성과제 데이터 위치

    SNR = 10 # SNR (dB) = 10 log (signal_power / Noise_power); [0, 5, 10]
    for noise in range(4):
        # make_1mix(df, dataset_dir, save_dir, noise=noise + 1, SNR=SNR, Normalize = False)
        make_Nmix(df, dataset_dir, save_dir, noise=noise + 1, SNR=SNR, Normalize = False)
    