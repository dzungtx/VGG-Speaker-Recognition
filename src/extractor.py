from collections import namedtuple
import os
import numpy as np
import model
import utils


class Extractor:
    def __init__(self):
        netConfig = {
            'net': 'resnet34s',
            'ghost_cluster': 2,
            'vlad_cluster': 8,
            'bottleneck_dim': 512,
            'aggregation_mode': 'gvlad',
            'loss': 'softmax',
            'dim': (257, None, 1),
            'n_classes': 5994,
        }
        netConfig = namedtuple(
            "NetConfig", netConfig.keys())(*netConfig.values())

        self.net = model.vggvox_resnet2d_icassp(input_dim=netConfig.dim,
                                                num_class=netConfig.n_classes,
                                                mode='eval', args=netConfig)
        self.net.load_weights(os.path.join(
            '../model/gvlad_softmax/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5'), by_name=True)

    def process(self, filePath):
        fftConfig = {
            'nfft': 512,
            'spec_len': 250,
            'win_length': 400,
            'hop_length': 160,
            'sampling_rate': 16000,
        }
        specs = utils.load_data(filePath, win_length=fftConfig['win_length'], sr=fftConfig['sampling_rate'],
                                hop_length=fftConfig['hop_length'], n_fft=fftConfig['nfft'],
                                spec_len=fftConfig['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        return self.net.predict(specs)
