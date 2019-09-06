from extractor import Extractor
import numpy as np


def main():
    e = Extractor()
    v1 = e.process(
        '/home/dzung/data/voice_datatset/dzung_2708/1.wav')
    v2 = e.process(
        '/home/dzung/data/voice_datatset/dzung_2708/2.wav')
    print(np.sum(v1*v2))

    return


if __name__ == "__main__":
    main()
