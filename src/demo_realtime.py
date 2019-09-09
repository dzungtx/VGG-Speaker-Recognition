from datetime import datetime as dt
from extractor import Extractor
from imutils import paths
from queue import Queue
import argparse
import numpy as np
import os
import shutil
import sounddevice as sd
import soundfile as sf
import tempfile
import time


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help='show list of audio devices and exit')
    parser.add_argument('-d', '--device', type=int_or_str,
                        help='input device (numeric ID or substring)')
    parser.add_argument('-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument('-c', '--channels', type=int, default=1,
                        help='number of input channels')
    parser.add_argument('filename', nargs='?', metavar='FILENAME',
                        help='audio file to store recording to')
    parser.add_argument('-t', '--subtype', type=str,
                        help='sound file subtype (e.g. "PCM_24")')
    parser.add_argument('-s', '--split-after', type=int, default=3,
                        help='split after this number of seconds')
    parser.add_argument('-th', '--threshold', type=float, default=0.85,
                        help='threshold for match score')
    args = parser.parse_args()

    detect_real_time(args)


def record_input(args):
    if args.list_devices:
        print(sd.query_devices())
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info['default_samplerate'])
    queue = Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, flush=True)
        queue.put(indata.copy())

    with sd.InputStream(samplerate=args.samplerate, device=args.device,
                        channels=args.channels, callback=callback):
        counter = 0
        start_time = time.time()
        now = dt.now()
        context = {'counter': counter, 'year': now.year, 'month': now.month,
                   'day': now.day, 'hour': now.hour, 'minute': now.minute, 'second': now.second}
        if args.filename is None:
            filename = tempfile.mktemp(prefix='rec_unlimited_%d_' % counter,
                                       suffix='.wav', dir='')
        else:
            filename = args.filename.format(**context)
        with sf.SoundFile(filename, mode='x', samplerate=args.samplerate,
                          channels=args.channels, subtype=args.subtype) as file:
            print("Recording to: " + repr(filename))
            while True:
                if time.time() - start_time > args.split_after:
                    start_time += args.split_after
                    counter += 1
                    break
                file.write(queue.get())


def record_input_voice(args):
    print('Enter Name')
    name = input("Enter name")
    print('Speech three times, each time 3 seconds')
    for i in range(3):
        filename = "dataset/" + name + "_" + str(i) + ".wav"
        if os.path.exists(filename):
            os.remove(filename)
        setattr(args, 'filename', filename)
        record_input(args)


def test_voice(args, data, extractor):
    filename = 'dataset/test_voice.wav'
    record_test_voice(args, filename)
    start_time = time.time()
    test_voice_result = extractor.process(filename)
    match_name = None
    max_score = 0
    for name, values in data.items():
        for value in values:
            match_score = np.sum(value*test_voice_result)
            if match_score > max_score:
                max_score = match_score
                match_name = name
            print("Score with: " + name + " - ", match_score)
    if max_score >= args.threshold:
        print('=======================================================')
        print('Match: ', match_name)
        print('=======================================================')
    else:
        print('=======================================================')
        print('Not match')
        print('=======================================================')
    print("[INFO] done in {} seconds".format(
        round(time.time() - start_time, 2)))


def load_dataset(args, extractor):
    data = {}
    start_time = time.time()
    print("[INFO] Loading dataset...")
    records = paths.list_files('dataset', validExts=('.wav'))
    for record in records:
        if record != 'dataset/test_voice.wav':
            name = record.split(os.path.sep)[-1].split('_')[0]
            if name not in data.keys():
                data[name] = []
            data[name].append(extractor.process(record))
    print("[INFO] done in {} seconds".format(
        round(time.time() - start_time, 2)))
    return data


def record_test_voice(args, filename):
    if os.path.exists(filename):
        os.remove(filename)
    setattr(args, 'filename', filename)
    record_input(args)


def load_extractor():
    start_time = time.time()
    print("[INFO] Loading model...")
    extractor = Extractor()
    print("[INFO] done in {} seconds".format(
        round(time.time() - start_time, 2)))
    return extractor


def detect_real_time(args):
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    extractor = load_extractor()
    data = load_dataset(args, extractor)

    while True:
        print('Choose mode:')
        print('1. Input voice')
        print('2: Test voice')
        print('3: Clear dataset')
        print('4: Finish')
        mode = input("Enter mode")
        if mode == '1':
            record_input_voice(args)
            data = load_dataset(args, extractor)
        elif mode == '2':
            if len(data.keys()) == 0:
                print('Please record input voice first')
            else:
                print("[INFO] Testing voice...")
                print("[INFO] Press `Ctrl + c` to escape")
                while True:
                    try:
                        test_voice(args, data, extractor)
                    except KeyboardInterrupt:
                        break
        elif mode == '3':
            if os.path.exists('dataset'):
                shutil.rmtree('dataset')
            data = {}
        else:
            print('Finishing...')
            break


if __name__ == '__main__':
    main()
