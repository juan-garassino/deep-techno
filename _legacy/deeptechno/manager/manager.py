import argparse
import pretty_midi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from IPython import display
import numpy as np

class Manager:
    _SAMPLING_RATE = 44100  # Sample rate for audio playback

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--root", type=str, default='Code/juan-garassino/mySandbox/deepTechno', help="Root folder for the Maestro dataset or for custom data.")
        parser.add_argument("--data_dir", type=str, default="/dataset", help="Root folder for the Maestro dataset or for custom data.")
        parser.add_argument("--output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into.")
        parser.add_argument("--custom_dataset", action="store_true", help="Whether or not the specified root folder contains custom data.")

        return parser.parse_args()

    @staticmethod
    def display_audio_notebook(pm: pretty_midi.PrettyMIDI, seconds=30):
        waveform = pm.fluidsynth(fs=Manager._SAMPLING_RATE)
        # Take a sample of the generated waveform to mitigate kernel resets
        waveform_short = waveform[:seconds*Manager._SAMPLING_RATE]
        return display.Audio(waveform_short, rate=Manager._SAMPLING_RATE)

    @staticmethod
    def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
        if count:
            title = f'First {count} notes'
        else:
            title = f'Whole track'
            count = len(notes['pitch'])
        plt.figure(figsize=(20, 4))
        plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
        plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
        plt.plot(
            plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch')
        _ = plt.title(title)
