# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv


def record_save_audio(wave_output_filename,
                      freq=16000,
                      record_seconds=2):

    # Start recorder with the given values of duration and sample frequency
    recording = sd.rec(int(record_seconds * freq),
                       samplerate=freq,
                       channels=1)

    print("* recording")

    # Record audio for the given number of seconds
    sd.wait()

    print("* done recording")

    # This will convert the NumPy array to an audio file with the given sampling frequency
    write(wave_output_filename, freq, recording)

    # Convert the NumPy array to audio file
#    wv.write(wave_output_filename, recording, freq, sampwidth=2)
