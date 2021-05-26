
import os
import re

import numpy as np
import soundfile as sf
from dataclasses import dataclass
from tensorflow_tts.processor import BaseProcessor
from tensorflow_tts.utils import cleaners

valid_symbols = ["A", "A0", "U", "U0", "Y", "Y0", "I", "I0", "E0","O0", "K", "K0", "KH", "KH0", "G", "G0", "G", "GH0", "J0", "TSH", "TSH0", "SH0", "SH", "ZH", "ZH0", "DZ", "DZ0", "DZH", "DZH0", "R0", "R", "T", "T0", "TS", "TS0", "Z", "Z0", "S", "S0", "D", "D0", "N", "N0", "L", "L0", "P", "P0", "F", "F0", "B", "B0", "V", "V0", "M", "M0"]

_pad = "pad"
_eos = "eos"
_punctuation = "!'(),.? "
_special = "-"
_phonemes = ["@" + i for i in valid_symbols]

# Export all symbols:
RUSLAN_SYMBOLS = (
        [_pad] + list(_special) + list(_punctuation) + _phonemes + [_eos]
)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


@dataclass
class RuslanProcessor(BaseProcessor):
    """Thorsten processor."""

    cleaner_names: str = "russian_cleaners"
    positions = {
        "wave_file": 0,
        "text_norm": 1,
    }
    train_f_name: str = "metadata.csv"

    def create_items(self):
        if self.data_dir:
            with open(
                    os.path.join(self.data_dir, self.train_f_name), encoding="utf-8"
            ) as f:
                self.items = [self.split_line(self.data_dir, line, "|") for line in f]

    def split_line(self, data_dir, line, split):
        parts = line.strip().split(split)
        wave_file = parts[self.positions["wave_file"]]
        text_norm = parts[self.positions["text_norm"]]
        wav_path = os.path.join(data_dir, "wavs", f"{wave_file}.wav")
        speaker_name = "ruslan"
        return text_norm, wav_path, speaker_name

    def setup_eos_token(self):
        return _eos

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path)
        audio = audio.astype(np.float32)
        print(audio.shape[0] / rate)
        if audio.shape[1] > 1:
            audio = audio.sum(axis=1) / audio.shape[1]

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_path)[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text, grapheme=False):
        # change due to russian phonems format
        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        text = self._clean_text(text, [self.cleaner_names])
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self._symbols_to_sequence(
                    text
                )
                break
            sequence += self._symbols_to_sequence(m.group(1))
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        # add eos tokens
        sequence += [self.eos_id]
        return sequence


    def _clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
            text = cleaner(text)
        return text


    def _symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(["@" + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self.symbol_to_id and s != "_" and s != "~"
