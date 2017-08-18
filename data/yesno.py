import torchaudio

# https://github.com/dhpollack/audio, see VCTK branch
torchaudio.datasets.YESNO(".", download=True, dev_mode=True)
