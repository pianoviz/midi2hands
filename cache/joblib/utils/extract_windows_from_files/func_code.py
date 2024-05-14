# first line: 133
@memory.cache
def extract_windows_from_files(paths, window_size, step_size):
    all_windows = []
    all_labels = []
    mp = MidiEventProcessor()
    for path in tqdm(paths):
        events = mp.extract_note_events(path)
        windows, labels = extract_windows_and_labels(
            events, window_size, step_size, bidirectional=True
        )
        all_windows.extend(windows)
        all_labels.extend(labels)
    return np.array(all_windows), np.array(all_labels)
