"""MIDI event tokenizer: encodes MIDI files to integer token sequences and back."""
import pretty_midi

RANGE_NOTE_ON    = 128
RANGE_NOTE_OFF   = 128
RANGE_VEL        = 32
RANGE_TIME_SHIFT = 100

START_IDX = {
    "note_on":    0,
    "note_off":   RANGE_NOTE_ON,
    "time_shift": RANGE_NOTE_ON + RANGE_NOTE_OFF,
    "velocity":   RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
}


class SplitNote:
    def __init__(self, type, time, value, velocity):
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return f"<SplitNote time={self.time} type={self.type} value={self.value} vel={self.velocity}>"


class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return f"<Event type={self.type} value={self.value}>"

    def to_int(self):
        return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info["type"], info["value"])

    @staticmethod
    def _type_check(int_value):
        if int_value < RANGE_NOTE_ON:
            return {"type": "note_on", "value": int_value}
        elif int_value < RANGE_NOTE_ON + RANGE_NOTE_OFF:
            return {"type": "note_off", "value": int_value - RANGE_NOTE_ON}
        elif int_value < RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT:
            return {"type": "time_shift", "value": int_value - (RANGE_NOTE_ON + RANGE_NOTE_OFF)}
        else:
            return {"type": "velocity", "value": int_value - (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)}


class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {}

    def add_managed_note(self, note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)
            self._note_dict[note.pitch] = note.start


def _control_preprocess(ctrl_changes):
    sustains, manager = [], None
    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and sustains:
            sustains[-1].end = ctrl.time
    return sustains


def _note_preprocess(sustains, notes):
    note_stream = []
    for sustain in sustains:
        for note_idx, note in enumerate(notes):
            if note.start < sustain.start:
                note_stream.append(note)
            elif note.start > sustain.end:
                notes = notes[note_idx:]
                sustain.transposition_notes()
                break
            else:
                sustain.add_managed_note(note)
    for sustain in sustains:
        note_stream += sustain.managed_notes
    note_stream.sort(key=lambda x: x.start)
    return note_stream


def _divide_note(notes):
    result = []
    notes.sort(key=lambda x: x.start)
    for note in notes:
        result.append(SplitNote("note_on",  note.start, note.pitch, note.velocity))
        result.append(SplitNote("note_off", note.end,   note.pitch, None))
    return result


def _make_time_shift_events(prev_time, post_time):
    time_interval = int(round((post_time - prev_time) * 100))
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event("time_shift", RANGE_TIME_SHIFT - 1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval > 0:
        results.append(Event("time_shift", time_interval - 1))
    return results


def _snote_to_events(snote, prev_vel):
    result = []
    if snote.velocity is not None:
        modified_velocity = snote.velocity // 4
        if prev_vel != modified_velocity:
            result.append(Event("velocity", modified_velocity))
    result.append(Event(snote.type, snote.value))
    return result


def _event_seq_to_snote_seq(event_sequence):
    timeline, velocity, snote_seq = 0, 0, []
    for event in event_sequence:
        if event.type == "time_shift":
            timeline += (event.value + 1) / 100
        elif event.type == "velocity":
            velocity = event.value * 4
        else:
            snote_seq.append(SplitNote(event.type, timeline, event.value, velocity))
    return snote_seq


def _merge_note(snote_sequence):
    note_on_dict, result = {}, []
    for snote in snote_sequence:
        if snote.type == "note_on":
            note_on_dict[snote.value] = snote
        elif snote.type == "note_off" and snote.value in note_on_dict:
            on = note_on_dict[snote.value]
            if snote.time - on.time > 0:
                result.append(pretty_midi.Note(on.velocity, snote.value, on.time, snote.time))
    return result


def encode_midi(file_path):
    """Encode a MIDI file to a list of integer tokens."""
    events, notes = [], []
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    for inst in mid.instruments:
        ctrls = _control_preprocess([c for c in inst.control_changes if c.number == 64])
        notes += _note_preprocess(ctrls, inst.notes)

    dnotes = _divide_note(notes)
    dnotes.sort(key=lambda x: x.time)

    cur_time, cur_vel = 0, 0
    for snote in dnotes:
        events += _make_time_shift_events(cur_time, snote.time)
        events += _snote_to_events(snote, cur_vel)
        cur_time = snote.time
        cur_vel = snote.velocity if snote.velocity is not None else cur_vel

    return [e.to_int() for e in events]


def decode_midi(idx_array, file_path=None):
    """Decode a list of integer tokens back to a PrettyMIDI object."""
    event_sequence = [Event.from_int(i) for i in idx_array]
    snote_seq = _event_seq_to_snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x: x.start)

    mid = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(1, False, "deepTechno")
    instrument.notes = note_seq
    mid.instruments.append(instrument)

    if file_path is not None:
        mid.write(file_path)
    return mid
