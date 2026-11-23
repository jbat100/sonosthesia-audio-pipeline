import bisect
import json
import math
import os
import statistics
import struct
from collections import OrderedDict, defaultdict

from colorama import just_fix_windows_console
from termcolor import colored

from sonosthesia_audio_pipeline.utils import ANALYSIS_VERSION


XAS_HEADER = struct.Struct('<4sHHHHIQIIQIIQII')
XAS_CHUNK_HEADER = struct.Struct('<4sBBHIHHIIIQQI')
XAS_INDEX_HEADER = struct.Struct('<4sHHHHIQII')
XAS_INDEX_ENTRY = struct.Struct('<BBHIQIIQQII')
CONTINUOUS_FRAME = struct.Struct('<fffff')
PEAK_EVENT = struct.Struct('<BBHIIff')

XAS_MAGIC = b'XAS1'
CHUNK_MAGIC = b'XASK'
INDEX_MAGIC = b'XASI'
XAS_MAJOR_VERSION = 1
XAS_MINOR_VERSION = 0
CONTINUOUS_KIND = 1
DISCRETE_KIND = 2
FRAME_CODEC = 1
PEAK_CODEC = 2
HEADER_SIZE = 64
CHUNK_HEADER_SIZE = 48
INDEX_HEADER_SIZE = 32
INDEX_ENTRY_SIZE = 48
FLAG_HAS_INDEX = 1
FLAG_FINALIZED = 2
DEFAULT_CHUNK_PAYLOAD_SIZE = 32 * 1024
DEFAULT_ANALYSIS_HOP = 512

just_fix_windows_console()


def _build_crc32c_table():
    table = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        table.append(crc)
    return tuple(table)


CRC32C_TABLE = _build_crc32c_table()


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _align8(value):
    return (value + 7) & ~7


def infer_sample_rate(continuous, analysis_hop=DEFAULT_ANALYSIS_HOP):
    if len(continuous) < 2:
        raise ValueError(
            'At least two continuous samples are required to infer the audio '
            'sample rate; provide sample_rate explicitly.'
        )

    deltas = [
        float(right['time']) - float(left['time'])
        for left, right in zip(continuous, continuous[1:])
    ]
    period = statistics.median(deltas)
    if not math.isfinite(period) or period <= 0:
        raise ValueError('Continuous sample times must increase to infer the audio sample rate.')

    sample_rate = round(analysis_hop / period)
    if sample_rate <= 0:
        raise ValueError('Could not infer a positive audio sample rate.')

    inferred_period = analysis_hop / sample_rate
    tolerance = max(inferred_period * 1e-5, 1e-9)
    if any(abs(delta - inferred_period) > tolerance for delta in deltas):
        raise ValueError(
            'Continuous samples are not regularly spaced at the expected analysis hop.'
        )
    return sample_rate


def _descriptor(data, sample_rate, analysis_hop):
    info = data['info']
    continuous = data['continuous']
    first_tick = (
        round(float(continuous[0]['time']) * sample_rate) if continuous else 0
    )
    return {
        'format': 'sonosthesia-streaming-audio-analysis',
        'version': 1,
        'sampleRate': int(sample_rate),
        'analysisHop': int(analysis_hop),
        'duration': float(info['duration']),
        'timelineOriginSeconds': 0.0,
        'info': info,
        'streams': [
            {
                'id': 0,
                'kind': 'continuous',
                'name': 'continuous',
                'firstTick': first_tick,
                'periodTicks': int(analysis_hop),
                'interpolation': 'linear',
                'encoding': 'frame-f32-v1',
                'components': [
                    {'name': 'rms', 'unit': 'dB', 'codec': 'float32'},
                    {'name': 'lows', 'unit': 'dB', 'codec': 'float32'},
                    {'name': 'mids', 'unit': 'dB', 'codec': 'float32'},
                    {'name': 'highs', 'unit': 'dB', 'codec': 'float32'},
                    {'name': 'centroid', 'unit': 'Hz', 'codec': 'float32'},
                ],
            },
            {
                'id': 1,
                'kind': 'discrete',
                'name': 'peaks',
                'encoding': 'peak-v1',
                'eventType': 'peak',
                'channels': [
                    {'id': 0, 'name': 'main'},
                    {'id': 1, 'name': 'lows'},
                    {'id': 2, 'name': 'mids'},
                    {'id': 3, 'name': 'highs'},
                ],
            },
        ],
    }


def _pack_header(
    descriptor_size,
    descriptor_crc,
    sample_rate,
    duration_ticks,
    index_offset=0,
    chunk_count=0,
    file_size=0,
    flags=0,
):
    prefix = XAS_HEADER.pack(
        XAS_MAGIC,
        XAS_MAJOR_VERSION,
        XAS_MINOR_VERSION,
        HEADER_SIZE,
        flags,
        sample_rate,
        duration_ticks,
        HEADER_SIZE,
        descriptor_size,
        index_offset,
        chunk_count,
        0,
        file_size,
        descriptor_crc,
        0,
    )
    return prefix[:-4] + struct.pack('<I', crc32c(prefix[:HEADER_SIZE - 4]))


def _pack_chunk(kind, codec, stream_id, records, first_tick, last_tick):
    payload = b''.join(records)
    header = XAS_CHUNK_HEADER.pack(
        CHUNK_MAGIC,
        kind,
        codec,
        0,
        stream_id,
        CHUNK_HEADER_SIZE,
        0,
        len(payload),
        len(records),
        0,
        first_tick,
        last_tick,
        crc32c(payload),
    )
    return header + payload + (b'\0' * ((-len(payload)) & 7))


def _continuous_chunks(continuous, sample_rate, analysis_hop, target_payload_size):
    records_per_chunk = max(1, target_payload_size // CONTINUOUS_FRAME.size)
    for start in range(0, len(continuous), records_per_chunk):
        batch = continuous[start:start + records_per_chunk]
        ticks = [round(float(frame['time']) * sample_rate) for frame in batch]
        expected = [ticks[0] + index * analysis_hop for index in range(len(batch))]
        if ticks != expected:
            raise ValueError(
                'Continuous samples do not match the declared sample rate and analysis hop.'
            )

        records = [
            CONTINUOUS_FRAME.pack(
                float(frame['rms']),
                float(frame['lows']),
                float(frame['mids']),
                float(frame['highs']),
                float(frame['centroid']),
            )
            for frame in batch
        ]
        yield (
            CONTINUOUS_KIND,
            FRAME_CODEC,
            0,
            records,
            ticks[0],
            ticks[-1],
        )


def _peak_chunks(peaks, sample_rate, target_payload_size):
    records_per_chunk = max(1, target_payload_size // PEAK_EVENT.size)
    ordered = sorted(peaks, key=lambda peak: float(peak['start']))
    batch = []
    first_tick = None
    last_tick = 0
    for peak in ordered:
        channel = int(peak['channel'])
        start_tick = round(float(peak['start']) * sample_rate)
        duration_ticks = round(float(peak['duration']) * sample_rate)
        if channel < 0 or channel > 3:
            raise ValueError('Peak channel must be between 0 and 3.')
        if duration_ticks < 0 or duration_ticks > 0xFFFFFFFF:
            raise ValueError('Peak duration exceeds the XAS limit.')
        if first_tick is None:
            first_tick = start_tick
            last_tick = first_tick
        start_delta = start_tick - first_tick
        if batch and (
            len(batch) >= records_per_chunk or start_delta > 0xFFFFFFFF
        ):
            yield DISCRETE_KIND, PEAK_CODEC, 1, batch, first_tick, last_tick
            batch = []
            first_tick = start_tick
            start_delta = 0
            last_tick = first_tick

        batch.append(
            PEAK_EVENT.pack(
                channel,
                0,
                0,
                start_delta,
                duration_ticks,
                float(peak['magnitude']),
                float(peak['strength']),
            )
        )
        last_tick = max(last_tick, start_tick + duration_ticks)

    if batch:
        yield DISCRETE_KIND, PEAK_CODEC, 1, batch, first_tick, last_tick


def write_streaming_with_header(
    data,
    header,
    file_path,
    sample_rate=None,
    analysis_hop=DEFAULT_ANALYSIS_HOP,
    chunk_payload_size=DEFAULT_CHUNK_PAYLOAD_SIZE,
):
    if len(header) != 3 or header[0] != ANALYSIS_VERSION:
        raise ValueError(f'XAS conversion requires analysis version {ANALYSIS_VERSION}.')
    if sample_rate is None:
        sample_rate = infer_sample_rate(data['continuous'], analysis_hop)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer.')
    if chunk_payload_size < max(CONTINUOUS_FRAME.size, PEAK_EVENT.size):
        raise ValueError('chunk_payload_size is too small for an XAS record.')

    descriptor_bytes = json.dumps(
        _descriptor(data, sample_rate, analysis_hop),
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    if len(descriptor_bytes) > 1024 * 1024:
        raise ValueError('XAS descriptor exceeds the 1 MiB limit.')

    entries = []
    chunk_generators = (
        _continuous_chunks(
            data['continuous'], sample_rate, analysis_hop, chunk_payload_size
        ),
        _peak_chunks(data['peaks'], sample_rate, chunk_payload_size),
    )

    with open(file_path, 'w+b') as stream:
        stream.write(b'\0' * HEADER_SIZE)
        stream.write(descriptor_bytes)
        stream.write(b'\0' * (_align8(stream.tell()) - stream.tell()))

        for generator in chunk_generators:
            for kind, codec, stream_id, records, first_tick, last_tick in generator:
                chunk_offset = stream.tell()
                chunk_bytes = _pack_chunk(
                    kind, codec, stream_id, records, first_tick, last_tick
                )
                payload_size = len(records) * (
                    CONTINUOUS_FRAME.size if codec == FRAME_CODEC else PEAK_EVENT.size
                )
                stream.write(chunk_bytes)
                entries.append(
                    XAS_INDEX_ENTRY.pack(
                        kind,
                        codec,
                        0,
                        stream_id,
                        chunk_offset,
                        len(chunk_bytes),
                        len(records),
                        first_tick,
                        last_tick,
                        payload_size,
                        0,
                    )
                )

        index_offset = stream.tell()
        index_payload = b''.join(entries)
        stream.write(
            XAS_INDEX_HEADER.pack(
                INDEX_MAGIC,
                XAS_MAJOR_VERSION,
                INDEX_HEADER_SIZE,
                INDEX_ENTRY_SIZE,
                0,
                len(entries),
                index_offset + INDEX_HEADER_SIZE,
                crc32c(index_payload),
                0,
            )
        )
        stream.write(index_payload)
        file_size = stream.tell()
        duration_ticks = round(float(data['info']['duration']) * sample_rate)
        header_bytes = _pack_header(
            len(descriptor_bytes),
            crc32c(descriptor_bytes),
            sample_rate,
            duration_ticks,
            index_offset,
            len(entries),
            file_size,
            FLAG_HAS_INDEX | FLAG_FINALIZED,
        )
        stream.seek(0)
        stream.write(header_bytes)
        stream.flush()
        os.fsync(stream.fileno())

    print(
        colored(
            f'Wrote streaming analysis to {file_path} '
            f'with {len(entries)} indexed chunks and {file_size} bytes',
            'green',
        )
    )


class StreamingAnalysisReader:
    def __init__(self, file_path, verify_crc=True, cache_chunks_per_stream=2):
        if cache_chunks_per_stream < 1:
            raise ValueError('cache_chunks_per_stream must be positive.')
        self.file_path = os.fspath(file_path)
        self.verify_crc = verify_crc
        self.cache_chunks_per_stream = cache_chunks_per_stream
        self._stream = open(self.file_path, 'rb')
        initialized = False
        try:
            self._read_header_and_descriptor()
            self._read_index()
            initialized = True
        finally:
            if not initialized:
                self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._stream.closed:
            self._stream.close()

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def duration_ticks(self):
        return self._duration_ticks

    def _read_exact(self, size, field):
        data = self._stream.read(size)
        if len(data) != size:
            raise ValueError(f'Unexpected end of XAS file while reading {field}.')
        return data

    def _read_header_and_descriptor(self):
        self._stream.seek(0)
        header_bytes = self._read_exact(HEADER_SIZE, 'header')
        (
            magic,
            major,
            minor,
            header_size,
            flags,
            sample_rate,
            duration_ticks,
            descriptor_offset,
            descriptor_size,
            index_offset,
            chunk_count,
            reserved,
            file_size,
            descriptor_crc,
            header_crc,
        ) = XAS_HEADER.unpack(header_bytes)

        if magic != XAS_MAGIC:
            raise ValueError(f'Invalid XAS magic: {magic!r}.')
        if major != XAS_MAJOR_VERSION:
            raise ValueError(f'Unsupported XAS major version: {major}.')
        if header_size != HEADER_SIZE or descriptor_offset != HEADER_SIZE:
            raise ValueError('Invalid XAS header size or descriptor offset.')
        if flags & ~(FLAG_HAS_INDEX | FLAG_FINALIZED):
            raise ValueError(f'Unknown XAS header flags: 0x{flags:x}.')
        if reserved != 0 or sample_rate == 0:
            raise ValueError('Invalid XAS header fields.')
        physical_file_size = os.fstat(self._stream.fileno()).st_size
        if index_offset > physical_file_size:
            raise ValueError('XAS index offset exceeds the physical file size.')
        if file_size and file_size != physical_file_size:
            raise ValueError('XAS file size does not match its header.')
        if descriptor_size > 1024 * 1024:
            raise ValueError('XAS descriptor exceeds the 1 MiB limit.')
        if self.verify_crc and crc32c(header_bytes[:60]) != header_crc:
            raise ValueError('XAS header CRC-32C mismatch.')

        self._flags = flags
        self._sample_rate = sample_rate
        self._duration_ticks = duration_ticks
        self._index_offset = index_offset
        self._chunk_count = chunk_count
        self._minor_version = minor
        self._stream.seek(descriptor_offset)
        descriptor_bytes = self._read_exact(descriptor_size, 'descriptor')
        if self.verify_crc and crc32c(descriptor_bytes) != descriptor_crc:
            raise ValueError('XAS descriptor CRC-32C mismatch.')
        try:
            self._descriptor = json.loads(descriptor_bytes.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('Invalid UTF-8 JSON in XAS descriptor.') from error
        self._validate_descriptor()

    def _validate_descriptor(self):
        descriptor = self._descriptor
        if (
            descriptor.get('format') != 'sonosthesia-streaming-audio-analysis'
            or descriptor.get('version') != XAS_MAJOR_VERSION
        ):
            raise ValueError('XAS descriptor type or version is invalid.')
        if int(descriptor.get('sampleRate', 0)) != self._sample_rate:
            raise ValueError('XAS descriptor sample rate does not match its header.')
        streams = descriptor.get('streams')
        if not isinstance(streams, list):
            raise ValueError('XAS descriptor is missing its stream list.')
        self._stream_descriptors = {}
        for stream in streams:
            stream_id = int(stream.get('id', -1))
            if stream_id in self._stream_descriptors:
                raise ValueError(f'Duplicate XAS stream ID: {stream_id}.')
            self._stream_descriptors[stream_id] = stream

        continuous = self._stream_descriptors.get(0)
        peaks = self._stream_descriptors.get(1)
        if (
            not continuous
            or continuous.get('kind') != 'continuous'
            or continuous.get('encoding') != 'frame-f32-v1'
            or not peaks
            or peaks.get('kind') != 'discrete'
            or peaks.get('encoding') != 'peak-v1'
        ):
            raise ValueError('XAS descriptor is missing required version-1 streams.')
        self._period_ticks = int(continuous.get('periodTicks', 0))
        if self._period_ticks <= 0:
            raise ValueError('XAS continuous periodTicks must be positive.')
        if self._period_ticks != int(descriptor.get('analysisHop', 0)):
            raise ValueError('XAS analysis hop and continuous period do not match.')

    def _read_index(self):
        if not self._flags & FLAG_HAS_INDEX:
            raise ValueError('Indexed XAS access requires a finalized file with an index.')
        if self._index_offset <= HEADER_SIZE:
            raise ValueError('Invalid XAS index offset.')
        self._stream.seek(self._index_offset)
        raw_header = self._read_exact(INDEX_HEADER_SIZE, 'index header')
        (
            magic,
            major,
            header_size,
            entry_size,
            flags,
            entry_count,
            first_entry_offset,
            index_crc,
            reserved,
        ) = XAS_INDEX_HEADER.unpack(raw_header)
        if (
            magic != INDEX_MAGIC
            or major != XAS_MAJOR_VERSION
            or header_size != INDEX_HEADER_SIZE
            or entry_size != INDEX_ENTRY_SIZE
            or flags != 0
            or reserved != 0
        ):
            raise ValueError('Invalid XAS index header.')
        if entry_count != self._chunk_count:
            raise ValueError('XAS index and header chunk counts differ.')
        if first_entry_offset != self._index_offset + INDEX_HEADER_SIZE:
            raise ValueError('Invalid XAS index entry offset.')
        physical_file_size = os.fstat(self._stream.fileno()).st_size
        if entry_count > (physical_file_size - first_entry_offset) // entry_size:
            raise ValueError('XAS index entry count exceeds the remaining file size.')

        raw_entries = self._read_exact(entry_count * entry_size, 'index entries')
        if self.verify_crc and crc32c(raw_entries) != index_crc:
            raise ValueError('XAS index CRC-32C mismatch.')

        self._entries = []
        self._entries_by_stream = defaultdict(list)
        previous_offset = -1
        for offset in range(0, len(raw_entries), entry_size):
            entry = XAS_INDEX_ENTRY.unpack(raw_entries[offset:offset + entry_size])
            (
                kind,
                codec,
                entry_flags,
                stream_id,
                chunk_offset,
                stored_size,
                record_count,
                first_tick,
                last_tick,
                payload_size,
                entry_reserved,
            ) = entry
            if entry_flags != 0 or entry_reserved != 0:
                raise ValueError('Invalid reserved field in XAS index entry.')
            if stream_id not in self._stream_descriptors:
                raise ValueError(f'XAS index refers to unknown stream {stream_id}.')
            stream_kind = (
                CONTINUOUS_KIND
                if self._stream_descriptors[stream_id].get('kind') == 'continuous'
                else DISCRETE_KIND
            )
            if kind != stream_kind or kind not in (CONTINUOUS_KIND, DISCRETE_KIND):
                raise ValueError('XAS chunk kind does not match its stream descriptor.')
            if chunk_offset <= previous_offset or stored_size < CHUNK_HEADER_SIZE:
                raise ValueError('XAS index entries are not ordered or have invalid sizes.')
            if (
                chunk_offset + stored_size > self._index_offset
                or not (CHUNK_HEADER_SIZE + payload_size <= stored_size)
            ):
                raise ValueError('XAS chunk bounds exceed the data section.')
            expected_codec = FRAME_CODEC if kind == CONTINUOUS_KIND else PEAK_CODEC
            expected_record_size = (
                CONTINUOUS_FRAME.size if kind == CONTINUOUS_KIND else PEAK_EVENT.size
            )
            if (
                codec != expected_codec
                or payload_size != record_count * expected_record_size
                or last_tick < first_tick
            ):
                raise ValueError('Invalid XAS index entry encoding or timing.')
            item = {
                'kind': kind,
                'codec': codec,
                'stream_id': stream_id,
                'chunk_offset': chunk_offset,
                'stored_size': stored_size,
                'record_count': record_count,
                'first_tick': first_tick,
                'last_tick': last_tick,
                'payload_size': payload_size,
            }
            self._entries.append(item)
            self._entries_by_stream[stream_id].append(item)
            previous_offset = chunk_offset

        for entries in self._entries_by_stream.values():
            entries.sort(key=lambda item: item['first_tick'])
        self._start_ticks_by_stream = {
            stream_id: [entry['first_tick'] for entry in entries]
            for stream_id, entries in self._entries_by_stream.items()
        }
        self._cache = {
            stream_id: OrderedDict() for stream_id in self._stream_descriptors
        }
        self._continuous_cache = OrderedDict()
        self._event_cache = OrderedDict()

    def _load_chunk(self, entry):
        stream_id = entry['stream_id']
        cache = self._cache[stream_id]
        cache_key = entry['chunk_offset']
        if cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key]

        self._stream.seek(entry['chunk_offset'])
        raw_header = self._read_exact(CHUNK_HEADER_SIZE, 'chunk header')
        (
            magic,
            kind,
            codec,
            flags,
            header_stream_id,
            header_size,
            reserved0,
            payload_size,
            record_count,
            reserved1,
            first_tick,
            last_tick,
            payload_crc,
        ) = XAS_CHUNK_HEADER.unpack(raw_header)
        if (
            magic != CHUNK_MAGIC
            or flags != 0
            or reserved0 != 0
            or reserved1 != 0
            or header_size != CHUNK_HEADER_SIZE
            or kind != entry['kind']
            or codec != entry['codec']
            or header_stream_id != stream_id
            or payload_size != entry['payload_size']
            or record_count != entry['record_count']
            or first_tick != entry['first_tick']
            or last_tick != entry['last_tick']
        ):
            raise ValueError('XAS chunk header does not match its index entry.')
        payload = self._read_exact(payload_size, 'chunk payload')
        if self.verify_crc and crc32c(payload) != payload_crc:
            raise ValueError('XAS chunk payload CRC-32C mismatch.')

        cache[cache_key] = payload
        cache.move_to_end(cache_key)
        while len(cache) > self.cache_chunks_per_stream:
            cache.popitem(last=False)
        return payload

    def _continuous_payload(self, entry):
        cache_key = entry['chunk_offset']
        if cache_key in self._continuous_cache:
            self._continuous_cache.move_to_end(cache_key)
            return self._continuous_cache[cache_key]
        frames = tuple(CONTINUOUS_FRAME.iter_unpack(self._load_chunk(entry)))
        self._continuous_cache[cache_key] = frames
        self._continuous_cache.move_to_end(cache_key)
        while len(self._continuous_cache) > self.cache_chunks_per_stream:
            self._continuous_cache.popitem(last=False)
        return frames

    def _event_payload(self, entry):
        cache_key = entry['chunk_offset']
        if cache_key in self._event_cache:
            self._event_cache.move_to_end(cache_key)
            return self._event_cache[cache_key]
        events = tuple(PEAK_EVENT.iter_unpack(self._load_chunk(entry)))
        self._event_cache[cache_key] = events
        self._event_cache.move_to_end(cache_key)
        while len(self._event_cache) > self.cache_chunks_per_stream:
            self._event_cache.popitem(last=False)
        return events

    @staticmethod
    def _interpolate(left, right, amount):
        return tuple(a + (b - a) * amount for a, b in zip(left, right))

    def continuous_at(self, time_seconds):
        tick = round(float(time_seconds) * self._sample_rate)
        entries = self._entries_by_stream[0]
        if not entries or tick < entries[0]['first_tick'] or tick > entries[-1]['last_tick']:
            return None

        starts = self._start_ticks_by_stream[0]
        index = bisect.bisect_right(starts, tick) - 1
        index = max(0, index)
        entry = entries[index]
        if tick > entry['last_tick']:
            if index + 1 >= len(entries):
                return None
            next_entry = entries[index + 1]
            if next_entry['first_tick'] != entry['last_tick'] + self._period_ticks:
                return None
            if tick >= next_entry['first_tick']:
                entry = next_entry
                index += 1
        frames = self._continuous_payload(entry)
        frame_index, remainder = divmod(tick - entry['first_tick'], self._period_ticks)
        if frame_index < 0 or frame_index >= len(frames):
            return None
        if remainder == 0:
            return frames[frame_index]

        if frame_index == len(frames) - 1:
            if index + 1 >= len(entries):
                return None
            next_entry = entries[index + 1]
            expected_tick = entry['last_tick'] + self._period_ticks
            if next_entry['first_tick'] != expected_tick:
                return None
            next_frames = self._continuous_payload(next_entry)
            amount = remainder / self._period_ticks
            return self._interpolate(frames[-1], next_frames[0], amount)

        amount = remainder / self._period_ticks
        return self._interpolate(frames[frame_index], frames[frame_index + 1], amount)

    def events_overlapping(self, start_seconds, end_seconds):
        start_tick = round(float(start_seconds) * self._sample_rate)
        end_tick = round(float(end_seconds) * self._sample_rate)
        if end_tick < start_tick:
            start_tick, end_tick = end_tick, start_tick

        result = []
        for entry in self._entries_by_stream[1]:
            if entry['first_tick'] > end_tick:
                break
            if entry['last_tick'] < start_tick:
                continue
            for channel, _, _, delta, duration, magnitude, strength in self._event_payload(entry):
                event_start = entry['first_tick'] + delta
                event_end = event_start + duration
                if event_start <= end_tick and event_end >= start_tick:
                    result.append(
                        {
                            'channel': channel,
                            'start': event_start / self._sample_rate,
                            'duration': duration / self._sample_rate,
                            'magnitude': magnitude,
                            'strength': strength,
                        }
                    )
        result.sort(key=lambda event: event['start'])
        return result

    def read_analysis(self):
        continuous = []
        for entry in self._entries_by_stream[0]:
            for index, frame in enumerate(self._continuous_payload(entry)):
                tick = entry['first_tick'] + index * self._period_ticks
                continuous.append(
                    {
                        'time': tick / self._sample_rate,
                        'rms': frame[0],
                        'lows': frame[1],
                        'mids': frame[2],
                        'highs': frame[3],
                        'centroid': frame[4],
                    }
                )

        peaks = []
        for entry in self._entries_by_stream[1]:
            for channel, _, _, delta, duration, magnitude, strength in self._event_payload(entry):
                peaks.append(
                    {
                        'channel': channel,
                        'start': (entry['first_tick'] + delta) / self._sample_rate,
                        'duration': duration / self._sample_rate,
                        'magnitude': magnitude,
                        'strength': strength,
                    }
                )
        peaks.sort(key=lambda event: event['start'])
        return {
            'continuous': continuous,
            'peaks': peaks,
            'info': self._descriptor['info'],
        }


def read_streaming_with_header(file_path):
    with StreamingAnalysisReader(file_path) as reader:
        data = reader.read_analysis()
    print(colored(f'Loaded streaming analysis from {file_path}', 'green'))
    return [ANALYSIS_VERSION, 0, 0], data
