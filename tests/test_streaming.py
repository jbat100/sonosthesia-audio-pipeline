import json
import os
import tempfile
import unittest

from sonosthesia_audio_pipeline.conversion import run_conversion
from sonosthesia_audio_pipeline.streaming import (
    StreamingAnalysisReader,
    crc32c,
    write_streaming_with_header,
)
from sonosthesia_audio_pipeline.utils import (
    ANALYSIS_VERSION,
    JSON_ANALYSIS_EXTENSION,
    MSGPACK_ANALYSIS_EXTENSION,
    STREAMING_ANALYSIS_EXTENSION,
    write_packed_with_header,
)


def sample_data(frame_count=6):
    continuous = []
    for index in range(frame_count):
        continuous.append(
            {
                'time': index * 512 / 48000,
                'rms': float(index),
                'lows': float(index + 1),
                'mids': float(index + 2),
                'highs': float(index + 3),
                'centroid': float(100 * index),
            }
        )
    return {
        'continuous': continuous,
        'peaks': [
            {
                'channel': 0,
                'start': 0.001,
                'duration': 0.004,
                'magnitude': -12.5,
                'strength': 0.8,
            },
            {
                'channel': 2,
                'start': 0.008,
                'duration': 0.0,
                'magnitude': -20.0,
                'strength': 0.4,
            },
        ],
        'info': {
            'duration': 0.01,
            'main': {
                'band': {'lower': 20.0, 'upper': 8000.0},
                'magnitude': {'lower': -80.0, 'upper': -5.0},
                'peaks': 1,
            },
            'lows': {
                'band': {'lower': 30.0, 'upper': 100.0},
                'magnitude': {'lower': -80.0, 'upper': -5.0},
                'peaks': 0,
            },
            'mids': {
                'band': {'lower': 500.0, 'upper': 2000.0},
                'magnitude': {'lower': -80.0, 'upper': -5.0},
                'peaks': 1,
            },
            'highs': {
                'band': {'lower': 4000.0, 'upper': 16000.0},
                'magnitude': {'lower': -80.0, 'upper': -5.0},
                'peaks': 0,
            },
            'centroid': {'lower': 0.0, 'upper': 500.0},
        },
    }


class StreamingAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temporary_directory.name, 'audio.xas')

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_file(self):
        write_streaming_with_header(
            sample_data(),
            [ANALYSIS_VERSION, 0, 0],
            self.path,
            sample_rate=48000,
            chunk_payload_size=40,
        )

    def test_crc32c_uses_castagnoli_variant(self):
        self.assertEqual(crc32c(b'123456789'), 0xE3069283)

    def test_round_trip_preserves_analysis_content_with_small_chunks(self):
        self.write_file()

        with StreamingAnalysisReader(self.path) as reader:
            result = reader.read_analysis()
            chunk_count = len(reader._entries_by_stream[0])

        self.assertEqual(len(result['continuous']), 6)
        self.assertEqual(chunk_count, 3)
        self.assertEqual(result['continuous'][3]['rms'], 3.0)
        self.assertEqual(result['peaks'][1]['channel'], 2)
        self.assertEqual(result['info']['main']['peaks'], 1)

    def test_continuous_evaluation_interpolates_across_chunk_boundary(self):
        self.write_file()

        with StreamingAnalysisReader(self.path) as reader:
            value = reader.continuous_at((1.5 * 512) / 48000)
            first_payload = reader._continuous_payload(
                reader._entries_by_stream[0][0]
            )
            second_payload = reader._continuous_payload(
                reader._entries_by_stream[0][0]
            )

        self.assertIsNotNone(value)
        self.assertAlmostEqual(value[0], 1.5)
        self.assertAlmostEqual(value[4], 150.0)
        self.assertIs(first_payload, second_payload)

    def test_peak_range_query_includes_overlapping_and_point_events(self):
        self.write_file()

        with StreamingAnalysisReader(self.path) as reader:
            first = reader.events_overlapping(0.004, 0.004)
            second = reader.events_overlapping(0.008, 0.008)

        self.assertEqual([event['channel'] for event in first], [0])
        self.assertEqual([event['channel'] for event in second], [2])

    def test_corrupted_payload_is_rejected(self):
        self.write_file()
        with StreamingAnalysisReader(self.path) as reader:
            chunk_offset = reader._entries[0]['chunk_offset']
        with open(self.path, 'r+b') as stream:
            stream.seek(chunk_offset + 48)
            original = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(bytes([original[0] ^ 1]))

        with StreamingAnalysisReader(self.path) as reader:
            with self.assertRaisesRegex(ValueError, 'CRC-32C mismatch'):
                reader.continuous_at(0.0)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base_path = os.path.join(self.temporary_directory.name, 'audio')

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_converts_between_all_three_formats(self):
        xaa_path = self.base_path + MSGPACK_ANALYSIS_EXTENSION
        write_packed_with_header(
            sample_data(), [ANALYSIS_VERSION, 0, 0], xaa_path
        )

        run_conversion(
            xaa_path,
            output_format=STREAMING_ANALYSIS_EXTENSION,
        )
        xas_path = self.base_path + STREAMING_ANALYSIS_EXTENSION
        self.assertTrue(os.path.isfile(xas_path))

        run_conversion(xas_path, output_format=JSON_ANALYSIS_EXTENSION)
        json_path = self.base_path + JSON_ANALYSIS_EXTENSION
        with open(json_path, encoding='utf-8') as stream:
            json_data = json.load(stream)
        self.assertEqual(
            json_data['content']['continuous'][2]['rms'],
            2.0,
        )

        run_conversion(json_path, output_format=MSGPACK_ANALYSIS_EXTENSION)
        self.assertTrue(os.path.isfile(xaa_path))


if __name__ == '__main__':
    unittest.main()
