# Sonosthesia Streaming Audio Analysis Format

## 1. Status

This document proposes a binary streaming format for the output of
`sonosthesia-audio analysis`. It is a design specification, not a description of the
current MessagePack-based `.xaa` format.

The proposed format is version 1 of the **Sonosthesia Streaming Audio Analysis**
format, abbreviated **XAS**. The recommended file extension is `.xas`. Existing
`.xaa` and `.json` files remain unchanged.

## 2. Goals

The format is designed for playback, scrubbing, and live timeline evaluation in
Unity and other realtime engines.

Its primary goals are:

1. Read a single bounded chunk for a requested time instead of loading the whole
   file.
2. Distinguish continuous, regularly sampled analysis from discrete, interval-based
   events.
3. Make chunk boundaries and payload layouts useful for sequential streaming and
   random access.
4. Keep the representation simple enough for a reader to implement with basic
   `FileStream`, `Seek`, and byte-array operations.
5. Store values without repeated property names, object wrappers, or per-frame
   timestamps.
6. Permit independent caching and interpolation of continuous channels.
7. Preserve enough information to reproduce the semantic content of analysis
   format version 2.

### Non-goals

- XAS is not intended to be human readable.
- Compression is not required in version 1. Reducing seeking and retained-memory
  cost is preferred over maximizing offline compression ratio.
- The format does not embed the source audio.
- XAS does not prescribe a Unity API. It specifies bytes and reader behavior only.

## 3. Terminology and data model

All multibyte integers and floating-point values are **little-endian**. Floating-point
values are IEEE 754 `float32` unless stated otherwise. Structures do not inherit C,
C++, or C# compiler padding; their byte layout is defined by this document.

XAS contains a sequence of logical **streams**:

- A **continuous stream** contains regularly timed frames. Every frame has the same
  set of numeric components and is evaluated by interpolation.
- A **discrete-event stream** contains finite time intervals. An event exists at a
  start time, has a duration, and may carry numeric attributes. It is evaluated by
  querying events that overlap a time or time range, not by interpolation.

Version 1 defines:

| Stream ID | Stream kind | Logical name | Contents |
| --- | --- | --- | --- |
| 0 | Continuous | `continuous` | RMS, lows, mids, highs, centroid |
| 1 | Discrete event | `peaks` | Main, lows, mids, and highs peak events |

Additional stream descriptors and stream kinds may be added by a later format
version.

### Time representation

Times use unsigned 64-bit integer **ticks**. The ticks per second are declared by
`ticksPerSecond` in the file header.

Version 1 requires `ticksPerSecond` to equal the sample rate of the analyzed audio.
Thus, a time in seconds is:

```text
timeSeconds = tick / ticksPerSecond
```

Tick arithmetic avoids per-frame floating-point timestamp storage. Writers should
preserve the analyzed audio sample rate rather than converting it to a universal
rate.

`timelineOriginSeconds` is optional source-media metadata. It does not affect XAS
timing: the first analysis frame is ordinarily at timeline tick 0.

## 4. File layout

An XAS v1 file has this layout:

```text
+--------------------------+
| File header, 64 bytes    |
+--------------------------+
| UTF-8 JSON descriptor    |
+--------------------------+
| Padding to 8-byte edge   |
+--------------------------+
| Chunk 0                  |
| Chunk padding            |
| Chunk 1                  |
| Chunk padding            |
| ...                      |
+--------------------------+
| Index section            |
+--------------------------+
```

There is no global data payload allocation. Chunks are written sequentially and
may be consumed as they arrive. When writing is complete, the writer appends the
index and patches the header to point to it. In an unfinalized stream, the index
may be absent.

For ordinary local-file use the chunk index enables random access. For an HTTP
response, a growing file, or another forward-only source, a reader can process
chunks sequentially without requiring the index to have been read.

## 5. File header

The fixed header is exactly 64 bytes:

| Offset | Size | Type | Field | Meaning |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | bytes | `magic` | ASCII `XAS1` |
| 4 | 2 | `uint16` | `majorVersion` | `1` |
| 6 | 2 | `uint16` | `minorVersion` | `0` |
| 8 | 2 | `uint16` | `headerSize` | `64` |
| 10 | 2 | `uint16` | `flags` | See below |
| 12 | 4 | `uint32` | `ticksPerSecond` | Audio sample rate; nonzero |
| 16 | 8 | `uint64` | `durationTicks` | Analysis duration |
| 24 | 4 | `uint32` | `descriptorOffset` | Normally `64` |
| 28 | 4 | `uint32` | `descriptorSize` | JSON descriptor byte count |
| 32 | 8 | `uint64` | `indexOffset` | Absolute offset of the index; 0 while unfinalized |
| 40 | 4 | `uint32` | `chunkCount` | Number of chunks; 0 while unknown |
| 44 | 4 | `uint32` | `reserved` | Must be zero |
| 48 | 8 | `uint64` | `fileSize` | Final file size; 0 while unknown |
| 56 | 4 | `uint32` | `descriptorCrc32` | CRC-32C of descriptor bytes |
| 60 | 4 | `uint32` | `headerCrc32` | CRC-32C of bytes 0 through 59 |

Header flag bits are:

| Bit | Name | Meaning |
| ---: | --- | --- |
| 0 | `HAS_INDEX` | File contains a complete index |
| 1 | `FINALIZED` | Header offsets, sizes, and CRC fields are final |
| 2-15 | Reserved | Must be zero |

The descriptor offset must be 64 in version 1. Readers must use
`descriptorOffset` and `descriptorSize` nevertheless.

If `FINALIZED` is clear, a forward-only reader may still consume complete chunks.
It must not attempt random access from the provisional header. A reader must reject
an otherwise complete file with incompatible major version, invalid `headerSize`,
unknown flag bits, or failed integrity fields.

## 6. Descriptor

The descriptor is a UTF-8 JSON object immediately following the header. It contains
metadata and stream declarations but no frame or event arrays. It should remain
below 64 KiB and must not exceed 1 MiB.

The version 1 descriptor is:

```json
{
  "format": "sonosthesia-streaming-audio-analysis",
  "version": 1,
  "sampleRate": 48000,
  "analysisHop": 512,
  "duration": 181.53847916666666,
  "timelineOriginSeconds": 0.0,
  "info": {
    "duration": 181.53847916666666,
    "main": {
      "band": {"lower": 20.0, "upper": 8000.0},
      "magnitude": {"lower": -86.0, "upper": -6.0},
      "peaks": 745
    },
    "lows": {
      "band": {"lower": 30.0, "upper": 100.0},
      "magnitude": {"lower": -86.0, "upper": -6.0},
      "peaks": 678
    },
    "mids": {
      "band": {"lower": 500.0, "upper": 2000.0},
      "magnitude": {"lower": -93.0, "upper": -13.0},
      "peaks": 684
    },
    "highs": {
      "band": {"lower": 4000.0, "upper": 16000.0},
      "magnitude": {"lower": -99.0, "upper": -19.0},
      "peaks": 683
    },
    "centroid": {"lower": 0.0, "upper": 8195.0}
  },
  "streams": [
    {
      "id": 0,
      "kind": "continuous",
      "name": "continuous",
      "firstTick": 0,
      "periodTicks": 512,
      "interpolation": "linear",
      "encoding": "frame-f32-v1",
      "components": [
        {"name": "rms", "unit": "dB", "codec": "float32"},
        {"name": "lows", "unit": "dB", "codec": "float32"},
        {"name": "mids", "unit": "dB", "codec": "float32"},
        {"name": "highs", "unit": "dB", "codec": "float32"},
        {"name": "centroid", "unit": "Hz", "codec": "float32"}
      ]
    },
    {
      "id": 1,
      "kind": "discrete",
      "name": "peaks",
      "encoding": "peak-v1",
      "eventType": "peak",
      "channels": [
        {"id": 0, "name": "main"},
        {"id": 1, "name": "lows"},
        {"id": 2, "name": "mids"},
        {"id": 3, "name": "highs"}
      ]
    }
  ]
}
```

`sampleRate` and `analysisHop` are represented as JSON numbers and must agree with
the corresponding tick fields. `sampleRate` must equal header
`ticksPerSecond`. `periodTicks` must equal `analysisHop`.

Unknown JSON properties must be ignored. This permits metadata such as source
hashes, tool versions, or application-defined labels without changing the major
format version. A stream's `encoding` is required and is not treated as an
extension point for version-1 readers.

The descriptor should be generated in a deterministic property order when
reproducible output is desired, but descriptor property order is not semantically
significant.

## 7. Chunk framing

Each chunk begins on an 8-byte-aligned absolute file offset. Its fixed header is
48 bytes:

| Offset | Size | Type | Field | Meaning |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | bytes | `magic` | ASCII `XASK` |
| 4 | 1 | `uint8` | `kind` | `1` continuous, `2` discrete |
| 5 | 1 | `uint8` | `codec` | See stream encoding table |
| 6 | 2 | `uint16` | `flags` | Reserved in version 1; zero |
| 8 | 4 | `uint32` | `streamId` | Descriptor stream ID |
| 12 | 2 | `uint16` | `headerSize` | `48` |
| 14 | 2 | `uint16` | `reserved0` | Zero |
| 16 | 4 | `uint32` | `payloadSize` | Payload bytes, excluding padding |
| 20 | 4 | `uint32` | `recordCount` | Frames or events in payload |
| 24 | 4 | `uint32` | `reserved1` | Zero |
| 28 | 8 | `uint64` | `firstTick` | First continuous sample/event start |
| 36 | 8 | `uint64` | `lastTick` | Last frame tick or maximum event end |
| 44 | 4 | `uint32` | `payloadCrc32` | CRC-32C of payload only |

The payload CRC does not include padding. Padding bytes after a payload must be
zero. The 48-byte header plus payload plus padding are called the chunk's
**stored size**.

Version 1 stream encodings are:

| Value | Encoding |
| ---: | --- |
| 1 | `frame-f32-v1` |
| 2 | `peak-v1` |

A reader must reject an unknown major version or encoding rather than guessing a
different layout. A reader must also reject a record count or payload size that is
inconsistent with the selected fixed-size encoding.

Recommended chunk payloads are 16-64 KiB. A 64 KiB maximum payload is recommended
but not mandatory. Writers should choose a bound that keeps one decoded chunk,
the next/previous chunk, and the index within the application's memory budget.

## 8. Continuous payload

The version 1 continuous stream has five `float32` components in this exact order:

1. `rms`
2. `lows`
3. `mids`
4. `highs`
5. `centroid`

The payload is an array of frames in **array-of-structures** order:

```c
struct ContinuousFrameV1
{
    float rms;
    float lows;
    float mids;
    float highs;
    float centroid;
};
```

Each frame is exactly 20 bytes. For frame `i`:

```text
frameTick[i] = chunk.firstTick + i * stream.periodTicks
```

`chunk.lastTick` must be the timestamp of the last frame. Consequently:

```text
lastTick = firstTick + (recordCount - 1) * periodTicks
```

The representation is 20 bytes per frame instead of storing six 64-bit
floating-point values and repeated field names.

Continuous chunks are ordered by `firstTick` within their stream. They are
non-overlapping. Gaps are permitted; a reader must not silently treat a gap as
continuous data. A writer should emit a contiguous stream for normal analysis
output.

### Continuous evaluation

For a request at tick `t`, a reader should:

1. Use the index to find the chunk containing `t`, or the pair of chunks adjacent
   to it at a gap or boundary.
2. Validate and decode only the selected chunk payload.
3. Evaluate the sample immediately before `t` and, when needed, the sample after
   it.
4. Linearly interpolate finite neighboring values.

The exact behavior at the beginning and end of a stream is application-defined:
hold, clamp, or return unavailable. A reader must not pretend values exist outside
the valid interval unless its API explicitly applies clamping.

## 9. Discrete-event payload

Each peak event is a 20-byte record:

```text
struct PeakEventV1
{
    uint8   channel;
    uint8   flags;
    uint16  reserved;
    uint32  startDeltaTicks;
    uint32  durationTicks;
    float32 magnitude;
    float32 strength;
};
```

`channel` values are:

| Value | Meaning |
| ---: | --- |
| 0 | Main |
| 1 | Lows |
| 2 | Mids |
| 3 | Highs |

The absolute event start is:

```text
startTick = chunk.firstTick + startDeltaTicks
```

`durationTicks` is the peak duration, not the time to the next event. `magnitude`
is the peak magnitude in dB and `strength` is the normalized onset strength, as in
analysis format version 2.

Event records are sorted by absolute start tick. `flags` and `reserved` are zero in
version 1. `chunk.lastTick` is the maximum `startTick + durationTicks` in the
chunk.

The 32-bit deltas limit a single discrete-event chunk's time span to approximately
24.8 hours at 48 kHz. Writers must start a new chunk before this limit, regardless
of event count.

### Discrete-event evaluation

An event is active at tick `t` when:

```text
startTick <= t <= startTick + durationTicks
```

Boundary inclusivity at the exact end is defined as inclusive so a zero-duration
event can be observed as a point event. For a range query `[a, b]`, an event
overlaps when:

```text
startTick <= b && startTick + durationTicks >= a
```

Events in a chunk may extend beyond the next chunk's `firstTick`. Readers must use
`lastTick` to test chunk overlap and must not assume only the single chunk
containing a query point can contain relevant events.

## 10. Index

The index begins at an 8-byte-aligned offset and is written only after the data
chunks. The header's `indexOffset`, `chunkCount`, and `HAS_INDEX` flag direct readers
to it.

The index starts with a 32-byte table header:

| Offset | Size | Type | Field | Meaning |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | bytes | `magic` | ASCII `XASI` |
| 4 | 2 | `uint16` | `majorVersion` | `1` |
| 6 | 2 | `uint16` | `headerSize` | `32` |
| 8 | 2 | `uint16` | `entrySize` | `48` |
| 10 | 2 | `uint16` | `flags` | Zero |
| 12 | 4 | `uint32` | `entryCount` | Number of index entries |
| 16 | 8 | `uint64` | `firstEntryOffset` | Absolute offset of first entry |
| 24 | 4 | `uint32` | `indexCrc32` | CRC-32C of index entries |
| 28 | 4 | `uint32` | `reserved` | Zero |

Each 48-byte index entry is:

| Offset | Size | Type | Field | Meaning |
| ---: | ---: | --- | --- | --- |
| 0 | 1 | `uint8` | `kind` | Continuous or discrete |
| 1 | 1 | `uint8` | `codec` | Stream payload encoding |
| 2 | 2 | `uint16` | `flags` | Reserved in version 1 |
| 4 | 4 | `uint32` | `streamId` | Descriptor stream ID |
| 8 | 8 | `uint64` | `chunkOffset` | Absolute chunk header offset |
| 16 | 4 | `uint32` | `storedSize` | Header, payload, and padding |
| 20 | 4 | `uint32` | `recordCount` | Frames or events |
| 24 | 8 | `uint64` | `firstTick` | Same as chunk header |
| 32 | 8 | `uint64` | `lastTick` | Same as chunk header |
| 40 | 4 | `uint32` | `payloadSize` | Excludes padding |
| 44 | 4 | `uint32` | `reserved` | Zero |

Entries are stored in increasing `chunkOffset` order. For practical random access,
readers may build one small in-memory array per stream from this index. The entire
array is metadata: it does not contain the analysis samples.

The index CRC covers only the concatenated 48-byte entries, not the index header.
CRC-32C, rather than CRC-32/ISO-HDLC, is used throughout this document. If CRC
verification is too costly for a realtime code path, a reader may defer it, but it
must provide an integrity-verifying mode for validation tools.

CRC-32C uses the reflected Castagnoli polynomial `0x82F63B78`, initial value
`0xFFFFFFFF`, final XOR `0xFFFFFFFF`, and no reflection or augmentation beyond
that standard reflected algorithm. `firstEntryOffset` must equal
`indexOffset + 32`.

## 11. Reader behavior

### Opening and seeking a finalized file

1. Read and validate the 64-byte header.
2. Read `descriptorSize` bytes at `descriptorOffset`.
3. Validate descriptor CRC and parse the descriptor.
4. Read the index header and its entries at `indexOffset`.
5. Validate index counts and bounds.
6. Do not read data payloads yet.
7. For a query, seek to the selected chunk, validate its header and CRC, and
   decode only that payload.

A reader should cache one or two chunks per stream. A Unity timeline playback path
can keep the current and next chunk resident and evict chunks when the playhead
moves away. Loading the whole `.xas` file into a byte array is explicitly
discouraged.

### Sequential streaming

1. Read header and descriptor.
2. Move to the first chunk offset:
   `align8(descriptorOffset + descriptorSize)`.
3. Read each 48-byte chunk header.
4. Skip or decode exactly `payloadSize` bytes.
5. Skip zero padding to the next 8-byte boundary.
6. Stop after the declared chunk count when finalized, or when the index magic is
   encountered for a provisional stream.

The chunk header is self-describing, so a sequential reader does not need the tail
index.

### Scrubbing and boundary cases

- For continuous data, a backward or forward scrub should reuse the cached chunk
  while the requested sample remains in its interval.
- At a chunk boundary, interpolation can require loading the adjacent chunk.
- For event ranges, test every candidate chunk for which `firstTick <= queryEnd`
  and `lastTick >= queryStart`.
- Consumers should return bounded arrays or enumerations rather than constructing
  a whole-file event list.

### Validation rules

A conforming reader must reject:

- Invalid magic, unsupported major version, or truncated fixed structures.
- Offsets or sizes outside the physical file bounds.
- Descriptor or chunk CRC mismatches when integrity checking is enabled.
- A chunk whose `streamId`, kind, or codec does not match its descriptor.
- A non-regular continuous chunk or a payload whose size differs from the record
  count times the known record size.
- Index fields that differ from the corresponding chunk headers.

Malformed length fields must not be used for allocation without checking them
against both the remaining file length and the engine's configured maximum.

## 12. Writer behavior

1. Write a 64-byte provisional header and the JSON descriptor.
2. Pad to an 8-byte boundary.
3. Buffer a single bounded set of continuous frames and flush a complete chunk.
4. Buffer events until a chunk target size is reached or a 32-bit tick delta would
   overflow, then flush a complete chunk.
5. Keep index entries in a compact list while writing data.
6. Append the index.
7. Seek to the header and patch duration, chunk count, index offset, file size,
   finalization flags, and CRC values.
8. Flush the file.

This is compatible with an analysis pipeline that discovers values incrementally.
The writer does not need to retain all continuous frames or all events, only its
current chunk buffers and the small index.

Writers should use 16-64 KiB payload targets. A 64 KiB continuous payload contains
3,276 frames at 20 bytes per frame. With a 512-sample hop at 48 kHz, one chunk
therefore spans about 34.9 seconds. A 32 KiB target spans about 17.4 seconds and
is a conservative initial choice for mobile and XR devices.

For each event chunk, use the first event's start tick as `firstTick` and encode
later starts as deltas. If events remain from a prior temporal range because a
long event crosses chunk boundaries, the writer must retain event order and
calculate the chunk's `lastTick` from all event ends.

## 13. Mapping from analysis version 2

The current `.xaa` and `.json` content maps directly into XAS v1:

| Analysis v2 | XAS v1 |
| --- | --- |
| `info` | Descriptor `info` |
| `continuous[].rms` | Continuous component 0 |
| `continuous[].lows` | Continuous component 1 |
| `continuous[].mids` | Continuous component 2 |
| `continuous[].highs` | Continuous component 3 |
| `continuous[].centroid` | Continuous component 4 |
| `peaks[]` | Peak stream event records |
| `peaks[].channel` | Event `channel` |
| `peaks[].start` | Event `startTick` converted to ticks |
| `peaks[].duration` | Event `durationTicks` converted to ticks |
| `peaks[].magnitude` | Event `magnitude` |
| `peaks[].strength` | Event `strength` |

Writers converting seconds to ticks must use nearest-integer rounding consistently.
Continuous frame ticks should be generated directly from analysis frame indices
and the hop size where available, rather than round-tripping through binary64
seconds.

## 14. Efficiency comparison

For the `continuous` collection, analysis version 2 MessagePack stores a map for
every frame, including property names and a timestamp. The equivalent XAS v1 frame
is fixed at 20 bytes:

```text
5 float32 values = 20 bytes
```

For peak events, the fixed XAS record is also 20 bytes and omits five repeated
property names. Timing is represented with integer sample ticks rather than
binary64 seconds.

The exact file-size saving depends on implementation details and chunk count, but
the important memory bound is structural: a reader retaining two 32 KiB data
payloads uses about 64 KiB of sample/event payload memory regardless of whether
the source audio lasts three minutes or three hours. Only the descriptor and chunk
index grow with the file; each index entry is 48 bytes and represents many seconds
of analysis.

## 15. Evolution and compatibility

- Major version changes may alter fixed structures or payload layouts.
- Minor version changes may add optional header flags or descriptor fields but
  must leave a version-1 reader able to consume known chunks safely.
- Unknown index/header flag bits must be rejected when they may alter layout.
- Unknown descriptor properties must be ignored.
- New streams do not change stream IDs 0 and 1.
- A new numeric codec must receive a new codec ID and an unambiguous descriptor
  encoding name.
- Once assigned, stream, chunk, or codec identifiers must not be repurposed.

Compression can be introduced later as a block-local codec. A compressed chunk must
always remain independently decodable after a single seek, and the index should
add stored and decoded payload sizes in a new major or compatible minor layout.
Version 1 deliberately keeps raw fixed-size payloads so Unity and native readers
require no compression dependency.

## 16. Design rationale

This proposal follows the useful parts of the HTRJ1 record format:

- A fixed, searchable header.
- A versioned stream/chunk model.
- Data emitted in bounded chunks.
- A tail index containing offsets and time ranges.
- Reader caches that hold one active chunk per target.
- Values for evaluation stored compactly in fixed binary layouts.

XAS specializes that model for audio analysis:

- Continuous samples omit timestamps because their timing is regular.
- Discrete events use integer tick deltas and duration.
- The two data classes have distinct records and query semantics.
- Chunk time ranges allow event overlap checks without scanning the event data.
- The descriptor carries the small `info` object and channel descriptions instead
  of duplicating them in each data record.

The result is seekable and cacheable like a recorded trajectory, while retaining a
forward-only chunk path for streaming analysis output.
