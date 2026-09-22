"""SRT writer sanity checks (no video needed). Run: make check"""
from transcriber import Segment, segments_to_srt, format_timestamp

assert format_timestamp(3661.5) == "01:01:01,500"
assert format_timestamp(0) == "00:00:00,000"
assert format_timestamp(59.999) == "00:00:59,999"

srt = segments_to_srt([Segment(0, 1.234, "hello"), Segment(2, 3, "world")])
expected = (
    "1\n"
    "00:00:00,000 --> 00:00:01,234\n"
    "hello\n"
    "\n"
    "2\n"
    "00:00:02,000 --> 00:00:03,000\n"
    "world\n"
)
assert srt == expected, srt

assert segments_to_srt([]) == ""
print("check ok")
