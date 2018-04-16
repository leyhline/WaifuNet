#!/usr/bin/env python3

import re
from collections import Counter


LOG_FILENAME = "resize_images.log"


if __name__ == "__main__":
    with open(LOG_FILENAME, "r") as fd:
        error_counter = (line.rsplit("(")[0].rstrip() for line in fd if line)
        error_counter = Counter(re.sub(r"""\d+ extraneous bytes before marker 0x.{2}""", "some extraneous bytes before marker", line) for line in error_counter)
    for message, count in error_counter.most_common():
        print(message, "|", count)
