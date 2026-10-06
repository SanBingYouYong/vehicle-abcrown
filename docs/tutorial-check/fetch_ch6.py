"""Fetch the Chapter 6 (Vancomycert) case-study files from the 5 GB Zenodo
artefact without downloading all of it: a zip's index is at its end, so HTTP
range requests can read the index and then only the wanted members.

    python tools/tutorial-check/fetch_ch6.py   # -> tools/tutorial-check/ch6-data/
"""

import io
import os
import shutil
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/20717396/files/vehicle-artefact.zip/content"
MEMBERS = ["pk.vcl", "pk.onnx", "pk_mean.idx", "pk_std.idx"]
PREFIX = "vehicle-artefact/medical-case-study/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch6-data")


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    os.makedirs(OUT, exist_ok=True)
    z = zipfile.ZipFile(io.BufferedReader(HTTPFile(URL), buffer_size=1 << 20))
    for m in MEMBERS:
        with z.open(PREFIX + m) as src, open(os.path.join(OUT, m), "wb") as dst:
            shutil.copyfileobj(src, dst)
        print("fetched", m)


if __name__ == "__main__":
    main()
