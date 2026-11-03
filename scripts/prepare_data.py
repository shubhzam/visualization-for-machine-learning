"""Install official VisEval data from a local ZIP or the pinned official release tree."""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

COMMIT='55162fac5765d8873180d962cce7e20e75b2c092'
URL=f'https://raw.githubusercontent.com/microsoft/VisEval/{COMMIT}/viseval_dataset.zip'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--destination',type=Path,default=Path('data'))
    args=parser.parse_args()
    if args.archive:
        content=args.archive.read_bytes()
    else:
        with urllib.request.urlopen(URL,timeout=120) as response:
            content=response.read()
    destination=args.destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            path=(destination/info.filename).resolve()
            if not path.is_relative_to(destination):
                raise ValueError('Unsafe archive path')
        archive.extractall(destination)
    (destination/'VISEVAL_SOURCE.txt').write_text(f'Source: {URL}\nCommit: {COMMIT}\nZIP SHA256: {hashlib.sha256(content).hexdigest()}\nLicense: MIT (Microsoft VisEval)\n')
    print('Dataset ready:',destination/'visEval_dataset')

if __name__=='__main__':
    main()
