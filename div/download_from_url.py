#main source: https://pytorch.org/text/_modules/torchtext/utils.html#download_from_url
import requests
import os
import re


def download_from_url(url, path=None, root='.data', overwrite=False):


    def _process_response(r, root, filename):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        if filename is None:
            d = r.headers['content-disposition']
            filename = re.findall("filename=\"(.+)\"", d)
            if filename is None:
                raise RuntimeError("Filename could not be autodetected")
            filename = filename[0]
        path = os.path.join(root, filename)
        if os.path.exists(path):
            print('File %s already exists.' % path)
            if not overwrite:
                return path
            print('Overwriting file %s.' % path)
        print('Downloading file {} to {} ...'.format(filename, path))

        with open(path, "wb") as file:
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    file.write(chunk)
        print('File {} downloaded.'.format(path))
        return path

    if path is None:
        _, filename = os.path.split(url)
    else:
        root, filename = os.path.split(path)

    if not os.path.exists(root):
        raise RuntimeError(
            "Download directory {} does not exist. "
            "Did you create it?".format(root))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        return _process_response(response, root, filename)
    else:
        # google drive links get filename from google drive
        filename = None

    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    return _process_response(response, root, filename)
