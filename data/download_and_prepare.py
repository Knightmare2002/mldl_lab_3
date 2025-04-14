import os
import shutil
import zipfile
import urllib.request

def download_and_prepare_tiny_imagenet(dest_dir='tiny-imagenet'):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(dest_dir, "tiny-imagenet-200.zip")

    os.makedirs(dest_dir, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    annotations_path = os.path.join(dest_dir, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
    with open(annotations_path) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            cls_dir = os.path.join(dest_dir, 'tiny-imagenet-200', 'val', cls)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copyfile(
                os.path.join(dest_dir, 'tiny-imagenet-200', 'val', 'images', fn),
                os.path.join(cls_dir, fn)
            )
    shutil.rmtree(os.path.join(dest_dir, 'tiny-imagenet-200', 'val', 'images'))
