import gdown
import tarfile
import os

os.chdir('data')
url = 'https://drive.google.com/drive/folders/19bvwt9CdLHqdVBGZUZ3-ex9OD24y7bOu?usp=drive_link'

output = 'data.tgz'
gdown.download(url, output, quiet=False)

tar = tarfile.open(output, "r:")
tar.extractall()
tar.close()