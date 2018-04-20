
import os.path
import sys

import logging
logger = logging.getLogger(__name__)

try:
  from google.cloud import storage
except ImportError as e:
  logger.warn("Could not import google.cloud, will not save to bucket")
  pass



class FileWritey(object):
  """Tries to write on traditional filesystem and Google Cloud storage"""

  def __init__(self, args, filename):
    self.args = args
    self.filename = filename
    self.trad_file = None

  def copy_to_bucket(self):
    if 'google.cloud' in sys.modules and self.args.bucket is not None and self.args.gcs_dir is not None:
      client = storage.Client()
      bucket = client.get_bucket(self.args.bucket)
      blob2 = bucket.blob(os.path.join(self.args.gcs_dir, self.filename))
      blob2.upload_from_filename(filename=os.path.join(self.args.output_dir, self.filename))

  def __enter__(self):
    os.makedirs(self.args.output_dir, exist_ok=True)
    self.trad_file = open(os.path.join(self.args.output_dir, self.filename), 'w')
    return self.trad_file

  def __exit__(self, type, value, traceback):
    self.trad_file.close()
    self.copy_to_bucket()

    

