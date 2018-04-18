

import tensorflow as tf
import csv
import sys
import os.path

import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Some environments have these components. Try to load them, and if they
# are not available, gracefully fallback
# --------------------------------------------------------------------------

try:
  import matplotlib
  matplotlib.use("Agg") # Work in terminal

  from IPython.display import clear_output
  from matplotlib import pyplot as plt
except ImportError as e:
  logger.warn("Could not import matplotlib, no graphs will be generated")
  pass


try:
  from google.colab import auth
  from googleapiclient.discovery import build
  from googleapiclient.http import MediaFileUpload
except ImportError as e:
  logger.warn("Could not import googleapiclient, will not save to google drive")
  pass


from .writer import TextWritey
from .decorator import debounce

class Ploty(object):

  def __init__(self, 
    args,
    title='', x='', y='', 
    legend=True, log_y=False, log_x=False, 
    clear_screen=True, terminal=True
  ):

    self.args = args
    self.title = title
    self.label_x = x
    self.label_y = y
    self.log_y = log_y
    self.log_x = log_x
    self.clear_screen = clear_screen
    self.legend = legend
    self.terminal = terminal

    self.header = ["x", "y", "label"]
    self.datas = {}
    
    self.c_i = 0

    try:
      self.cmap = plt.cm.get_cmap('hsv', 10)
      self.fig = plt.figure()
      self.ax = self.fig.add_subplot(111)
      
      if self.log_x:
        self.ax.set_xscale('log')
      
      if self.log_y:
        self.ax.set_yscale('log')
    except Exception:
      self.cmap = None
      pass
    
    
  def ensure(self, name, extra_data):
    if name not in self.datas:
      self.datas[name] = {
        "x": [],
        "y": [],
        "m": ".",
        "l": '-'
      }

      if self.cmap is not None:
        self.datas[name]["c"] = self.cmap(self.c_i)

      for i in extra_data.keys():
        self.datas[name][i] = []
        if i not in self.header:
          self.header.append(i)

      self.c_i += 1

  # This method assumes extra_data will have the same keys every single call, otherwise csv writing will crash
  def add_result(self, x, y, name, marker="o", line="-", extra_data={}):
    self.ensure(name, extra_data)
    self.datas[name]["x"].append(x)
    self.datas[name]["y"].append(y)
    self.datas[name]["m"] = marker
    self.datas[name]["l"] = line

    for key, value in extra_data.items():
      self.datas[name][key].append(value)

    if self.terminal:
      print('{{"metric": "{}", "value": {}, "x": {} }}'.format(name,y,x))


  
  def render(self):
    self.render_pre()
    
    for k, d in self.datas.items():
      plt.plot(d['x'], d['y'], d["l"]+d["m"], label=k)
      
    self.render_post()
      
      
  def render_pre(self):
    if self.clear_screen and not self.terminal:
      clear_output()

    plt.cla()
    
  def render_post(self):
    img_name = self.args.output_dir + '/' + self.title.replace(" ", "_") + '.png'
    
    artists = []

    self.fig.suptitle(self.title, fontsize=14, fontweight='bold')
    self.ax.set_xlabel(self.label_x)
    self.ax.set_ylabel(self.label_y)
    
    if self.legend:
        lgd = plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        artists.append(lgd)
        
    try:
        os.remove(img_name)
    except FileNotFoundError:
        pass
        
    plt.savefig(img_name, bbox_extra_artists=artists, bbox_inches='tight')
    logger.info("Saved image: " + img_name)
    
    if not self.terminal:
	    plt.show()

  @property
  def filename(self):
    return self.title.replace(" ", "_") + '.csv'
  
  @property
  def file_path(self):
    return os.path.join(self.args.output_dir, self.filename)

  def write(self):
    self.save_csv()

    if 'matplotlib' in sys.modules:
      self.render()


    
  
  def save_csv(self):
    try:
      os.remove(self.file_path)
    except FileNotFoundError:
      pass
    
    with TextWritey(self.args, self.filename) as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(self.header)
      
      for k, d in self.datas.items():
        for i in range(len(d["x"])):
          row = [
            k if h == "label" else d[h][i] for h in self.header
          ]
          writer.writerow(row)

    logger.info("Saved CSV: " + self.file_path)
    
  
  def copy_to_drive(self, snapshot=False):   
    auth.authenticate_user()
    drive_service = build('drive', 'v3')
    
    if snapshot:
      name = self.title + "_latest"
    else:
      name = self.title +'_' + str(datetime.now())
      
    def do_copy(source_name, dest_name, mime):
      file_metadata = {
        'name': dest_name,
        'mimeType': mime
      }
      media = MediaFileUpload(self.args.output_dir + source_name, 
                              mimetype=file_metadata['mimeType'],
                              resumable=True)
      
      created = drive_service.files().create(body=file_metadata,
                                             media_body=media,
                                             fields='id').execute()
      
    do_copy(self.title+'.csv', name + '.csv', 'text/csv')
    do_copy(self.title+'.png', name + '.png', 'image/png')


  