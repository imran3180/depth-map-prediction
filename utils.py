import os

def check_folder(log_dir):
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  return log_dir