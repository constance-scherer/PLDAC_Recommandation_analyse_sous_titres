#! /usr/bin/env python
'''
Module to remove empty folders recursively. Can be used as standalone script or be imported into existing script.
'''

import os, sys

def removeEmptyFolders(path, removeRoot=True):
  'Function to remove empty folders'
  if not os.path.isdir(path):
    return

  # remove empty subfolders
  files = os.listdir(path)
  if len(files):
    for f in files:
      fullpath = os.path.join(path, f)
      if os.path.isdir(fullpath):
        removeEmptyFolders(fullpath)

  # if folder empty, delete it
  files = os.listdir(path)
  if len(files) == 0 and removeRoot:
    
    os.rmdir(path)
    
def usageString():
  'Return usage string to be output in error cases'
  return 'Usage: %s directory [removeRoot]' % sys.argv[0]

if __name__ == "__main__":
  removeRoot = True
  
  if len(sys.argv) < 1:
    
    sys.exit(usageString())
    
  if not os.path.isdir(sys.argv[1]):
    
    sys.exit(usageString())
    
  if len(sys.argv) == 2 and sys.argv[2] != "False":
    
    sys.exit(usageString())
  else:
    removeRoot = False
    
  removeEmptyFolders(sys.argv[1], removeRoot)