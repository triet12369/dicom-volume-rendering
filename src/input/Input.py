import vtk
import os
# input base classes

class Input(object):
    def __init__(self):
        self._data = None
        self._filenames = None
        pass

    def get_data(self):
        return self._data

    def get_filenames(self):
        return self._filenames


class DirectoryInput(Input):
    def load_from_dir(self, directory):
        if os.path.isdir(directory):
            dirs = list(map(lambda x: os.path.join(directory, x), os.listdir(directory)))
            print("dirs", dirs)
            self._filenames = dirs
        else:
            print("Error in DirectoryInput: not a directory")
