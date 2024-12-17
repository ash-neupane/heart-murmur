import pathlib

class FileLoader:
    """
    Handles loading all types of file available with the dataset and parses it.
    1-1 relationship w/ a file
    """
    def __init__(self, root_path):
        """
        """
        if isinstance(root_path, pathlib.Path):
            self.root = root_path
        else:
            self.root = pathlib.Path(root_path)

    def load(self, filename):
        """
        routes to the data_file specific loader
        """
        pass

    def _load_tsv(self, filename):
        """
        """
        pass

    def _load_wav(self, filename):
        """
        
        """
        pass

    def _load_hea(self, filename):
        """
        """
        pass

    def _load_annotation_txt(self, filename):
        """
        """
        pass

    def _load_csv(self, filename):
        """
        """
        pass

    def _load_records(self, filename):
        """
        """
        pass