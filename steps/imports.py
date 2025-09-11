import pandas as pd

class ImportData:
    """Class to handle data import from CSV files."""
    def __init__(self, paths_dict):
        self.paths_dict = paths_dict
        self.dfs = {}

    def load_data(self, path):
        try:
            data = pd.read_csv(path)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def  _execute(self):
        for key, path in self.paths_dict.items():
            self.dfs[key] = self.load_data(path)
