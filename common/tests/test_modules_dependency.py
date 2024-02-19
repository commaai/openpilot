import unittest
import os


def get_main_directories(path=''):
    main_directories = []
    for item in os.listdir(path):
        if item.startswith('.'):
            continue
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            main_directories.append(os.path.basename(item_path))
    return main_directories


def parse_file(file_path='', leading_folder='', errors=None, main_dirs=None):
    if errors is None:
        errors = []
    excluded_main_dirs = [item for item in main_dirs if item != leading_folder]
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, start=1): 
            if 'import' in line or 'from' in line:
                for f_name in excluded_main_dirs:
                    if ' '+f_name+'.' in line.strip() and not 'openpilot.' in line.strip():
                        if not f_name+'_repo' in file_path:
                            error_msg = f"Error: File: {file_path}, External folder: {f_name}, Line #{line_num}: {line.strip()}"
                            errors.append(error_msg)
    return errors


class TestParseFiles(unittest.TestCase):
    def setUp(self):
        self.current_dir = os.path.dirname(__file__)
        self.relative_path = ""

        while not os.path.exists(os.path.join(self.current_dir, "openpilot")) and self.current_dir != os.path.dirname(self.current_dir):
            self.current_dir = os.path.dirname(self.current_dir)
            self.relative_path = os.path.join("..", self.relative_path)

        if os.path.exists(os.path.join(self.current_dir, "openpilot")):
            self.root_dir = self.current_dir

        self.main_dirs = get_main_directories(self.root_dir)
        self.excluded_dir_list = ['cereal', 'body', 'rednose', 'rednose_repo', 'opendbc', 'panda', 'generated']
        self.main_dirs = [dir for dir in self.main_dirs if dir not in self.excluded_dir_list]


    def test_parse_files(self):
        error_list = []

        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            for name in dirs:
                dir_path = os.path.join(root, name)
                if dir_path.split(self.root_dir)[1].startswith('/.'):
                    continue
                leading_folder = dir_path.split(self.root_dir)[1].split("/")[1]
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if file_path.endswith('.py'):
                        parse_file(file_path, leading_folder, error_list, self.main_dirs)

        error_message = "\n".join(error_list)
        self.assertEqual(len(error_list), 0, f"Errors found, unexpected dependencies! : \n{error_message}")


if __name__ == '__main__':
    unittest.main()