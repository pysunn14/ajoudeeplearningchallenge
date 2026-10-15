import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'project_paths.py'


def load_paths(source=SOURCE):
    spec = importlib.util.spec_from_file_location('project_paths', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectPathsTests(unittest.TestCase):
    def test_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                paths = load_paths()
                self.assertEqual(
                    paths.project_path('dataset', 'sample.parquet'),
                    str(SOURCE.parents[1] / 'dataset' / 'sample.parquet'),
                )
            finally:
                os.chdir(previous)

    def test_moved_checkout_with_spaces_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / '대회 프로젝트'
            source = root / 'src' / SOURCE.name
            source.parent.mkdir(parents=True)
            shutil.copyfile(SOURCE, source)
            paths = load_paths(source)
            self.assertEqual(paths.project_path('model', 'bert'), str(root / 'model' / 'bert'))
            with patch.dict(os.environ, {'HOME': directory}, clear=True):
                paths.configure_evaluation_cache()
                self.assertEqual(os.environ['HOME'], directory)
                self.assertEqual(os.environ['HF_HOME'], str(root / 'hf_cache'))
                self.assertTrue((root / 'hf_cache' / 'datasets').is_dir())

    def test_explicit_cache_settings_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                'HOME': str(root / 'home'),
                'XDG_CACHE_HOME': str(root / 'xdg'),
                'HF_HOME': str(root / 'hf'),
                'TRANSFORMERS_CACHE': str(root / 'transformers'),
                'HF_DATASETS_CACHE': str(root / 'datasets'),
                'TORCH_HOME': str(root / 'torch'),
                'CLIP_CACHE_DIR': str(root / 'clip'),
            }
            with patch.dict(os.environ, expected, clear=True):
                paths = load_paths()
                paths.configure_evaluation_cache()
                paths.configure_evaluation_cache()
                self.assertEqual(dict(os.environ), expected)
                for key in ('HF_HOME', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'TORCH_HOME', 'CLIP_CACHE_DIR'):
                    self.assertTrue(Path(expected[key]).is_dir())

    def test_default_subdirectories_follow_custom_hf_home(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'HF_HOME': directory}, clear=True):
                load_paths().configure_evaluation_cache()
                self.assertEqual(os.environ['HF_DATASETS_CACHE'], str(Path(directory) / 'datasets'))
                self.assertEqual(os.environ['TRANSFORMERS_CACHE'], str(Path(directory) / 'transformers'))


if __name__ == '__main__':
    unittest.main()
