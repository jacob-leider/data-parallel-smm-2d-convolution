import tomllib
from pprint import pformat
from docutils import nodes
from importlib import import_module
from pprint import pformat
from docutils.parsers.rst import Directive
from sphinx import addnodes
import subprocess
import os
import sys

cur_directory = os.path.dirname(__file__)
parent_directory = os.path.abspath(
    os.path.join(cur_directory, '..'))
sys.path.append(parent_directory)

static_directory = os.path.join(cur_directory, '_static')
if not os.path.exists(static_directory):
    os.makedirs(static_directory)


class PrettyPrintIterable(Directive):
    required_arguments = 1

    def run(self):
        module_path, member_name = self.arguments[0].rsplit('.', 1)
        module = import_module(module_path)
        member = getattr(module, member_name)

        code = pformat(
            member,
            indent=2,
            width=80,
            depth=3,
            compact=False,
            sort_dicts=False,
        )

        literal = nodes.literal_block(code, code)
        literal['language'] = 'python'

        return [addnodes.desc_content('', literal)]


with open(os.path.join(parent_directory, 'pyproject.toml'), 'rb') as f:
    pyproject_data = tomllib.load(f)

name = pyproject_data.get('project', {}).get('name', '')
repo = pyproject_data.get('project', {}).get('homepage', '')
docs = pyproject_data.get('project', {}).get('documentation', '')
repo_main = repo + '/tree/main'
repo_src = repo_main + '/src/ai3'
repo_csrc = repo_src + '/csrc'

rst_prolog = f'''
.. _repo: {repo}
.. |repo| replace:: **Source Code**
.. _custom: {repo_src + '/custom'}
.. |custom| replace:: custom
.. _custom_cmake: {repo_src + '/cmake/custom.cmake'}
.. |custom_cmake| replace:: *custom.cmake*
.. _doc: {docs}
.. |doc| replace:: **Documentation**
.. |name| replace:: *{name}*
'''

project = name
copyright = '2024, Timothy Cronin'
author = 'Timothy Cronin'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'breathe',
]

master_file = 'index'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_show_sourcelink = False
html_sidebars = {
 '**': []
}
html_theme_options = {
        'navbar_align': 'left',
        'icon_links': [
            {
                'name': 'GitHub',
                'url': repo,
                'icon': 'fa-brands fa-github',
                'type': 'fontawesome',
                },
            {
                'name': 'PyPI',
                'url': docs,
                'icon': 'fa-brands fa-python',
                'type': 'fontawesome',
                }
            ]
        }

breathe_projects = {
        name: os.path.join(os.getcwd(), 'doxygen', 'xml')
}
breathe_default_project = name

doctest_global_setup = '''
import ai3
import torch
import torchvision
from example.manual_conv2d import ConvNet
'''

def setup(app):
    app.add_directive('pprint', PrettyPrintIterable)
    subprocess.call('make clean', shell=True, cwd=cur_directory)
    subprocess.call(
        f'PROJECT_NAME=*{name}* REPO_MAIN={repo_main} REPO_SRC={repo_src} REPO_CSRC={repo_csrc} doxygen',
        shell=True, cwd=cur_directory)
