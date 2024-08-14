from pprint import pformat
from docutils import nodes
from importlib import import_module
from pprint import pformat
from docutils.parsers.rst import Directive
from sphinx import addnodes
import subprocess
import os
import sys

parent_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)


class PrettyPrintIterable(Directive):
    required_arguments = 1

    def run(self):
        module_path, member_name = self.arguments[0].rsplit(".", 1)
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
        literal["language"] = "python"

        return [addnodes.desc_content("", literal)]


project = 'ai3'
copyright = '2024, Timothy Cronin'
author = 'Timothy Cronin'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'breathe',
]

master_file = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']

autodoc_member_order = 'bysource'

subprocess.call('make clean', shell=True)
subprocess.call('doxygen', shell=True)

breathe_projects = {
    "ai3": os.path.join(os.getcwd(), "doxygen", "xml")
}
breathe_default_project = "ai3"


def setup(app):
    app.add_directive('pprint', PrettyPrintIterable)
