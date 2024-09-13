import re
import os
import ai3
import textwrap
import inspect


def prune_rst_links_and_remove_args(obj) -> str:
    docstring = inspect.getdoc(obj)
    assert (docstring)

    docstring = re.sub(r':func:`([^`]+)`', r'`\1`', docstring)
    docstring = re.sub(r':class:`([^`]+)`', r'`\1`', docstring)
    docstring = re.sub(r':type:`([^`]+)`', r'`\1`', docstring)
    docstring = textwrap.dedent(docstring).strip()

    paragraphs = docstring.split('\n\n')
    paragraphs = [p for p in paragraphs if not p.startswith('Args:')]

    docstring = '\n\n'.join(paragraphs)

    return docstring


def clean_rst_prolog():
    from docs.conf import rst_prolog
    if rst_prolog.startswith('\n'):
        rst_prolog = rst_prolog.lstrip('\n')

    if not rst_prolog.endswith('\n'):
        rst_prolog += '\n'

    return rst_prolog


if __name__ == "__main__":
    with open(os.path.join('docs', 'intro.rst'), 'r') as index_file:
        index_content = index_file.read()
    with open(os.path.join('docs', 'algo_platform_tables.rst'), 'r') as index_file:
        algo_platform_tables = index_file.read()
    with open('README.rst', 'w') as readme_file:
        readme_file.write(clean_rst_prolog())
        readme_file.write('\n')
        readme_file.write(index_content)

        doc = prune_rst_links_and_remove_args(ai3)
        readme_file.write(''.join(doc.splitlines(keepends=True)[1:]))
        readme_file.write('\n\n')

        sc_doc = prune_rst_links_and_remove_args(ai3.swap_conv2d)

        readme_file.writelines(['*swap_conv2d*\n',
                                '~~~~~~~~~~~~~\n',
                                sc_doc])
        readme_file.write('\n\n')

        sb_doc = prune_rst_links_and_remove_args(ai3.swap_backend)
        readme_file.writelines(['*swap_backend*\n',
                                '~~~~~~~~~~~~~~\n',
                                sb_doc])
        readme_file.write('\n\n')

        readme_file.write(algo_platform_tables)
