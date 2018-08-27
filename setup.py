#!/usr/bin/env python

import os

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('polyhedron',parent_package,top_path)
    cddlib_dir = 'cddlib-094d-p1'
    sources = ['cddcore.c','cddlp.c','cddmp.c','cddio.c',
               'cddlib.c','cddproj.c','setoper.c'
               ]
    sources = [os.path.join(cddlib_dir,'lib-src',fn) for fn in sources]
    include_dirs = [os.path.join(cddlib_dir,'lib-src')]
    config.add_library('cddlib',
                       sources = sources,
                       include_dirs = include_dirs
                       )

    config.add_extension('_cdd',
                         sources = ['_cddmodule.c'],
                         libraries = ['cddlib'],
                         include_dirs = include_dirs
                         )
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration,
          version='0.2.1')
