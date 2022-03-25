import distutils.ccompiler
import distutils.sysconfig
import re
import numpy as np
import subprocess
from os.path import join, dirname, realpath, splitext, exists, lexists, islink, relpath
from os import walk, sep, remove, listdir, stat, symlink

from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.command import build as build_module, clean as clean_module

SCRIPT_DIR = dirname(realpath(__file__))
WIKIDATA_LINKER_SOURCE_DIR = join(SCRIPT_DIR, "src")
WIKIDATA_LINKER_MODULE_NAME = "wikidata_linker_utils_cython"
WIKIDATA_LINKER_INTERNAL_MODULE_NAME = WIKIDATA_LINKER_MODULE_NAME

version_file = join(SCRIPT_DIR, WIKIDATA_LINKER_MODULE_NAME, "VERSION")
if exists(version_file):
    with open(version_file) as f:
        VERSION = f.read().strip()
else:
    VERSION = "0.0.0"

FIND_CYTHON_CPP_INCLUDES = re.compile('^[^#]+extern\W+from\W+"(?P<path>[^"]+)"')


def extract_cython_cpp_include_paths(file_path):
    with open(file_path, "rt") as f:
        for line in f:
            res = FIND_CYTHON_CPP_INCLUDES.search(line)
            if res is not None:
                yield res.groupdict()['path']


def path_to_module_name(path):
    BASE_DIRS = ["cython"]
    relative_path = relpath(path, join(WIKIDATA_LINKER_SOURCE_DIR))
    path_no_ext, _ = splitext(relative_path)
    for base_dir in BASE_DIRS:
        if path_no_ext.startswith(base_dir):
            return path_no_ext.lstrip(base_dir + sep).replace(sep, '.')
    raise Exception("Cannot convert path %r to module name" % (relative_path,))


def find_files_by_suffix(path, suffix):
    """Recursively find files with specific suffix in a directory"""
    for relative_path, dirs, files in walk(path):
        for fname in files:
            if fname.endswith(suffix):
                yield join(path, relative_path, fname)


class clean(clean_module.clean):
    """Make a `cleanall` rule to get rid of intermediate and library files"""

    def run(self):
        print("Cleaning up cython files...")
        # Just in case the build directory was created by accident,
        # note that shell=True should be OK here because the command is constant.
        for place in ["build",
                      join("src", "cython", WIKIDATA_LINKER_INTERNAL_MODULE_NAME, "*.c"),
                      join("src", "cython", WIKIDATA_LINKER_INTERNAL_MODULE_NAME, "*.cpp"),
                      join("src", "cython", WIKIDATA_LINKER_INTERNAL_MODULE_NAME, "*.so")]:
            subprocess.Popen("rm -rf %s" % (place,),
                             shell=True,
                             executable="/bin/bash",
                             cwd=SCRIPT_DIR)


compiler = distutils.ccompiler.new_compiler()
distutils.sysconfig.customize_compiler(compiler)
BLACKLISTED_COMPILER_SO = ['-Wp,-D_FORTIFY_SOURCE=2']
build_ext.compiler = compiler

ext_modules = []

for pyx_file in find_files_by_suffix(join(WIKIDATA_LINKER_SOURCE_DIR, "cython"), ".pyx"):
    extra_cpp_sources = []

    # pxd files are like header files for pyx files
    # and they can also have relevant includes.
    relevant_files = [pyx_file]
    pxd_file = pyx_file[:-3] + "pxd"
    if exists(pxd_file):
        relevant_files.append(pxd_file)

    # find all the cpp files referenced from pyx files
    # and if some exist in src/cpp folder, compile them
    # as well
    for cpy_file in relevant_files:
        for header_path in extract_cython_cpp_include_paths(cpy_file):
            hypothetical_source_path = header_path.rstrip('.h') + '.cpp'
            hypothetical_source_full_path = join(WIKIDATA_LINKER_SOURCE_DIR, 'cpp', hypothetical_source_path)
            if exists(hypothetical_source_full_path):
                extra_cpp_sources.append(hypothetical_source_full_path)
    ext_modules.append(Extension(
        name=path_to_module_name(pyx_file),
        sources=[pyx_file] + extra_cpp_sources,
        library_dirs=[],
        language='c++',
        extra_compile_args=['-std=c++11', '-Wno-unused-function',
                            '-Wno-sign-compare', '-Wno-unused-local-typedef',
                            '-Wno-undefined-bool-conversion', '-O3',
                            '-Wno-reorder'],
        extra_link_args=[],
        libraries=[],
        extra_objects=[],
        include_dirs=[join(WIKIDATA_LINKER_SOURCE_DIR, "cpp"),
                      np.get_include()]
    ))

################################################################################
##                      FIND PYTHON PACKAGES                                  ## # noqa
################################################################################

py_packages = []
# for file in find_files_by_suffix(join(WIKIDATA_LINKER_SOURCE_DIR, "python"), ".py"):
#     module_path = dirname(file)
#     py_packages.append(path_to_module_name(module_path))


################################################################################
##              BUILD COMMAND WITH EXTRA WORK WHEN DONE                       ## # noqa
################################################################################


def symlink_built_package(module_name, dest_directory):
    build_dir_contents = listdir(join(SCRIPT_DIR, "build"))
    lib_dot_fnames = []
    for name in build_dir_contents:
        if name.startswith("lib."):
            lib_dot_fnames.append(join(SCRIPT_DIR, "build", name))
    # get latest lib. file created and symlink it to the project
    # directory for easier testing
    lib_dot_fnames = sorted(
        lib_dot_fnames,
        key=lambda name: stat(name).st_mtime,
        reverse=True
    )
    if len(lib_dot_fnames) == 0:
        return

    most_recent_name = join(lib_dot_fnames[0], module_name)
    symlink_name = join(dest_directory, module_name)

    if lexists(symlink_name):
        if islink(symlink_name):
            remove(symlink_name)
        else:
            print(
                ("non symlink file with name %r found in project directory."
                 " Please remove to create a symlink on build") % (symlink_name,)
            )
            return

    symlink(most_recent_name,
            symlink_name,
            target_is_directory=True)
    print("Created symlink pointing to %r from %r" % (
        most_recent_name,
        join(SCRIPT_DIR, module_name)
    ))


class build_with_posthooks(build_module.build):
    def run(self):
        build_module.build.run(self)


# Make a `cleanall` rule to get rid of intermediate and library files
class clean_with_posthooks(clean_module.clean):
    def run(self):
        clean_module.clean.run(self)

        # remove cython generated sources
        for file_path in find_files_by_suffix(join(WIKIDATA_LINKER_SOURCE_DIR, 'cython'), '.cpp'):
            remove(file_path)


setup(
    name=WIKIDATA_LINKER_MODULE_NAME,
    version=VERSION,
    cmdclass={"build": build_with_posthooks, 'build_ext': build_ext, 'clean': clean_with_posthooks},
    install_requires=["numpy"],
    extras_require={"dev": []},
    author="anonymous author",
    language='c++',
    author_email="anonymous@gmail.com",
    ext_modules=ext_modules,
    description="Utilities for running wikidata_linker.",
    package_dir={'': join(WIKIDATA_LINKER_SOURCE_DIR, 'cython')},
    packages=py_packages,
)
