from setuptools import setup
from setuptools.extension import Extension
import os
import sys
import stat
import shutil
import glob
from pprint import pprint

if os.path.isabs(__file__):
	script_dir = os.path.dirname(__file__)
else:
	script_dir = os.path.dirname(os.path.abspath(__file__))

def package_files(package_name, dir_name):
	paths = []
	for (path, directories, filenames) in os.walk(os.path.join(package_name, dir_name)):
		for filename in filenames:
			paths.append(os.path.relpath(os.path.join(path, filename), package_name))
	return paths

def no_cythonize(extensions):
	for extension in extensions:
		sources = []
		for sfile in extension.sources:
			path, ext = os.path.splitext(sfile)
			if ext in ('.pyx', '.py'):
				if extension.language == 'c++':
					ext = '.cpp'
				else:
					ext = '.c'
				sfile = path + ext
			sources.append(sfile)
		extension.sources[:] = sources
	return extensions

extra_libs_in = [ "CharLS", "fnv", "fnvfile", "fnvreduce", "minizip" ]

lib_dir = None
bin_dir = None
inc_dir = None
sdk = None
remove_args = []

use_cython = False
copy_files = False
for i in range(len(sys.argv)):
	a = sys.argv[i]
	if a.startswith("sdk="):
		sdk = a[4:]
		remove_args.append(a)
	elif a.startswith("incdir="):
		inc_dir = a[7:]
		remove_args.append(a)
	elif a.startswith("libdir="):
		lib_dir = a[7:]
		remove_args.append(a)
	elif a.startswith("bindir="):
		bin_dir = a[7:]
		remove_args.append(a)
	elif a == "--cythonize":
		use_cython = True
		remove_args.append(a)
	elif a == "--copy-files":
		copy_files = True
		remove_args.append(a)

for a in remove_args:
	sys.argv.remove(a)

if sys.platform.startswith('win32'):
    if lib_dir is None or bin_dir is None:
        lib_dirs = ["../../tmp/build-msvc2015_64-Release/lib", "../lib/Release"]
        dylib_dirs = ["../../tmp/build-msvc2015_64-Release/bin", "../bin/Release"]
    else:
        lib_dirs = [ lib_dir ]
        dylib_dirs = [ bin_dir ]
    lib_prefix = ""
    lib_suffix = ".dll"
    extra_libs_in += [ "sqlite3", "zlib1" ]
elif sys.platform.startswith('darwin'):
	if lib_dir is None:
		lib_dirs = ["../../tmp/build-clang_64-Release/lib", "../lib"]
	else:
		lib_dirs = [ lib_dir ]
	dylib_dirs = lib_dirs
	lib_prefix = "lib"
	lib_suffix = ".dylib"
	extra_libs_in += [ "tbb" ]
elif sys.platform.startswith('linux'):
	if lib_dir is None:
		lib_dirs = ["../../tmp/build-gcc_64-Release/lib", "../lib"]
	else:
		lib_dirs = [ lib_dir ]
	dylib_dirs = lib_dirs
	lib_prefix = "lib"
	lib_suffix = ".so"
	extra_libs_in += [ "tbb" ]
else:
	print("unsupported platform")
	exit(1)

# runtime_library_dirs causes an exception on Windows (MSVC)
# it doesn't work on macOS and is silently ignored
# and it doesn't do what I want it to on Linux
# so convert runtime_library_dirs to link flags that do what I want
# also there is no rpath on Windows so this needs to be emulated with delay load and a preprocessor macro to pass the "rpath" to the delay load helper
def MyExtension(name, sources, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_compile_args=None, extra_link_args=None, language="c++"):
	if sys.platform.startswith('win32'):
		if define_macros is None:
			define_macros = []
		if runtime_library_dirs is not None:
			define_macros.append(('DLLDIR', os.pathsep.join(runtime_library_dirs)))
			runtime_library_dirs = None
		if libraries is not None:
			if extra_link_args is None:
				extra_link_args = []
			for lib in libraries:
				extra_link_args.append('/DELAYLOAD:%s.dll' % lib)
		if extra_compile_args is None:
			extra_compile_args = []
		extra_compile_args.append("/wd4190")
		extra_compile_args.append("/wd4996")
		extra_compile_args.append("/wd4251")
		extra_compile_args.append("/wd4275")
		define_macros.append(('_CRT_SECURE_NO_WARNINGS', ))
	elif sys.platform.startswith('darwin'):
		if runtime_library_dirs is not None:
			if extra_link_args is None:
				extra_link_args = []
			for rp in runtime_library_dirs:
				extra_link_args.append("-Wl,-rpath,@loader_path/" + rp)
			runtime_library_dirs = None
		if extra_compile_args is None:
			extra_compile_args = []
		extra_compile_args.append("-std=c++11")
		extra_compile_args.append("-Wno-return-type-c-linkage")
	elif sys.platform.startswith('linux'):
		if runtime_library_dirs is not None:
			if extra_link_args is None:
				extra_link_args = []
			for rp in runtime_library_dirs:
				extra_link_args.append("-Wl,-rpath,$ORIGIN/" + rp)
			runtime_library_dirs = None
		if extra_compile_args is None:
			extra_compile_args = []
		extra_compile_args.append("-std=c++11")

	return Extension(name, sources, include_dirs=include_dirs, define_macros=define_macros, undef_macros=undef_macros, library_dirs=library_dirs, libraries=libraries, runtime_library_dirs=runtime_library_dirs, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, language=language)
# end MyExtension

for d in dylib_dirs:
	d = os.path.abspath(d)
	if os.path.isdir(d):
		dylib_dir = d
		break

if inc_dir is None:
	inc_dirs = ["../include", "../"]
else:
	inc_dirs = [ inc_dir ]

if sdk is None:
	if os.path.isfile("fnv/cam/_Platinum.pyx") or os.path.isfile("fnv/cam/_BHP.pyx"):
		sdk = "cam"
	else:
		sdk = "file"

if "--shadow-dir" in sys.argv:
	# we may be installed to a read only location and setup tools doesn't properly support out of source builds
	# so copy everything to a temp directory and build from there
	idx = sys.argv.index("--shadow-dir")
	shadow_dir = sys.argv[idx+1]
	del sys.argv[idx:idx+2]
	del sys.argv[0]
	for d in lib_dirs:
		d = os.path.abspath(d)
		if os.path.isdir(d):
			lib_dir = d
			break
	for d in inc_dirs:
		d = os.path.abspath(d)
		if os.path.isdir(d):
			inc_dir = d
			break
	if os.path.exists(shadow_dir):
		print("error: shadow-dir: %s must not exist." % shadow_dir)
		exit(1)
	print("creating: %s, you should manually delete this folder when you're done." % shadow_dir)
	shutil.copytree(script_dir, shadow_dir)
	cwd = os.getcwd()
	os.chdir(shadow_dir)
	try:
		import subprocess
		args = sys.argv
		if use_cython:
			args.append("--cythonize")
		if copy_files:
			args.append("--copy-files")
		args.append("sdk="+sdk)
		args.append("incdir="+inc_dir)
		args.append("bindir="+dylib_dir)
		args.append("libdir="+lib_dir)
		subprocess.check_call([sys.executable, "setup.py"] + args)
	except:
		raise
	finally:
		os.chdir(cwd)
		exit()

extra_packages = []
if sdk == "all":
	name = "SuperSDK"
elif sdk == "file":
	name = "FileSDK"
elif sdk == "cam":
	name = "CameraSDK"

extensions = [
	MyExtension("fnv._fnv",
		["fnv/_fnv.pyx", "dllmain.cpp"],
		libraries=['fnv'],
		include_dirs=inc_dirs,
		runtime_library_dirs=['_lib'],
		library_dirs=lib_dirs),
	MyExtension("fnv.reduce",
		["fnv/reduce.pyx", "dllmain.cpp"],
		libraries=['fnv', 'fnvreduce'],
		include_dirs=inc_dirs,
		runtime_library_dirs=['_lib'],
		library_dirs=lib_dirs),
	MyExtension("fnv.file",
		["fnv/file.pyx", "dllmain.cpp"],
		libraries=['fnvfile', 'fnv', 'fnvreduce'],
		include_dirs=inc_dirs,
		runtime_library_dirs=['_lib'],
		library_dirs=lib_dirs)
	]

if sdk == "cam" or sdk == "all":
	extensions = extensions + [
		MyExtension("fnv._cam",
			["fnv/_cam.pyx", "dllmain.cpp"],
			libraries=['fnv', 'fnvreduce', 'fnvcam', 'ResourceTree'],
			include_dirs=inc_dirs,
			runtime_library_dirs=['_lib'],
			library_dirs=lib_dirs),
		MyExtension("fnv.cam._BHP",
			["fnv/cam/_BHP.pyx", "dllmain.cpp"],
			libraries=['fnv', 'fnvreduce', 'fnvcam', 'fnvcamBHP', 'bhpSDK'],
			include_dirs=inc_dirs,
			runtime_library_dirs=['../_lib'],
			library_dirs=lib_dirs),
		MyExtension("fnv.cam._Platinum",
			["fnv/cam/_Platinum.pyx", "dllmain.cpp"],
			libraries=['fnv', 'fnvreduce', 'fnvcam', 'fnvcamPlatinum'],
			include_dirs=inc_dirs,
			runtime_library_dirs=['../_lib'],
			library_dirs=lib_dirs)
		]
	extra_packages = ["fnv.cam"]
	extra_libs_in += [ "bhpSDK", "CommLayer", "dns", "fnvcam", "fnvcamBHP", "fnvcamPlatinum", "PhoenixProtocol", "ResourceTree", "GenICamRefImpl",
		"fnvcamFLIRRNDIS", "fnvcamPureGEV40", "fnvcamPureGEV41", "fnvcamPureGEV50", "fnvcamPureGEV51", "fnvcamSerial" ]

extra_libs = []

for i in extra_libs_in:
	lib = os.path.join(dylib_dir, lib_prefix + i + lib_suffix)
	liblist = [lib]
	# glob is to catch versioned so names
	if sys.platform.startswith('linux'):
		liblist += glob.glob(lib + ".*")
	for f in liblist:
		# not all plugins are available on all platforms
		if os.path.isfile(f):
			extra_libs.append(f)

if not os.path.isdir('fnv/_lib'):
	os.mkdir('fnv/_lib')

extra_files = []

for i in extra_libs:
	dest = os.path.join('fnv', '_lib', os.path.basename(i))
	if copy_files:
		shutil.copy(i, dest)
		#os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
	extra_files.append(os.path.join("_lib", os.path.basename(i)))

if "GenICamRefImpl" in extra_libs_in:
	gicdir = "GenApi" if sys.platform.startswith("darwin") else "GenICam"
	dest = os.path.join("fnv", "_lib", gicdir)
	if copy_files and not os.path.isdir(dest):
		shutil.copytree(os.path.join(dylib_dir, gicdir), dest)
	for (path, directories, filenames) in os.walk(dest):
		for filename in filenames:
			extra_files.append(os.path.relpath(os.path.join(path, filename), "fnv"))

if use_cython:
	from Cython.Build import cythonize
	# note: embedding signatures makes everything look documented to sphinx so "undocumented" members are still included
	extensions = cythonize(extensions, compiler_directives={'language_level':3,'embedsignature':True}, include_path=['.'])
else:
	extensions = no_cythonize(extensions)

# if the action is stage then we just do --copy-files and --cythonize
if "stage" in sys.argv:
	exit()

setup(
	name=name,
	version='3.1.0',
	description='Python wrapper around FLIR %s' % name,
	author="FLIR Systems Inc.",
	url='https://www.flir.com/',
	ext_modules = extensions,
	packages=["fnv"] + extra_packages,
	package_data={'fnv': extra_files},
	zip_safe=False
)
