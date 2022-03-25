try:
	from wikidata_linker_utils_cython.successor_mask import *
except ImportError as e:
	print("Cython extension not yet installed, cannot import cython package.")
	pass

