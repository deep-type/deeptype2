try:
    from wikidata_linker_utils_cython.entity_ranking import *
except ImportError as e:
    print("Cython extension not yet installed, cannot import cython package.")
    pass

