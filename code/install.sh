SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )

pushd "$SCRIPT_DIR/wikidata_linker_utils"
pip3 install -e . --user
popd

pushd "$SCRIPT_DIR/wikidata_linker_utils_cython"
pip3 install . --user
popd
