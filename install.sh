#!/bin/bash
#
# install the requirement packages

# environment check
## python
[ ! $(command -v python) ] && {
    echo "No python interpreter in your PATH"
    exit 1
}

## python>3
[ "$(python -V 2>&1 | awk '{print $2}' | cut -d '.' -f 1)" -ne 3 ] && {
    echo "Require python3+, instead $(python -V 2>&1)"
    exit 1
}

set -e
<<"PARSER"
("package", type=str, default='cat', nargs='*',
    choices=['all', 'cat', 'ctcdecode', 'kenlm', 'ctc-crf', 'fst-decoder'],
    help="Select modules to be installed/uninstalled. Default: cat.")
('-r', "--remove", action='store_true', default=False,
    help="Remove modules instead of installing.")
('-f', "--force", action='store_true', default=False,
    help="Force to install modules whatever they exist or not.")
PARSER
eval $(python cat/utils/parseopt.py $0 $*)

function check_py_package() {
    name=$1
    [ -z $name ] && {
        echo "error calling check_py_package()"
        return 1
    }

    cd /tmp
    python -c "import $name" >/dev/null 2>&1
    echo "$?"
}

function exc_install() {
    name=$1
    [ -z $name ] && {
        echo "error calling exc_install()"
        return 1
    }

    case $name in
    ctcdecode | cat | all)
        # install ctcdecode is annoying...
        [[ $force == "False" && $(check_py_package ctcdecode) -eq 0 ]] || {
            if [ ! -d src/ctcdecode ]; then
                git clone --recursive https://github.com/maxwellzh/ctcdecode.git src/ctcdecode
            else
                cd src/ctcdecode
                git pull --recurse-submodules
                cd - >/dev/null
            fi

            # ctcdecode doesn't support -e, but we want to install locally
            # ... so we cannot put it in requirements.txt
            python -m pip install src/ctcdecode || return 1
        }
        ;;&
    kenlm | cat | all)
        # install kenlm
        # kenlm is a denpendency of cat, so we first check the python package installation
        [[ $force == "False" && $(check_py_package kenlm) -eq 0 && -x src/bin/lmplz && -x src/bin/build_binary ]] || {
            python -m pip install -e git+https://github.com/kpu/kenlm.git#egg=kenlm

            cd src/kenlm
            mkdir -p build && cd build
            (cmake .. && make -j $(nproc)) || {
                echo "If you meet building error and have no idea why it is raised, "
                echo "... please first confirm all requirements are installed. See"
                echo "... https://github.com/kpu/kenlm/blob/master/BUILDING"
                return 1
            }

            # link executable binary files
            cd ../../
            mkdir -p bin && cd bin
            ln -snf ../kenlm/build/bin/* ./ && cd ../../
        }
        ;;&
    ctc-crf | cat | all)
        # install ctc-crf loss function
        [[ $force == "False" && $(check_py_package ctc_crf) -eq 0 ]] || {
            if [ $(command -v gcc-7) ]; then
                export ver=7
            elif [ $(command -v gcc-6) ]; then
                export ver=6
            else
                echo "gcc-6/gcc-7 command not found. You may need to install one of them."
                return 1
            fi

            cd src/ctc_crf
            CC=gcc-${ver} CXX=g++-${ver} make || return 1
            # test the installation
            echo "Test CTC-CRF installation:"
            cd test && python main.py || return 1
            cd ../../../
        }
        ;;&
    cat | all)
        # change dir to a different one to test whether cat module has been installed.
        [[ $force == "False" && $(check_py_package cat) -eq 0 ]] || {
            python -m pip install -r requirements.txt || return 1
            python -m pip install -e . || return 1
            # check installation
            $(cd egs && python -c "import cat") >/dev/null || return 1
        }
        ;;&
    fst-decoder | all)
        # install the fst decoder
        # test kaldi installation
        [[ $force == "False" && -x src/bin/latgen-faster ]] || {
            [ -z $KALDI_ROOT ] && {
                echo "\$KALDI_ROOT variable is not set, try"
                echo "  KALDI_ROOT=<path to kaldi> $0 ..."
                return 1
            }
            export KALDI_ROOT=$KALDI_ROOT
            cd src/fst-decoder && make || return 1
            [ ! -f $(command -v ./latgen-faster) ] && {
                echo "It seems the installation is success, but executable"
                echo "... binary 'latgen-faster' not found at src/fst-decoder."
                return 1
            }
            cd - >/dev/null
            [ ! -d src/bin ] && mkdir src/bin
            cd src/bin/ && ln -snf ../fst-decoder/latgen-faster ./
            cd - >/dev/null
        }
        ;;
    *) ;;
    esac

    echo "installed module:$name"
    return 0
}

function exc_rm() {
    name=$1
    [ -z $name ] && return 0

    case $name in
    cat | all)
        # FIXME: maybe we should clean building dependencies?
        python -m pip uninstall -y cat
        python setup.py clean --all
        ;;&
    ctcdecode | all)
        python -m pip uninstall -y ctcdecode
        rm -rf src/ctcdecode
        ;;&
    kenlm | all)
        python -m pip uninstall -y kenlm

        [ -d src ] && {
            cd src/
            [ -d kenlm/build/bin ] && {
                for exc in $(ls kenlm/build/bin); do
                    rm -if bin/$exc
                done
            }
            [ -d kenlm ] && rm -rf kenlm/
            cd - >/dev/null
        }
        ;;&
    ctc-crf | all)
        python -m pip uninstall -y ctc_crf

        cd src/ctc_crf
        make clean
        cd - >/dev/null
        ;;
    fst-decoder)
        rm -if src/bin/latgen-faster
        rm -rf src/fst-decoder/latgen-faster
        ;;
    *) ;;
    esac

    echo "removed module:$name"
    return 0
}

# squeeze the packages to 'all' once there is 'all'
for p in $package; do
    if [ $p == "all" ]; then
        package="all"
        break
    fi
done

if [ $remove == "False" ]; then
    # install packages
    for p in $package; do
        exc_install $p || {
            echo "failed to install $p." 1>&2
            exit 1
        }
    done
elif [ $remove == "True" ]; then
    # remove packages
    for p in $package; do
        exc_rm $p || {
            echo "failed to remove $p." 1>&2
            exit 1
        }
    done
fi

echo "$0 done"
exit 0
