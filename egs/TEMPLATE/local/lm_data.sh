set -e -u

dir="data/local-lm"
n_utts=50000
url="https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"

[ $n_utts -le 500 ] && {
    echo "#utterances must > 500 for spliting train & dev" >&2
    exit 1
}

mkdir -p $dir
cd $dir
if [ ! -f .completed ]; then
    # download and process data
    echo "Start downloading corpus, please wait..."
    wget $url -q -O - | gunzip -c | head -n $n_utts |
        tr '[:upper:]' '[:lower:]' >libri-part.txt
    echo "Corpus downloaded. ($n_utts utterances from librispeech corpus)"

    # take the last 500 utterances as dev
    head -n $(($n_utts - 500)) libri-part.txt >libri-part.train
    tail -n 500 libri-part.txt >libri-part.dev
    touch .completed
else
    echo "Found previous processed data."
fi
cd - >/dev/null

echo "$0 done"
exit 0
