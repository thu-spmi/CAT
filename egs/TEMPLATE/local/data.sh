set -e
set -u

mkdir -p data/local
cwd=$(pwd)

cd data/local
if [ ! -f .completed ]; then
    # download data
    [ ! -f yesno.tar.gz ] &&
        wget https://www.openslr.org/resources/1/waves_yesno.tar.gz -O yesno.tar.gz

    # check downloaded file
    [ "$(md5sum yesno.tar.gz | awk '{print $1}')" != "962ff6e904d2df1126132ecec6978786" ] && (
        echo "MD5SUM check failed for yesno.tar.gz, please rm it then re-run the script."
        exit 1
    )

    # untar
    tar -zxf yesno.tar.gz
    touch .completed
else
    echo "Found previous processed data. Skip download"
fi
cd $cwd

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

python local/compute_fbank.py data/local/waves_yesno/
echo "FBank spectrum generate done."

python utils/data/resolvedata.py

echo "$0 done"
exit 0
