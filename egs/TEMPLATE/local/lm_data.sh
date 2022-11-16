set -e
set -u

mkdir -p data/local-lm
cwd=$(pwd)

cd data/local-lm
if [ ! -f .completed ]; then
    # download data
    [ ! -f ptbdataset.zip ] &&
        wget https://data.deepai.org/ptbdataset.zip

    # check downloaded file
    [ "$(md5sum ptbdataset.zip | awk '{print $1}')" != "bd5e54b9d7715eec3db089219809b986" ] && (
        echo "MD5SUM check failed for ptbdataset.zip, please rm it then re-run the script."
        exit 1
    )

    # untar
    unzip ptbdataset.zip
    touch .completed
else
    echo "Found previous processed data. Skip download"
fi
cd $cwd

echo "$0 done"
exit 0
