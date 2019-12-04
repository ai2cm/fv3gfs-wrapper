# Add the Cloud SDK distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg |  apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update the package list and install the Cloud SDK
apt-get update &&  apt-get install -y python-dev google-cloud-sdk

# Install fast crc mod for multithreaded downloads
# the snakemake workflow will fail on gsutil downloads without this
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python2.7 get-pip.py
PIP="python2.7 -m pip"
echo "PIP Version"
$PIP -V
$PIP uninstall crcmod
$PIP install --no-cache-dir -U crcmod
