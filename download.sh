mkdir -p circor_data
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
mv physionet.org/files/circor-heart-sound/1.0.3/* circor_data
rm -rf physionet.org
echo "Download complete! Files are in circor_data/"