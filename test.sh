#scp -r -P 12331 --exclude workspace  * neo@113.108.13.218:/home/neo/test
rsync -auzv --delete -e "ssh -p 12331" --exclude workspace  * neo@113.108.13.218:/home/neo/test