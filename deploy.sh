#rsync -auzv * --exclude workspace root@saas.netkiller.cn:/srv/workstation
rsync -auzv * --exclude workspace root@dev.ideasprite.com:/srv/workstation
ssh root@dev.ideasprite.com "systemctl restart workstation.service"