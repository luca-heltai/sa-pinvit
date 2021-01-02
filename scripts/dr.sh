IMG=dealii/dealii:master-focal
echo $IMG
docker run  --user $(id -u):$(id -g) \
    --rm -t \
    -v `pwd`:/builds/app $IMG /bin/sh -c "cd /builds/app; $@"
