IMG=dealii/dealii:master-focal
echo $IMG
docker run  --user $(id -u):$(id -g) \
    --rm -i -t \
    -v `pwd`:/builds/app $IMG /bin/sh -c "cd /builds/app; $@"
