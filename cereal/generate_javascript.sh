#!/bin/bash

rm -r gen/ts
rm -r gen/js

mkdir gen/ts
mkdir gen/js

echo "Installing needed npm modules"
npm i capnpc-ts capnp-ts

capnpc -o node_modules/.bin/capnpc-ts:gen/ts log.capnp car.capnp
capnpc -o node_modules/.bin/capnpc-ts:gen/ts car.capnp

cat log.capnp | egrep '\([a-zA-Z]*\.[^\s]+\.[^s]+\)' | sed 's/^.*([a-zA-Z]*\.\([a-zA-Z.]*\)).*/\1/' | while read line
do
  TOKEN=`echo $line | sed 's/\./_/g'`
  ROOT=`echo $line | sed 's/\..*$//g'`
  cat gen/ts/log.capnp.ts | grep '^import.*'${TOKEN}
  if [[ "$?" == "1" ]]
  then
    sed -i 's/^\(import {.*\)'${ROOT}'\(,*\) \(.*\)$/\1'${ROOT}', '${TOKEN}'\2 \3/' ./gen/ts/log.capnp.ts
  fi
done

tsc ./gen/ts/* --lib es2015 --outDir ./gen/js
