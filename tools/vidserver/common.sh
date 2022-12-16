#!/bin/bash

regex_date='[0-9]{4}-[0-9]{2}-[0-9]{2}'
regex_time='[0-9]{2}-[0-9]{2}-[0-9]{2}'
regex_route='^('"${regex_date}"')--('"${regex_time}"')--(.*)$'
function join() {
  join_char="$1"
  in="$(cat)"
  echo $in | sed 's/ /'"${join_char}"'/g'
}
function get_segments() {
  # filter stdin (usually 'ls' output) for stuff matching op route things
  grep -E "${regex_route}"
}
function get_routes() {
  # filter stdin (usually 'ls' output) and remove seg num
  sed -nE 's/'"${regex_route}"'/\1--\2/gp' | uniq
}
