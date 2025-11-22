#!/bin/bash
fetch() {
  echo "fetch $1"
  mkdir -p assets/$1
  rmdir assets/$1
  curl -o assets/$1 https://$1
}
fetch "d3js.org/d3.v7.min.js"
fetch "dagrejs.github.io/project/dagre/latest/dagre.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/default.min.css"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/highlight.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/languages/python.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/languages/x86asm.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/languages/cpp.min.js"
fetch "unpkg.com/@highlightjs/cdn-assets@11.10.0/styles/tokyo-night-dark.min.css"
