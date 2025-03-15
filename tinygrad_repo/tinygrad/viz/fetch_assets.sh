#!/bin/bash
fetch() {
  echo "fetch $1"
  mkdir -p assets/$1
  rmdir assets/$1
  curl -o assets/$1 https://$1
}
fetch "d3js.org/d3.v5.min.js"
fetch "dagrejs.github.io/project/dagre-d3/latest/dagre-d3.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/default.min.css"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/highlight.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/languages/python.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/languages/cpp.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/dompurify/1.0.3/purify.min.js"
fetch "cdnjs.cloudflare.com/ajax/libs/dompurify/1.0.3/purify.min.js.map"
fetch "unpkg.com/@highlightjs/cdn-assets@11.10.0/styles/tokyo-night-dark.min.css"
