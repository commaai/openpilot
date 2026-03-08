#!/bin/bash
fetch() {
  echo "fetch $1"
  mkdir -p assets/$1
  rmdir assets/$1
  curl -L -o assets/$1 https://$1
}
fetch "cdn.jsdelivr.net/npm/@alpine-collective/toolkit@1.0.2/dist/cdn.min.js"
fetch "cdn.jsdelivr.net/npm/@alpinejs/intersect@3.x.x/dist/cdn.min.js"
fetch "cdn.jsdelivr.net/npm/@alpinejs/focus@3.x.x/dist/cdn.min.js"
fetch "unpkg.com/@marcreichel/alpine-autosize@1.3.x/dist/alpine-autosize.min.js"
fetch "unpkg.com/@marcreichel/alpine-autosize@1.3.x/dist/alpine-autosize.min.js.map"
fetch "unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"
fetch "unpkg.com/dompurify@3.1.5/dist/purify.min.js"
fetch "unpkg.com/dompurify@3.1.5/dist/purify.min.js.map"
fetch "unpkg.com/marked@13.0.0/marked.min.js"
fetch "unpkg.com/marked-highlight@2.1.2/lib/index.umd.js"
fetch "unpkg.com/@highlightjs/cdn-assets@11.9.0/highlight.min.js"
fetch "cdn.jsdelivr.net/npm/purecss@3.0.0/build/base-min.css"
fetch "cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"
fetch "unpkg.com/@highlightjs/cdn-assets@11.9.0/styles/vs2015.min.css"
fetch "cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-solid-900.ttf"
fetch "cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-solid-900.woff2"
