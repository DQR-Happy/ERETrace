# Project homepage

Canonical project homepage: https://eretrace.golden-mole-0361.chatgpt.site

The complete static source lives in `docs/index.html` and `docs/assets/` in this
repository. It needs no package installation, build service, external font, or
analytics script. Serve it locally with:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory docs
```

`assets/results.json` is generated from the retained primary scalar data. Keep
that provenance when updating figures or labels. The model selector, theme
switch and command-copy control run entirely in the browser.

For hosting, copy `index.html` and `assets/` into any static host's public root.
The canonical deployment uses Sites with the same static assets and public
access. Hosting changes do not alter the code release or private submission
snapshot. Account-level custom domains should be managed independently of
this artifact.
