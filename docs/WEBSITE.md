# Project homepage

Canonical project homepage: https://dengqirui.me/ERETrace/

The project is hosted by **GitHub Pages**, from the `main` branch's `/docs`
directory in `DQR-Happy/ERETrace`. The domain is registered through Namecheap;
Namecheap supplies the domain/DNS service, while GitHub serves the site.

The complete static source lives in `docs/index.html` and `docs/assets/` in this
repository. It needs no package installation, build service, external font or
analytics script. Serve it locally with:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory docs
```

`assets/results.json` is generated from the retained primary scalar data. Keep
that provenance when updating figures or labels. The model selector, theme
switch and command-copy control run entirely in the browser.

For deployment, push updated static assets to `main`; GitHub Pages publishes
`/docs`. `.nojekyll` keeps the HTML/CSS/JS assets unchanged. No external app-hosting
account is needed. GitHub has issued a certificate for the account-level custom domain, and the
HTTPS homepage has been verified with standard TLS validation.
Hosting changes are separate commits and do not alter the v1.0.0 code tag or
the private submission snapshot.
