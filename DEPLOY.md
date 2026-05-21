# DEPLOY

The app is a standalone Streamlit container; everything else is your
reverse proxy's job. Designed to live behind any system nginx + certbot
on a shared host — the container binds `127.0.0.1:8501` only, and your
nginx vhost routes by `Host` header.

## Minimal recipe

```bash
docker compose -f deploy/docker-compose.yml up -d --build
curl -fsS http://127.0.0.1:8501/_stcore/health      # "ok"
```

Then wire your nginx vhost (sample at `deploy/nginx-tfbss.conf`) to:

```nginx
location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Host              $host;
    proxy_set_header X-Forwarded-Proto $scheme;
}

# Streamlit's WebSocket — without this the page loads but the spinner
# never resolves.
location /_stcore/stream {
    proxy_pass http://127.0.0.1:8501/_stcore/stream;
    proxy_http_version 1.1;
    proxy_set_header Upgrade            $http_upgrade;
    proxy_set_header Connection         "upgrade";
    proxy_set_header Host               $host;
    proxy_read_timeout  86400s;
}
```

Issue a Let's Encrypt cert with `certbot --nginx -d <hostname>`.

## Data layer

The container expects a populated `./data/` directory mounted read-only
at `/srv/app/data`. The viewer **does not regenerate** this directory at
runtime; build it once with:

```bash
make db                                    # writes data/canonical_promoter.duckdb
python data/build_depmap_pair_matrices.py  # slim DepMap matrices
python data/build_depmap_tf_target_correlations.py
python data/build_tf_pair_table.py         # TF×TF co-occurrence
```

These scripts read from `HPA_ANALYSIS_DIR` (upstream chip-atlas analysis
outputs) and `HPA_DEPMAP_RAW` (raw DepMap CSVs). Defaults are
`data/raw/analyses/` and `data/raw/depmap/` — override via env vars if
the source data lives elsewhere on your machine.

## Rollback

```bash
docker compose -f deploy/docker-compose.yml down
# disable the vhost in nginx if needed:
#   rm /etc/nginx/sites-enabled/<hostname> && nginx -t && systemctl reload nginx
```

## Healthcheck

The container exposes `/healthz` (Streamlit's internal `_stcore/health`
alias) used by the docker `HEALTHCHECK` directive. A site that loads but
hangs on the spinner is almost always a missing `/_stcore/stream`
WebSocket route in your nginx config.
