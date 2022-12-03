# Data Preparation

Prepare datasets in data directory. You can reference each datasets format on [each projects README](../../configs/projects).

```
/path/to/data
└── rsna2022
```

# Environment setup

Clone repo

```
$ git clone https://github.com/okotaku/clshub
```

Set env variables

```
$ export DATA_DIR=/path/to/data
```

Start a docker container

```
$ docker compose up -d clshub
```
