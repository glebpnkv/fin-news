# fin-news
Financial news analysis: using 🗞 to explain 📈

## Setup
### Python `venv` setup
The most seamless way to get everything up and running is via `uv`. 

Install `uv` (from [Installing UV](https://docs.astral.sh/uv/getting-started/installation/)):
```aiignore
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once `uv` is installed, run `uv sync --all-packages` to install all dependencies:
```shell
uv sync --all-packages
```

## Data
### BigData
To run `ravenpack_data` pipeline, you will need [RavenPack BigData API key](https://bigdata.com/).

Once you have the key, make a copy `apps/ravenpack_data/.env.example` (rename it to `.env`) and paste the key into the  
`.env` file. 

### Original model
Data comes from [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) repository