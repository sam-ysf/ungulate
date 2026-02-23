## Ungulate: Local LLM-based OCR and Search Engine

A fast, local document search engine built with C++ that combines Large Language Model (LLM) inference and a vector database for robust search capabilities.

## Project Overview

This project uses LLM inference to build a vector embedding database that can be queried locally. It uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for model inference, and it supports vector as well as full-text search.

- **Document OCR**: Extracts text from PDF files using a multimodal LLM and stores vectorized result into an embeddings database

- **File Search**: Supports vector similarity search (using [K-nearest neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)) and full-text keyword search (using [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25))

- **Web Interface**: Uses a web-based API for file uploads and search queries

## Building from Source

### Installing Dependencies

First, install the necessary development libraries.

```Bash
# For Debian systems
apt install \
  libssl-dev \
  libgs-dev \
  libopencv-dev \
  libpoppler-cpp-dev \
  libsqlite3-dev \
  nlohmann-json3-dev
```

Compile and install llama.cpp libs from source (build instructions are located [here](https://huggingface.co/rohan23998/llama-cpp-model/blob/main/docs/build.md)).

### Building and Running

```Bash
git clone --recursive https://github.com/sam-ysf/ungulate

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cmake --install .

#
# Modify the configuration (see next section)...
#

ungulate # Run
```

### Model configuration

Before running, copy the configuration file in `docs/model.json`(docs/model.json) to the configuration directory (default is `~/.config/ungulate/server`) and point the model paths to the desired GGUFs supported by llama.cpp. A multimodal model is required for OCR ([Qwen3.5](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) for instance).

```JSON
{
  "ocr": [{
    "gguf": "/path/to/Qwen3.5/Qwen3.5-9B-Q4_K_M.gguf",
    "mmproj": "/path/to/mmproj-F16.gguf",
    "description": "Qwen3.5",
    "prompt": "Convert this image to Markdown.",
    "params": {
      "ctx-size": "40960"
    }
  }],
  "embeddings": [{
    "gguf": "/path/to/embedding/model",
    "chunks": [
      "(.+)"
    ],
    "params": {
      "ctx-size": "40960"
    }
  }]
}
```

The `"prompt"` field must be included and can be used to specify the text output format (e.g. Markdown or LaTeX). The `"params"` field can be used to pass additional arguments to llama.cpp (see [docs/supported-llama-cpp-params.yaml](docs/supported-llama-cpp-params.yaml) for the complete list of supported parameters).

The `"embeddings"` section is optional but required to enable the generation and storage of vector embeddings from parsed text. This field takes an array, allowing indexing against multiple embedding models in sequence.

The `"chunks"` field must be passed an array of regex strings that extract all semantically-meaningful chunks to be converted to vectors (the above configuration delimits along line boundaries).

### Server configuration

By default, the query server (HTTP) listens on port 6868 while the client/dashboard listens on port 6767. This can be changed by editing the file `net.json` located in the configuration directory.

```JSON
{
  "dashboard-port": "6767",
  "query-port": "6868"
}
```

## API

The file list endpoint `file/list` (POST) returns a JSON list of parsed files alongside file metadata.

The file upload endpoint `/file/new` must be passed a multipart/form-data file stream.

The search endpoint `/search` must be passed a JSON object that contains a search query and search type. The search type must be either `"stochastic"` (to use vector search) or `"keyword"` (to use full-text search). If this field is omitted, keyword search is used by default.

```JSON
{
  "search-type": "stochastic",
  "query": {
    "value": "this is my search query"
  }
}
```

Calls to generate vector embeddings (using `/file/embeddings`) or to download a file (`/file/download`) require a JSON object containing the file UUID, which can be found in the file metadata (returned in the call to `/file/list`).

```JSON
{
  "file": 25
}
```

See [docs/api.yaml](docs/api.yaml) for the full list of API endpoints.

## Acknowledgments

See [docs/dependencies.yaml](docs/dependencies.yaml) for list of additional dependencies.
