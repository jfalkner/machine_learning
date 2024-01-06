# Running Llama 2 locally using CPU

Llama 2 is a large language model released by Meta in 2023, including both code and trained model snapshots. This makes
is similar to Stable Diffusion in the sense that you can run the code locally. It is pretty neat. Not to meantion that
you can fine-tune the model as well using a variety of appoaches including simply training the model more locally.

Most of basics for running locally are in [Meta'a Github repo](https://github.com/facebookresearch/llama); however,
Meta has a bunch more advanced and interesting stuff in their [Hugging Face repo named "Llama 2 Fine-tuning / Inference Recipies, Examples and Demo Apps"](https://github.com/facebookresearch/llama-recipes/). Overall, an outstanding job of an open LLM contribution from Meta!

## Running Llama 2 locally

It is straight-forward to run Llama 2 locally. You just need to download the CPP source-code from Meta's GitHub respository,
compile the code, download a pre-trained model and run it. Meta requires that you agree to their license before
downloading any of the models.

# Build llama.CPP

This assumes that you have `git` installed, are using Linux and have common build tools installed.

```
# Download the source-code
git clone https://github.com/ggerganov/llama.cpp.git

# Build the `./main` binary
cd llama.cpp
make
```

# Download a model snapshot

You'll end up downloading the models from [Meta's llama repository on Github](https://github.com/facebookresearch/llama).

There are several choices for pre-trained Llama 2 models. If unsure, start with the 7B model because it is the smallest
and will need the least amount of RAM to run (7 * 4 = ~28 GB). The larger models, namely 70B can use a huge amount of
RAM or VRAM, depending on how you run it.

For the most part, it is near impossible to train 70B unless you have a large amount of VRAM. Stick with the 7b model
unless you have a higher end computer or are using a service such as Google Colab or even GCP or AWS.

You'll need to convert Meta's model files to GGUF format using `llama.cpp/convert.py` or alternatively, cheat a little
and download the GGUF file directly from huggingface.co here.

```
# Example downloading the same files that Meta's ./download.sh script will get
apt-get -y install -qq aria2

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/resolve/main/model-00001-of-00002.safetensors -d models/Llama-2-7b-chat-hf -o model-00001-of-00002.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/resolve/main/model-00002-of-00002.safetensors -d models/Llama-2-7b-chat-hf -o model-00002-of-00002.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/model.safetensors.index.json -d models/Llama-2-7b-chat-hf -o model.safetensors.index.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/special_tokens_map.json -d models/Llama-2-7b-chat-hf -o special_tokens_map.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/resolve/main/tokenizer.model -d models/Llama-2-7b-chat-hf -o tokenizer.model
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/tokenizer_config.json -d models/Llama-2-7b-chat-hf -o tokenizer_config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/config.json -d models/Llama-2-7b-chat-hf -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/generation_config.json -d models/Llama-2-7b-chat-hf -o generation_config.json
```

Now convert the model's files to GGUF.

```
# Make sure you have the Python dependencies installed to run `convert.py`
pip install -r requirements.txt
./convert.py ../llama_models/llama/models/Llama-2-7b-chat-hf --outfile models/Llama-2-7b-chat-hf.gguf
```


# Running the model locally

This example is run on locally on a Steam Deck running Ubuntu Linux. No remarkable GPU or CPU to mention and 16GB of RAM.
You should be able to replicate this locally on most laptop and desktop PCs. You can also run this on Google Colab,
which provides access to a high end GPU.

TODO: example of using ./main locally.


## Links

Miscellaneous links related to Llama 2.

* [Meta's Llama 2 GitHub respository](https://github.com/facebookresearch/llama)
* [Meta AI's Llama 2 website](https://ai.meta.com/llama/)
* [Hugging Face tutorial for training the 70B model using PyTorch and FSDP](https://huggingface.co/blog/ram-efficient-pytorch-fsdp)
