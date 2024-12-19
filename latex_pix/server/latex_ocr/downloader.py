import urllib

from huggingface_hub import hf_hub_url
from modelscope import snapshot_download
from modelscope.hub.file_download import model_file_download


def check_access()->bool:
    """
    Try to access huggingface
    :return:  True if not accessible
    """
    url = "https://huggingface.co/"
    access_hf = False
    access_ms = False
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print(f"Success to access huggingface")
                access_hf = True
    except urllib.error.URLError as e:
        print(f"Failed to access huggingface, try download from modelscope")

    url = "https://modelscope.cn/"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print(f"Success to access modelscope")
                access_ms = True
    except urllib.error.URLError as e:
        print(f"Failed to access modelscope, fail")

    return True if access_ms else (True if not access_hf else False)

def download_model(model_name, model_file,cache_dir,access):
    if access:
        download_model_from_modelscope(model_name, model_file, cache_dir)
    else:
        download_model_from_hf(model_name, model_file, cache_dir)


def download_model_from_hf(model_name, model_file,cache_dir):
    # Get file size
    download_path = hf_hub_url(model_name, model_file)
    req = urllib.request.Request(download_path, method="HEAD")
    f = urllib.request.urlopen(req)
    size = f.headers['Content-Length']
    file_size = '{:.2f} MB'.format(int(size) / float(1 << 20))

    # Get cache_dir name
    cache_dir = cache_dir if cache_dir else "~/.cache/huggingface"
    ans = input(f"Will download model ({file_size}) to {cache_dir}. Ok? (Y/n) ")
    if ans.lower() == "n":
        quit(0)

def download_model_from_modelscope(model_name,model_file,cache_dir):
    for file_path in [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "README.md"
    ]:
        model_file_download(model_id=model_name,file_path=file_path,cache_dir=cache_dir)
