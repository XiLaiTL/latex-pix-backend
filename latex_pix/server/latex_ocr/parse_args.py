import argparse
import logging

from huggingface_hub import try_to_load_from_cache
from modelscope import check_local_model_is_latest

from latex_pix.server.latex_ocr.downloader import download_model, check_access
from .__about__ import __version__

MODEL_NAME = {
    "huggingface":"MixTeX/ZhEn-Latex-OCR",
    "modelscope":"MixTeX/MixTex-ZhEn-Latex-OCR"
}
ACCESS_MODEL = {
    False: "huggingface",
    True: "modelscope"
}

# Filename to check cache for
MODEL_FILE = "model.safetensors"

def run():
    logging.basicConfig()
    args = parse_option()
    args.func(args)

# latex_ocr_server info ...
def handle_info(args):
    if args.gpu_available:
        import torch

        is_available = torch.cuda.is_available()
        print(f"{is_available}")
        exit(not is_available)

# latex_ocr_server start ...
def handle_start(args):
    access = check_access()
    model_name = MODEL_NAME[ACCESS_MODEL[access]]
    try:
        filepath = try_to_load_from_cache(model_name, MODEL_FILE, cache_dir=args.cache_dir)
        if not isinstance(filepath, str):
            print("downloading")
            download_model(model_name,MODEL_FILE,args.cache_dir,access)
    except:
        pass
    finally:
        from .server import serve
        serve(model_name, args.port, args.cache_dir, args.cpu)
        
def parse_option():
    parser = argparse.ArgumentParser(prog="latex_ocr_server", description="A server that translates paths to images of equations to latex using protocol buffers.")
    parser.add_argument("--version", action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(help='sub-command help', required=True)
    start = subparsers.add_parser("start", help='start the server')
    start.add_argument("--port", default="50051")
    start.add_argument("-d", "--download", default=False, help="download model if needed without asking for confirmation", action='store_true')
    start.add_argument("--cache_dir", default=None, help="path to model cache. Defaults to ~/.cache/huggingface")
    start.add_argument("--cpu", default=False, action="store_true", help="use cpu, otherwise uses gpu if available")
    start.set_defaults(func=handle_start)

    info = subparsers.add_parser("info", help="get server info")
    info.add_argument("--gpu-available", required=True, action="store_true", help="check if gpu support is enabled")    
    info.set_defaults(func=handle_info)

    return parser.parse_args()

if __name__ == "__main__":
    run()