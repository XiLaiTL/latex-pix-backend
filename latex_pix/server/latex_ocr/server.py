import torch

from .protos import latex_ocr_pb2_grpc
from .protos import latex_ocr_pb2
import grpc
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
import threading
from concurrent import futures

from ..image_utils import convert_image


class LatexOCR(latex_ocr_pb2_grpc.LatexOCRServicer):
    def __init__(self, model, cache_dir, device):
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None

        self.load_thread = threading.Thread(target=self.load_models, args=(model,), daemon=True)
        print("Server started")
        self.load_thread.start()

    def GenerateLatex(self, request, context):
        image = convert_image(request.image_path)
        result = self.inference(image)
        return latex_ocr_pb2.LatexReply(latex=result)
    
    def IsReady(self, request, context):
        is_ready = not self.load_thread.is_alive()
        return latex_ocr_pb2.ServerIsReadyReply(is_ready=is_ready)
    
    def GetConfig(self, request, context):
        return latex_ocr_pb2.ServerConfig(device = self.device, cache_dir=self.cache_dir)

    def inference(self, image):


        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        outputs = self.model.generate(pixel_values)[0]
        sequence = self.tokenizer.decode(outputs)
        sequence = sequence.replace('\\[','\\begin{align*}').replace('\\]','\\end{align*}')
        return sequence

    def load_models(self, model):
        print("Loading model...", end="", flush=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(f"{self.cache_dir}/{model}" , local_files_only=True)
        print(" done", flush=True)

        print("Loading processor...", end="", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/{model}" , max_len=296, local_files_only=True)
        self.feature_extractor = AutoImageProcessor.from_pretrained(f"{self.cache_dir}/{model}" , local_files_only=True)
        print(" done", flush=True)


def serve(model_name: str, port: str, cache_dir: str, cpu: bool):
    if not cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Starting server on port {port}, using {device}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    latex_ocr_pb2_grpc.add_LatexOCRServicer_to_server(LatexOCR(model_name, cache_dir, device), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    server.wait_for_termination()