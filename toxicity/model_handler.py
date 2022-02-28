"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
from transformers import BertTokenizer, BertModel
from ts.utils.util import list_classes_from_module
from torch.nn.utils.rnn import pad_sequence
# from model import BertClassifier
from abc import ABC
import json, logging, os, ast, torch, importlib, numpy as np
from captum.attr import LayerIntegratedGradients

logger = logging.getLogger(__name__)


# command to create toxicity model:-----
# torch-model-archiver --model-name ToxicClassification --version 1.0 --model-file model.py --serialized-file toxicity_model/model.pt 
# --handler ./model_handler.py --extra-files "./setup_config.json,./index_to_name.json"

# serve only toxicity model:------
# torchserve --start --ncs --model-store model_store --models toxicity=ToxicClassification.mar --ts-config ./config.properties

# serve all models:------
# torchserve --start --ncs --model-store ./models --models toxicity=ToxicClassification.mar sentiment=BERTSeqClassification.mar --ts-config ./config.properties

class ModelHandler(BaseHandler, ABC):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        
        #  load the model, refer 'custom handler class' above for details
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.manifest = context.manifest
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")
        
        self.bert_model_name = self.setup_config['model_name']

        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            # self.model = BertClassifier(BertModel.from_pretrained(bert_model_name), self.setup_config['num_labels']).to(self.device)
            model_file = self.manifest["model"].get("modelFile", "")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            # self.model = torch.load(model_pt_path)
            self.model.to(self.device)
        
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        with open(mapping_file_path) as f:
            self.mapping = json.load(f)

        self._context = context
        self.initialized = True

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class(BertModel.from_pretrained(self.bert_model_name), int(self.setup_config['num_labels']))
        if model_pt_path:
            state_dict = torch.load(model_pt_path, map_location=self.device)
            model.load_state_dict(state_dict)
        return model

    def preprocess(self, requests):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            if self.setup_config["captum_explanation"]:
                input_text_target = ast.literal_eval(input_text)
                input_text = input_text_target["text"]
            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)
            inputs = self.tokenizer.encode_plus(input_text, max_length=int(max_length), pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
        return (input_ids_batch, attention_mask_batch)


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        # model_output = self.model.forward(model_input)
        input_ids_batch, attention_mask_batch = model_input
        inferences = []
        _, predictions = self.model(input_ids_batch, attention_mask_batch)
        print(predictions)
        num_rows, num_cols = predictions.shape
        print(self.mapping)
        for i in range(num_rows):
            out = predictions[i]
            # y_hat = out.argmax(1).item()
            # predicted_idx = str(y_hat)
            # inferences.append(self.mapping[predicted_idx])
            out = np.round(out.cpu().detach().numpy()*100, 1)
            data = {}
            print(out)
            for i in range(len(out)):
                data[self.mapping[str(i)]] = str(out[i])
            inferences.append(data)

        return inferences

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    # def handle(self, data, context):
    #     """
    #     Invoke by TorchServe for prediction request.
    #     Do pre-processing of data, prediction using model and postprocessing of prediciton output
    #     :param data: Input data for prediction
    #     :param context: Initial context contains model server system properties.
    #     :return: prediction output
    #     """
    #     model_input = self.preprocess(data)
    #     model_output = self.inference(model_input)
    #     return self.postprocess(model_output)

def captum_sequence_forward(inputs, attention_mask=None, position=0, model=None):
    """This function is used to get the predictions from the model and this function 
    can be used independent of the type of the BERT Task. 
    Args:
        inputs (list): Input for Predictions
        attention_mask (list, optional): The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them, it defaults to None.
        position (int, optional): Position depends on the BERT Task. 
        model ([type], optional): Name of the model, it defaults to None.
    Returns:
        list: Prediction Outcome
    """
    model.eval()
    model.zero_grad()
    pred = model(inputs, attention_mask=attention_mask)
    pred = pred[position]
    return pred