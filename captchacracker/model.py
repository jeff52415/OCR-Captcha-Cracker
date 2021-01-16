import logging
import os
import random
import string

import captcha
import editdistance
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from captcha.image import ImageCaptcha
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from captchacracker.utils.core import CRNN


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CaptchaCracker:
    def __init__(
        self,
        weight_path: str = "",
        device: str = "cuda",
        pretrained: bool = True,
        backbone: str = "light",
    ):
        """ Captcha Recognition Module
        Args:
            weight_path: pretrained weight path if existed
            device: instance type, either gpu (cuda) or cpu
            pretrained: take pretrained weight torchvision.models or not 
            backbone: Image feature extractor structure, either light or normal
        """
        self.characters = string.digits + string.ascii_uppercase
        self.width, self.height, self.n_len, self.n_class = (
            128,
            64,
            4,
            len(self.characters),
        )
        self.generator = ImageCaptcha(width=self.width, height=self.height)

        n = 1  ## keep zero position for "space" class in CTC loss
        self.character_index = {}  ## character to index dictionary
        self.index_character = {}  ## index to character dictionary
        for character in self.characters:
            self.character_index[character] = n
            self.index_character[n] = character
            n += 1

        self.model = CRNN(
            models.resnet34(pretrained=pretrained), backbone, len(self.characters) + 1
        )

        if torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info("device type : {}".format(self.device))

        self.model.to(self.device)

        self.ready_for_inference = False
        if os.path.exists(weight_path):
            try:
                self.model.load_weights(weight_path)
                self.ready_for_inference = True
                logger.info("ready for inference")
            except:
                logger.info("no available weight")
                pass

    def loader(self, batch_size: int = 8, width: int = 128, height: int = 64):
        """ generator : generate batch of captcha
        Args:
            batch_size: batch_size
            width: width of generated images
            height: height of generated images
        Returns:
            Random generated captcha images with corresponding label
        """
        channels = 3  ## RGB
        while True:
            images = np.zeros((batch_size, height, width, channels), dtype=np.uint8)
            labels = []
            for _ in range(batch_size):
                random_str = "".join(
                    [random.choice(self.characters) for j in range(self.n_len)]
                )
                img = self.generator.generate_image(random_str)
                for item in random_str:
                    labels.append(self.character_index.get(item))
                images[_] = img
            yield images, labels

    def decode(self, prediction: list) -> str:
        "Decode CTC output into target textlines"
        predict_string = ""
        previous_step = 0
        for character in prediction:
            if character > 0:
                if previous_step != character:
                    predict_string += self.index_character.get(character)
            previous_step = character
        return predict_string

    def get_editdistance_loss(self, y_true: str, y_pred: str) -> int:
        "Edit distance between two string"
        return editdistance.distance(y_true, y_pred)

    def get_similarity_score(self, y_true: str, y_pred: str) -> int:
        "Caculate score between 2 textlines"
        max_len = max(len(y_pred), len(y_true))
        return (max_len - self.get_editdistance_loss(y_true, y_pred)) / max_len

    def save(
        self,
        weights_path,
        iteration: int = 0,
        loss: float = 0,
        score: float = 0,
        *args,
        **kwargs
    ):
        """ Save current checkpoint to disk
        Args:
            weights_path: weight destination
        Returns:
            .pth or .pt file
        """
        father_folder = os.path.dirname(weights_path)
        if not os.path.exists(father_folder) and father_folder != "":
            os.makedirs(father_folder)

        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "loss": round(loss, 4),
                "score": round(score, 4),
            },
            weights_path,
        )

        logger.info("save successfully....")

    @torch.no_grad()
    def process(self, img) -> str:
        """ inference
        Args:
            img: input image
        Returns:
            prediction string
        """
        if not self.ready_for_inference:
            raise Exception("Please load proper weight for inference first !!!")
        self.model.eval()

        if isinstance(img, str):
            img = Image.open(img)
            img = np.array(img, dtype=np.uint8)
        elif isinstance(img, PIL.Image.Image):
            img = np.array(img, dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            img = img.astype(np.uint8)

        img = torch.tensor(img).type(torch.FloatTensor).to(self.device)
        img = torch.tensor(img).type(torch.LongTensor).to(self.device)
        img = img / 127.5 - 1
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        output = self.model(img)
        prediction = output.detach().clone()
        prediction = F.softmax(prediction, 2)
        prediction = prediction.argmax(2).reshape(-1)
        prediction = prediction.cpu().numpy()
        output = self.decode(prediction)

        return output

    def fit(
        self,
        batch_size: int = 64,
        output_step_length: int = 16,
        max_iterations: int = 10000,
        early_stop: int = 20,
        writer_folder: str = "crnn_ctc_writer",
        save_path: str = "weight/crnn_ctc_model.pth",
        iteration_per_epoch: int = 500,
    ):
        """ train model
        Args:
            batch_size: batch_size
            output_step_length: output size of CNN final layer on width dimention
            max_iterations: iteration time
            writer_folder: folder for storing SummaryWritter output
            save_path: folder for storing checkpoints
            iteration_per_epoch: iteration_per_epoch
        Returns:
            checkpoints
        """
        self.model.train()
        ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        input_lengths = torch.tensor(
            [output_step_length for _ in range(batch_size)], dtype=torch.long
        )  ## output timestep
        target_lengths = torch.tensor(
            [4 for _ in range(batch_size)], dtype=torch.long
        )  ## it can be dynamic, but in our case is fixed

        optimizer = optim.Adam(self.model.parameters(), lr=0.01 / batch_size)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.98
        )

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.train().parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("total parameters： {}".format(params))

        writer = SummaryWriter(log_dir=writer_folder)
        data_loader = self.loader(batch_size=batch_size)

        min_loss = 100
        max_scroe = 0
        buffer = 0

        for iteration in tqdm(range(max_iterations)):

            if (iteration != 0) and (iteration % iteration_per_epoch == 0):
                lr_scheduler.step()
            if iteration % iteration_per_epoch == 0:
                average_score_list = []
                average_loss_list = []

            images, labels = next(data_loader)
            images = torch.tensor(images).type(torch.FloatTensor).to(self.device)
            labels = torch.tensor(labels).type(torch.LongTensor).to(self.device)
            images = images / 127.5 - 1
            images = images.permute(0, 3, 1, 2)

            optimizer.zero_grad()

            output = self.model(images)

            prediction = output.detach().clone()
            prediction = F.softmax(prediction, 2)
            prediction = prediction.argmax(2).permute(1, 0)
            prediction = prediction.cpu().numpy()

            output = torch.nn.functional.log_softmax(output, dim=2)
            loss = ctc_loss(output, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            loss = loss.detach().item()
            average_loss_list.append(loss)

            decode_prediction = []
            for item in prediction:
                decode_string = self.decode(item)
                decode_prediction.append(decode_string)

            label_convert = labels.clone().detach().cpu().numpy()
            label_convert = label_convert.reshape(-1, 4)
            decode_label = []
            for item in label_convert:
                decode_string = ""
                for character in item:
                    decode_string += self.index_character.get(character)
                decode_label.append(decode_string)

            score_list = []

            for predict, ground_truth in zip(decode_prediction, decode_label):
                score = self.get_similarity_score(ground_truth, predict)
                score_list.append(score)

            average_score = np.mean(score_list)
            average_score_list.append(average_score)

            if iteration % iteration_per_epoch == 0:
                mean_loss = np.mean(average_loss_list)
                mean_score = np.mean(average_score_list)

                writer.add_scalar("train/loss", mean_loss, iteration)
                writer.add_scalar("train/score", mean_score, iteration)

                logger.info("current status : {} iterations".format(iteration))
                logger.info("current loss： {}".format(mean_loss))
                logger.info("score： {}".format(mean_score))
                logger.info("\n")

                if (mean_loss < min_loss) or (mean_score > max_scroe):
                    buffer = 0
                    self.save(save_path, iteration, mean_loss, mean_score)
                else:
                    buffer += 1

            if buffer >= early_stop:
                logger.info("loss no longer decrease and score no longer increase")
                logger.info("finish training")
                break
