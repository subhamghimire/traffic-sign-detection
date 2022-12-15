import torch
import torch.nn.functional as F
import PIL.Image as Image
import torchvision.transforms as T
from flask import *
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from torchvision import models



import pytorch_lightning as pl
from torch import nn as torch_nn


app = Flask(__name__)

# Classes of trafic signs
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Vehicle > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing vehicle > 3.5 tons'}


mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transforms = {'test': T.Compose([
    T.Resize(size=256),
    T.CenterCrop(size=224),
    T.ToTensor(),
    T.Normalize(mean_nums, std_nums)
]),
}

class LitModel(pl.LightningModule):

    def __init__(self, no_of_classes, learning_rate=0.0001):
        super().__init__()
        self.pretrain_model = models.resnet50(pretrained=True)
        self.pretrain_model.eval()
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        
        self.pretrain_model.fc = torch_nn.Linear(
            in_features=2048, 
            out_features=no_of_classes)

    def forward(self, input):
        output=self.pretrain_model(input)
        return output

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "valid_loss",
        }
        return [self.optimizer], [self.scheduler]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path of model checkpoint
model = LitModel.load_from_checkpoint(
    checkpoint_path='./model/epoch=2-valid_loss=1.958.ckpt'
)
model = model.to(device)
model.freeze()


def predict_proba(model, image_path):
    img = Image.open(image_path)
    # img = img.convert('RGB')
    img = transforms['test'](img).unsqueeze(0)
    pred = model(img.to(device))
    pred = F.softmax(pred, dim=1)
    return pred.detach().cpu().numpy().flatten()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the file from post request
            f = request.files['file']
            file_path = secure_filename(f.filename)
            f.save(file_path)
            filename = file_path.split(".")
            img = Image.open(file_path)
            target_name = filename[0] + ".jpg"
            rgb_image = img.convert('RGB')
            rgb_image.save(target_name)
            # Make prediction
            pred = predict_proba(model, target_name)
            # import pdb
            # pdb.set_trace()
            pred_class = pred.argmax(axis=-1)
            print(f"predicted class:", pred_class)

            pred_prob = pred.max(axis=-1)*100
            formated_pred_prob = "{:.2f}".format(pred_prob)
            print(f"Predicted probability:",formated_pred_prob)

            s = pred_class
            result = classes[s]
            print(f"Predicted sign:", result)

            os.remove(file_path)
            # os.remove(target_name)
            
            return {"data": result, "probability": formated_pred_prob, "status": 200}, 200

        except Exception as e:
            print(e)
            return {"data": "Warning!!!, Please upload valid images format only"}, 400
    return None, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')
