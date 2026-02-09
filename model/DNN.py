import logging
import joblib
from sympy import false
from torch.utils.data import Subset
from torch.utils.data import TensorDataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from opacus import PrivacyEngine
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model.ResNet import resnet18, resnet50, resnet18_dp
from model.VGG import vgg11_bn, vgg19_bn, vgg13_bn


class DNN(nn.Module):
    def __init__(self, args=None):
        super(DNN, self).__init__()

        self.logger = logging.getLogger("DNN")
        self.args = args
        self.device = args['device']
        if args['dataset_name']=='tinyimagenet':
            self.imagenet = True
        else:
            self.imagenet = False
        if args['dataset_name']=='cifar100':
            self.num_classes = 100
        elif args['dataset_name']=='celebA':
            self.num_classes = 40
        elif args['dataset_name']=='tinyimagenet':
            self.num_classes = 200
        elif args['dataset_name']=='sst5':
            self.num_classes = 5
        elif args['dataset_name']=='news20':
            self.num_classes = 20
        elif args['dataset_name']=='snli':
            self.num_classes = 3
        elif args['dataset_name']=='mnli':
            self.num_classes = 3
        elif args['dataset_name']=='mrpc':
            self.num_classes = 2
        elif args['dataset_name']=='imdb':
            self.num_classes = 2
        elif args['dataset_name']=='rte':
            self.num_classes = 2
        elif args['dataset_name']=='ag_news':
            self.num_classes = 4
        else:
            self.num_classes = 10
        self.model = self.determine_net(args['net_name'])

    def determine_net(self, net_name, pretrained=False):
        self.logger.debug("determin_net for %s" % net_name)
        self.in_dim = {
            "location": 168,
            "adult": 14,
            "accident": 29,
            "stl10": 96*96*3,
            "cifar10": 32*32*3,
            "cifar100": 32 * 32 * 3,
            "svhn": 32 * 32 * 3,
            "celebA": 128 *128*3,
            "mnist": 28*28*1,
            "fmnist": 28*28*1,
            "cinic10": 32 * 32 * 3,
            "tinyimagenet": 224 * 224 * 3,
            "sst5": 128,
            "news20": 256,
            "snli": 256,
            "mnli": 256,
            "mrpc": 256,
            "imdb": 512,
            "rte": 256,
            "ag_news": 256,
        }
        in_dim = self.in_dim[self.args['dataset_name']]
        out_dim = self.num_classes
        imagenet=self.imagenet
        if net_name == "mlp":
            return MLPTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "logistic":
            return LRTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "simple_cnn":
            return Simple_CNN_Tanh(num_classes=out_dim)
        elif net_name == "simple_cnn_dropout":
            return Simple_CNN_Tanh_dropout(num_classes=out_dim, in_channels=3, input_norm=None, drop_p=0.3)

        elif net_name == "resnet18":
            return resnet18(num_classes=out_dim)
        elif net_name == "resnet18_dp":
            return resnet18_dp(num_classes=out_dim)
        elif net_name == "resnet50":
            return resnet50(num_classes=out_dim)
        elif net_name == "densenet":
            return models.densenet121(num_classes=out_dim)
        elif net_name == "mobilenet":
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, out_dim)
            return model
        elif net_name == "vgg":
            return vgg11_bn(3, out_dim)
        elif net_name == "CNN_MNIST":
            return CNN_MNIST()
        elif net_name == "DT":
            return DecisionTreeClassifier()
        elif net_name == "RF":
            return RandomForestClassifier()
        elif net_name == "pythia70m":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return Pythia70mModel(num_classes=out_dim, max_length=max_length)
        elif net_name == "pythia70m_dropout":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return Pythia70mModel_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.5)
        elif net_name == "roberta":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return RoBERTaModel(num_classes=out_dim, max_length=max_length)
        elif net_name == "roberta_dropout":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return RoBERTaModel_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.95)
        elif net_name == "opt13b":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return OPT13BModel(num_classes=out_dim, max_length=max_length)
        elif net_name == "opt13b_dropout":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return OPT13BModel_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.95)
        elif net_name == "gpt2":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return GPT2SmallModel(num_classes=out_dim, max_length=max_length)
        elif net_name == "gpt2_dropout":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return GPT2SmallModel_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.95)
        elif net_name == "T5":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return T5Model(num_classes=out_dim, max_length=max_length)
        elif net_name == "T5_dropout":
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return T5Model_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.95)

        else:
            raise Exception("invalid net name")

    def train_model(self, train_loader, test_loader, save_name=None):
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'],weight_decay=1e-4)
        if self.args['optim'] == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=0.9, weight_decay=1e-4)

        if self.args['is_dp_defense']:
            privacy_engine = PrivacyEngine(secure_mode=False)

            self.model, optimizer, train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=0.5,
                max_grad_norm=1.0,
            )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        test_acc=0.
        for epoch in range(self.args['num_epochs']):
            losses = []
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if isinstance(batch, dict):
                    data = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    target = batch['labels'].to(self.device)
                else:
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)
                
                self.model.train()
                    
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)

                if self.args['max_norm'] > 0:
                    param_norm = nn.utils.parameters_to_vector(self.model.parameters()).norm()
                    if param_norm > self.args['max_norm']:
                        scale_factor = self.args['max_norm'] / param_norm
                        for param in self.model.parameters():
                            param.data *= scale_factor

            if self.args['is_dp_defense']:
                epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
                print(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"(ε = {epsilon:.2f}, δ = 1e-5)"
                )

            train_acc = train_correct / train_total if train_total > 0 else 0.0
            test_acc = self.test_model_acc(test_loader)
            print(f' epoch:{epoch} | train acc:{round(train_acc, 4)} | test acc: {round(test_acc, 4)}')




    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))

    def predict_proba2(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            feature = test_case[0][0]
            feature = torch.unsqueeze(feature.to(torch.float32), 0).to(self.device)
            logits = self.model(feature)
            posterior = F.softmax(logits, dim=1)
            return posterior.detach().cpu().numpy()

    def predict_proba(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            if isinstance(test_case, dict):
                feature = {k: v.unsqueeze(0).to(self.device) for k, v in test_case.items() if k != 'labels'}
            else:
                feature = torch.unsqueeze(test_case.to(torch.float32), 0).to(self.device)
            
            logits = self.model(feature)
            posterior = F.softmax(logits, dim=1)
            return posterior.detach().cpu().numpy()
    # def predict_proba(self, test_case):
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     with torch.no_grad():
    #         logits = self.model(test_case)
    #         posterior = F.softmax(logits, dim=1)
    #         return posterior.detach().cpu().numpy()

    def test_model_acc(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    data = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    target = batch['labels'].to(self.device)
                else:
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data).to(self.device)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            return correct / len(test_loader.dataset)

    def logits(self,target_sample):
        if isinstance(target_sample, Subset):
            with torch.no_grad():
                feature=target_sample[0][0]
                label=target_sample[0][1]
                feature =torch.unsqueeze(feature.to(torch.float32), 0).to(self.device)
                label=torch.unsqueeze(torch.tensor(label, dtype=torch.long),0).to(self.device)
                logits = self.model(feature).to(self.device)
                logits=logits[0].detach().cpu().numpy()
                probs=np.exp(logits)/np.sum(np.exp(logits))
               # confidence=probs[label]
                confidence=np.max(probs)
                # confidence = np.clip(confidence, 1e-10, 1 - 1e-10)
                # confidence=np.log(confidence/(1-confidence))
        else:
            ValueError("This is not a Subset.")
        return confidence

    def print_dropout_info(self):
        if hasattr(self.model, 'print_dropout_info'):
            self.model.print_dropout_info()
        else:
            print(f"\n=== Dropout 信息 ===")

            dropout_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    dropout_count += 1
                    actual_drop_p = 1.0 - module.p
                    print(f"  {name}: p={module.p:.6f} (保留概率), 实际丢弃概率={actual_drop_p:.6f}")
            if dropout_count > 0:
                print(f"总共找到 {dropout_count} 个Dropout层")
            else:
                print("未找到Dropout层")
            print(f"模型训练模式: {self.model.training}")
            print("=" * 20 + "\n")
    
    def forward_propagation(self, target_sample):
        self.model.eval()
        self.model = self.model.to(self.device)
        return self.model(target_sample)

    def forward(self,x):
        x=self.model(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_dim=3, out_dim=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d( in_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x



class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(3)
        x=self.conv(x)
        return x

def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

class Simple_CNN_Tanh(nn.Module):
    def __init__(self,num_classes=10, in_channels=3, input_norm=None,**kwargs):
        super(Simple_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.num_classes=num_classes

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh
       # act = nn.ReLU

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, self.num_classes))
        else:
            self.classifier = nn.Linear(c * 4 * 4, self.num_classes)
    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Simple_CNN_Tanh_dropout(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_norm=None, drop_p=0.95, **kwargs):
        super(Simple_CNN_Tanh_dropout, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop_p = drop_p
        self.keep_p = 1.0 - drop_p

        self.dropout2d = nn.Dropout2d(self.keep_p) if self.keep_p > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.keep_p) if self.keep_p > 0 else nn.Identity()

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):
        c = self.in_channels

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if self.drop_p > 0:
                    layers += [nn.Dropout2d(self.drop_p)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)
                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(
                nn.Linear(c * 4 * 4, hidden),
                act(),
                nn.Dropout(self.drop_p),
                nn.Linear(hidden, self.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.drop_p),
                nn.Linear(c * 4 * 4, self.num_classes)
            )

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

    def set_dropout_rate(self, drop_p):
        self.drop_p = drop_p
        self.keep_p = 1.0 - drop_p
        
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.p = self.keep_p



class MLPTorchNet(nn.Module):
    def __init__(self, in_dim=168, out_dim=9):
        super(MLPTorchNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # temperature = 4
        # x /= temperature
        # return F.log_softmax(x, dim=1)
        return x


class LRTorchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LRTorchNet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class Pythia70mModel(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="EleutherAI/pythia-70m"):
        super(Pythia70mModel, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):


        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class Pythia70mModel_dropout(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="EleutherAI/pythia-70m", dropout_rate=1.0):

        super(Pythia70mModel_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            hidden_dropout=dropout_rate,
            attention_dropout=dropout_rate,
            classifier_dropout=dropout_rate
        )
        
        keep_prob = 1.0 - dropout_rate
        self.dropout = nn.Dropout(keep_prob)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):

        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

       # logits = self.dropout(outputs.logits)
        
        return logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):

        self.dropout_rate = dropout_rate
        keep_prob = 1.0 - dropout_rate
        self.dropout.p = keep_prob
        self.model.config.hidden_dropout = dropout_rate
        self.model.config.attention_dropout = dropout_rate
        self.model.config.classifier_dropout = dropout_rate


class RoBERTaModel(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="roberta-base"):
        super(RoBERTaModel, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):

        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class RoBERTaModel_dropout(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="roberta-base", dropout_rate=0.95):
        super(RoBERTaModel_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            classifier_dropout=dropout_rate
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = self.dropout(outputs.logits)
        
        return logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate
        if hasattr(self.model.config, 'hidden_dropout_prob'):
            self.model.config.hidden_dropout_prob = dropout_rate
        if hasattr(self.model.config, 'attention_probs_dropout_prob'):
            self.model.config.attention_probs_dropout_prob = dropout_rate
        if hasattr(self.model.config, 'classifier_dropout'):
            self.model.config.classifier_dropout = dropout_rate


class OPT13BModel(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="facebook/opt-1.3b"):
        super(OPT13BModel, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):

        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class OPT13BModel_dropout(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="facebook/opt-1.3b", dropout_rate=0.95):
        super(OPT13BModel_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            hidden_dropout=dropout_rate,
            attention_dropout=dropout_rate,
            classifier_dropout=dropout_rate
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = self.dropout(outputs.logits)
        
        return logits
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate
        self.model.config.hidden_dropout = dropout_rate
        self.model.config.attention_dropout = dropout_rate
        self.model.config.classifier_dropout = dropout_rate


class GPT2SmallModel(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="gpt2"):
        super(GPT2SmallModel, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):

        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class GPT2SmallModel_dropout(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="gpt2", dropout_rate=0.95):
        super(GPT2SmallModel_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = self.dropout(outputs.logits)
        
        return logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate
        if hasattr(self.model.config, 'resid_pdrop'):
            self.model.config.resid_pdrop = dropout_rate
        if hasattr(self.model.config, 'attn_pdrop'):
            self.model.config.attn_pdrop = dropout_rate


class T5Model(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="t5-small"):
        super(T5Model, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):

        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        vocab_size = self.model.config.vocab_size
        input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
        
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            seq_lengths = torch.clamp(seq_lengths, min=0)
        else:
            non_pad_mask = (input_ids != pad_token_id) & (input_ids != 0)
            seq_lengths = (non_pad_mask.long().sum(dim=1) - 1).clamp(min=0)
        
        for i in range(batch_size):
            last_idx = seq_lengths[i].item()
            input_ids[i, last_idx] = eos_token_id
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class T5Model_dropout(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="t5-small", dropout_rate=0.95):
        super(T5Model_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        if hasattr(self.model.config, 'dropout_rate'):
            self.model.config.dropout_rate = dropout_rate
        if hasattr(self.model.config, 'layer_dropout'):
            self.model.config.layer_dropout = dropout_rate
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        vocab_size = self.model.config.vocab_size
        input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
        

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            seq_lengths = torch.clamp(seq_lengths, min=0)
        else:
            non_pad_mask = (input_ids != pad_token_id) & (input_ids != 0)
            seq_lengths = (non_pad_mask.long().sum(dim=1) - 1).clamp(min=0)
        
        for i in range(batch_size):
            last_idx = seq_lengths[i].item()
            input_ids[i, last_idx] = eos_token_id
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = self.dropout(outputs.logits)
        
        return logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):
        """动态设置dropout率"""
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate
        if hasattr(self.model.config, 'dropout_rate'):
            self.model.config.dropout_rate = dropout_rate
        if hasattr(self.model.config, 'layer_norm_epsilon'):
            pass
