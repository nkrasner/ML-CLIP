import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import XLMRobertaModel, ViTModel, ViTImageProcessor, AutoTokenizer
import wandb
# import transformers
# from transformers import AutoProcessor, CLIPModel

ROBERTA_MODEL = "xlm-roberta-large"
VIT_MODEL = "google/vit-base-patch16-224-in21k"


class MultiLingualCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, caption_file):
        self.images = {}
        for image_file in os.listdir(image_dir):
            index = int(image_file.split(".")[0])
            # with Image.open(os.path.join(image_dir, image_file)) as img:
            #     self.images[index] = img.copy()
            self.images[index] = os.path.join(image_dir, image_file)
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        self.image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL)
        with open(caption_file, "r") as f:
            self.captions = json.load(f)
        self.langs = list(self.captions[0]["captions"].keys())
        # self.max_len = max([max([len(self.tokenizer.encode(caption["captions"][lang])) for caption in self.captions]) for lang in self.langs])
        self.max_len = 128
        


    def __len__(self):
        return len(self.images)
    
    def preprocess_text(self, text):
        return torch.Tensor(self.tokenizer.encode(text, padding="max_length", max_length=self.max_len)).long()
    
    def __getitem__(self, idx):
        cap_dict = self.captions[idx]
        caps = [self.preprocess_text(cap_dict["captions"][lang]) for lang in self.langs]
        with Image.open(self.images[cap_dict["image_id"]]) as img:
            image = self.image_processor.preprocess(img.convert("RGB"), return_tensors="pt")
        # image = self.image_processor(self.images[cap_dict["image_id"]], return_tensors="pt")
        return image, *caps
        


class ClipModel(nn.Module):
    def __init__(self):
        super(ClipModel, self).__init__()
        # Define your layers here
        # Encoders
        self.text_encoder = XLMRobertaModel.from_pretrained(ROBERTA_MODEL)
        self.image_encoder = ViTModel.from_pretrained(VIT_MODEL)
        self.text_dense = nn.Linear(self.text_encoder.config.hidden_size, 512)
        self.image_dense = nn.Linear(self.image_encoder.config.hidden_size, 512)
        self.temp = nn.Parameter(torch.ones(1))
        self.encoders_frozen = False
        
    def set_freeze_encoders(self, val=True):
        for param in self.text_encoder.parameters():
            param.requires_grad = not val
        for param in self.image_encoder.parameters():
            param.requires_grad = not val
        self.encoders_frozen = val
        

    def forward(self, img, cap, attn_mask=None):
        # Define the forward pass of your model
        # Use the defined layers to process the input and return the output
        t_enc = self.text_dense(self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state[:, 0, :])
        i_enc = self.image_dense(self.image_encoder(pixel_values=img.pixel_values).last_hidden_state[:, 0, :])
        scores = torch.matmul(t_enc, i_enc.T) * self.temp
        return scores




def train(model, device, train_loader, optimizer, epochs, scheduler=None, val_loader=None, val_interval=100, warmup_ratio=None, save_file="models/clip-model.pt"):
    min_val_loss = float("inf")
    if warmup_ratio is not None:
        model.set_freeze_encoders(True)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader)
        sum_loss = 0
        pbar.set_description(f"Epoch {epoch}, LR {scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']}")
        for batch_idx, (image, *caps) in enumerate(pbar):
            if warmup_ratio is not None:
                if model.encoders_frozen and batch_idx / len(train_loader) >= warmup_ratio:
                    tqdm.write(f"Thawing encoders at step {batch_idx}/{len(train_loader)}")
                    model.set_freeze_encoders(False)
            image.pixel_values = image.pixel_values.squeeze(1).to(device)
            image = image
            caps = torch.cat(caps, dim=0).to(device)
            attn_mask = torch.ones_like(caps).float().to(device)
            attn_mask[caps == 0] = 0
            optimizer.zero_grad()
            output = model(image, caps, attn_mask)
            loss = F.cross_entropy(output, torch.tile(torch.arange(output.shape[0]//len(train_loader.dataset.langs)), (len(train_loader.dataset.langs),)).to(device))
            loss.backward()
            sum_loss += loss.item() / output.shape[0]
            optimizer.step()
            pbar.set_postfix(loss=loss.item() / output.shape[0], avg_loss=sum_loss/(batch_idx+1))
            wandb.log({"train_loss": loss.item() / output.shape[0], "avg_train_loss": sum_loss/(batch_idx+1), "learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']})
        

            if val_loader is not None and batch_idx > 0 and batch_idx % val_interval == 0:
                model.eval()
                sum_val_loss = 0
                with torch.no_grad():
                    for image, *caps in tqdm(val_loader, desc="Validation", leave=False):
                        image.pixel_values = image.pixel_values.squeeze(1).to(device)
                        image = image
                        caps = torch.cat(caps, dim=0).to(device)
                        attn_mask = torch.ones_like(caps).float().to(device)
                        attn_mask[caps == 0] = 0
                        output = model(image, caps, attn_mask)
                        loss = F.cross_entropy(output, torch.tile(torch.arange(output.shape[0]//len(val_loader.dataset.langs)), (len(val_loader.dataset.langs),)).to(device))
                        sum_val_loss += loss.item() / output.shape[0]
                tqdm.write(f"Epoch: {epoch}, Step: {batch_idx}, Validation Loss: {sum_val_loss/len(val_loader)}")
                wandb.log({"val_loss": sum_val_loss/len(val_loader)})
                if sum_val_loss < min_val_loss:
                    min_val_loss = sum_val_loss
                    torch.save(model.state_dict(), save_file)
            
        if scheduler is not None:
            scheduler.step()











if __name__ == "__main__":
    wandb.login()
    
    batch_size = 32  # Set the batch size.
    learning_rate = 1e-5  # Set the learning rate.
    warmup_ratio = 0.5
    
    model = ClipModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_dataset = MultiLingualCLIPDataset(os.path.join("data","images","train2017"), os.path.join("data","annotations","captions_train2017_en_only.json"))
    # train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "train2017"), os.path.join("data", "annotations", "captions_train2017_ordered_translated.json"))
    print(f"Loaded {len(train_dataset)} training examples")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "val2017"), os.path.join("data", "annotations", "captions_val2017_en_only.json"))
    # val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "val2017"), os.path.join("data", "annotations", "captions_val2017_ordered_translated.json"))
    print(f"Loaded {len(val_dataset)} validation examples")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_epochs = 10  # Set the total number of epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / total_epochs)
    
    wandb.init(
      # set the wandb project where this run will be logged
      project="DL Project",
      name=f"english-same-params-as-best",
      # track hyperparameters and run metadata
      config={
        "lr-init": learning_rate,
        "epochs": total_epochs,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
      }
    )
    
    train(model, device, train_loader, optimizer, epochs=total_epochs, scheduler=scheduler, val_loader=val_loader, val_interval=300, warmup_ratio=warmup_ratio, save_file=f"models/en-clip-model-{learning_rate}-{total_epochs}-{warmup_ratio}.pt")
    wandb.finish()
