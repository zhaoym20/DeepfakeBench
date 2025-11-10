import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import logging
from utils.tools import print_cpu_gpu_usage
logger = logging.getLogger(__name__)
from torchvision.transforms import ToPILImage
unloader = ToPILImage()
def sample_selection(feature,center,num_sample,prob,image,mode='center',shuf_dis=None):
    centroid = center.unsqueeze(-1)  # [128,1]
    distance = torch.mm(feature, centroid).squeeze()
    if distance.shape[0] < num_sample:
        num_sample = distance.shape[0]
    if mode=='hard':
        # 先按欧氏距离排序，然后等距选n/2个，然后在n/2区间中再选余弦相似度最远的n/2个。
        values, indices = distance.topk(num_sample // 4)
        _, hard_indices = prob.topk(num_sample // 4)
        selected_images = image[torch.cat((indices, hard_indices))]
    elif mode=='moderate':
        # 先按欧氏距离排序，然后等距选n个
        values, indices = distance.topk(num_sample // 4)
        moderate_index = torch.argsort(distance)[
                         len(distance) // 2 - num_sample // 8: len(distance) // 2 + num_sample // 8]
        selected_images = image[torch.cat((indices, moderate_index))]
    elif mode=='center':
        # 直接选top n个
        values, indices = distance.topk(num_sample // 2)
        selected_images = image[indices]
    elif mode=='sparse':
        sorted_indices = torch.argsort(distance, descending=True)
        # 等距采样，选择 n 个索引
        step = len(sorted_indices) // (num_sample // 2)
        selected_indices = []
        for i in range(0, len(sorted_indices), step):
            if i + step > distance.shape[0]-1 or len(selected_indices)==num_sample:
                break
            chunk = sorted_indices[i:min(i+step,distance.shape[0]-1)]
            chunk_center = sorted_indices[min(i+step//2,(distance.shape[0]-1-i) // 2 +i)]
            selected_indices.append(chunk_center)
            latest_small=2
            latest_idx = -1
            for each in chunk:

                tensor1_norm = torch.nn.functional.normalize(feature[each]-centroid, p=2, dim=0)
                tensor2_norm = torch.nn.functional.normalize(feature[chunk_center]-centroid, p=2, dim=0)

                cosine_similarity = torch.dot(tensor1_norm, tensor2_norm)
                if cosine_similarity<latest_small:
                    latest_small=cosine_similarity
                    latest_idx = each
            selected_indices.append(latest_idx)
        selected_images=image[torch.tensor(selected_indices)]
    elif mode == 'sparse_robust':

        sorted_indices = torch.argsort(distance, descending=True)
        # 等距采样，选择 n 个索引
        step = len(sorted_indices) // (num_sample // 2)
        selected_indices = []
        for i in range(0, len(sorted_indices), step):
            if i + step > distance.shape[0]-1 or len(selected_indices)==num_sample:
                break
            chunk = sorted_indices[i:min(i+step,distance.shape[0]-1)]
            robust_point_max_index = chunk[torch.argsort(shuf_dis[chunk])[-1]]
            chunk_center = robust_point_max_index
            selected_indices.append(chunk_center)
            latest_small=2
            latest_idx = -1
            for each in chunk:
                tensor1_norm = torch.nn.functional.normalize(feature[each], p=2, dim=0)
                tensor2_norm = torch.nn.functional.normalize(feature[chunk_center], p=2, dim=0)
                cosine_similarity = torch.dot(tensor1_norm, tensor2_norm)
                if cosine_similarity<latest_small:
                    latest_small=cosine_similarity
                    latest_idx = each
            selected_indices.append(latest_idx)
        selected_images=image[torch.tensor(selected_indices)]
    return selected_images

def grid_shuffle_tensor(tensor, grid_size=128):
    batch_size, channels, height, width = tensor.shape
    assert height % grid_size == 0 and width % grid_size == 0, "height 和 width 必须是 grid_size 的倍数"

    num_grid_h = height // grid_size
    num_grid_w = width // grid_size
    tensor_reshaped = tensor.view(batch_size, channels, num_grid_h, grid_size, num_grid_w, grid_size)
    tensor_reshaped = tensor_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    flat_grids = tensor_reshaped.view(batch_size, channels, num_grid_h * num_grid_w, grid_size, grid_size)
    permuted_indices = torch.randperm(num_grid_h * num_grid_w)
    shuffled_flat_grids = flat_grids[:, :, permuted_indices, :, :]
    shuffled_grids = shuffled_flat_grids.view(batch_size, channels, num_grid_h, num_grid_w, grid_size, grid_size)
    shuffled_tensor = shuffled_grids.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, channels, height, width)

    return shuffled_tensor


def get_memory(model, dataloader, alpha, num_sample, device, config):
    def model_forward(image, model, post_function=nn.Sigmoid()):
        data_dict={}
        data_dict['image']=image
        output = model(data_dict,inference=True)
        logit=output['cls']
        prob=output['prob']
        feat=output.get('od-feat',None)
        pred=(prob >= 0.5).float()
        return pred,prob,feat
    centroid_real = None
    features_real = None
    imagesets_real = None
    prob_real=None
    centroid_fake = None
    features_fake = None
    imagesets_fake = None
    prob_fake=None
    mode = config['replay_mode']
    cent_ma =config['center_move_avg']
    for i, datas in enumerate(tqdm(dataloader, desc="Generating Memory")):
        images = datas['image'].to(device)
        targets = datas['label'].to(device).float()
        if images[targets == 0.0].shape[0] != 0:
            with torch.no_grad():
                _,prob,feature = model_forward(images[targets==0.0], model)
            if features_real!=None:
                prob_real=torch.cat((prob_real,prob.cpu()),dim=0)
                features_real = torch.cat((features_real,feature.cpu()),dim=0)
                imagesets_real = torch.cat((imagesets_real,images[targets==0.0].cpu()),dim=0)
            else:
                prob_real=prob.cpu()
                features_real = feature.cpu()
                imagesets_real = images[targets==0.0].cpu()
            if cent_ma:
                for j in range(feature.shape[0]):#process batch
                    if centroid_real == None:
                        centroid_real = feature[j].squeeze().cpu()
                    else:
                        centroid_real = (1-alpha)*centroid_real + (alpha)*feature[j].squeeze().cpu()
        if images[targets == 1.0].shape[0] != 0:
            with torch.no_grad():
                _,prob,feature = model_forward(images[targets==1.0], model)
            if features_fake!=None:
                prob_fake=torch.cat((prob_fake,prob.cpu()),dim=0)
                features_fake = torch.cat((features_fake,feature.cpu()),dim=0)
                imagesets_fake = torch.cat((imagesets_fake,images[targets==1.0].cpu()),dim=0)
            else:
                prob_fake=prob.cpu()
                features_fake = feature.cpu()
                imagesets_fake = images[targets==1.0].cpu()
            if cent_ma:
                for j in range(feature.shape[0]):#process batch
                    if centroid_fake == None:
                        centroid_fake = feature[j].squeeze().cpu()
                    else:
                        centroid_fake = (1-alpha)*centroid_fake + (alpha)*feature[j].squeeze().cpu()
    if not cent_ma:
        centroid_real = torch.mean(features_real, dim=0).squeeze().cpu()
        centroid_fake = torch.mean(features_fake, dim=0).squeeze().cpu()
    logger.info("~~~~~~~generate centroid done!~~~~~~~~")
    selected_real_images = sample_selection(features_real,centroid_real,num_sample,prob_real,imagesets_real,mode=mode)
    selected_fake_images = sample_selection(features_fake,centroid_fake,num_sample,prob_fake,imagesets_fake,mode=mode)
    return selected_real_images,selected_fake_images,centroid_real,centroid_fake
normalizer = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10) # 缩放长度

def get_memory_v2(model, dataloader, alpha, num_sample, device, config):
    def model_forward(image, model, post_function=nn.Sigmoid()):
        data_dict={}
        data_dict['image']=image

        output = model(data_dict,inference=True)
        logit=output['cls']
        prob=output['prob']
        feat=output.get('od-feat',None)
        pred=(prob >= 0.5).float()
        return pred,prob,feat
    centroid_real = None
    features_real = None
    imagesets_real = None
    shuf_dis_real = None
    prob_real=None
    centroid_fake = None
    features_fake = None
    imagesets_fake = None
    shuf_dis_fake = None
    prob_fake=None
    mode = config['replay_mode']
    cent_ma =config['center_move_avg']
    for i, datas in enumerate(tqdm(dataloader, desc="Generating Memory")):
        images = datas['image'].to(device)
        targets = datas['label'].to(device).float()
        shuffled_imgs = grid_shuffle_tensor(images)
        if images[targets == 0.0].shape[0] != 0:
            with torch.no_grad():
                _,prob,feature = model_forward(images[targets==0.0], model)
                _,_,feature_shuf = model_forward(shuffled_imgs[targets==0.0], model)
                try:
                    real_shuff_distances = torch.diag(torch.mm(feature, feature_shuf.T).squeeze())
                except Exception as e:
                    real_shuff_distances = torch.mm(feature, feature_shuf.T).squeeze().unsqueeze(0)
            if features_real!=None:
                prob_real=torch.cat((prob_real,prob.cpu()),dim=0)
                features_real = torch.cat((features_real,feature.cpu()),dim=0)
                imagesets_real = torch.cat((imagesets_real,images[targets==0.0].cpu()),dim=0)
                shuf_dis_real = torch.cat((shuf_dis_real,real_shuff_distances.cpu()),dim=0)
            else:
                shuf_dis_real=real_shuff_distances.cpu()
                prob_real=prob.cpu()
                features_real = feature.cpu()
                imagesets_real = images[targets==0.0].cpu()
            if cent_ma:
                for j in range(feature.shape[0]):#process batch
                    if centroid_real == None:
                        centroid_real = feature[j].squeeze().cpu()
                    else:
                        centroid_real = (1-alpha)*centroid_real + (alpha)*feature[j].squeeze().cpu()
        if images[targets == 1.0].shape[0] != 0:
            with torch.no_grad():
                _,prob,feature = model_forward(images[targets==1.0], model)
                _,_,feature_shuf = model_forward(shuffled_imgs[targets==1.0], model)
                try:
                    fake_shuff_distances = torch.diag(torch.mm(feature, feature_shuf.T).squeeze())
                except Exception as e:
                    fake_shuff_distances = torch.mm(feature, feature_shuf.T).squeeze().unsqueeze(0)
            if features_fake!=None:
                prob_fake=torch.cat((prob_fake,prob.cpu()),dim=0)
                features_fake = torch.cat((features_fake,feature.cpu()),dim=0)
                imagesets_fake = torch.cat((imagesets_fake,images[targets==1.0].cpu()),dim=0)
                shuf_dis_fake = torch.cat((shuf_dis_fake,fake_shuff_distances.cpu()),dim=0)
            else:
                shuf_dis_fake=fake_shuff_distances.cpu()
                prob_fake=prob.cpu()
                features_fake = feature.cpu()
                imagesets_fake = images[targets==1.0].cpu()
            if cent_ma:
                for j in range(feature.shape[0]):#process batch
                    if centroid_fake == None:
                        centroid_fake = feature[j].squeeze().cpu()
                    else:
                        centroid_fake = (1-alpha)*centroid_fake + (alpha)*feature[j].squeeze().cpu()


    if not cent_ma:
        centroid_real = torch.mean(features_real, dim=0).squeeze().cpu()
        centroid_fake = torch.mean(features_fake, dim=0).squeeze().cpu()
    logger.info("~~~~~~~generate centroid done!~~~~~~~~")
    selected_real_images = sample_selection(features_real,centroid_real,num_sample,prob_real,imagesets_real,mode=mode,shuf_dis=shuf_dis_real)
    selected_fake_images = sample_selection(features_fake,centroid_fake,num_sample,prob_fake,imagesets_fake,mode=mode,shuf_dis=shuf_dis_fake)
    return selected_real_images,selected_fake_images,centroid_real,centroid_fake
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = float(temperature)
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """Supervised Contrastive Loss (device-safe & NaN-safe).
        features: (B, D)
        labels: (B,) long or (B,1)
        mask: (B,B) binary; if provided labels must be None.
        """
        if features.dim() != 2:
            raise ValueError(f"SupConLoss expects 2D features (B,D), got shape {features.shape}")
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.size(0)

        # Build mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.view(-1, 1)
            if labels.size(0) != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = (labels == labels.T).float().to(device)
        else:  # mask provided
            mask = mask.float().to(device)

        # Similarity / temperature scaling
        logits = torch.matmul(features, features.T) / self.temperature
        # Numerical stability
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)

        # Remove self-contrast terms
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        positives_mask = mask * logits_mask
        negatives_mask = 1.0 - mask

        num_pos = positives_mask.sum(dim=1)  # (B,)

        # Denominator: sum over all positives + negatives (excluding self)
        denominator = (exp_logits * (positives_mask + negatives_mask)).sum(dim=1, keepdim=True)
        denominator = denominator.clamp_min(1e-12)  # avoid log(0)

        log_probs = logits - torch.log(denominator)
        # Aggregate positive log-probs; skip rows with zero positives
        pos_log_prob = (log_probs * positives_mask).sum(dim=1)
        valid = num_pos > 0
        if not valid.any():
            return logits.new_tensor(0.0)
        pos_log_prob = pos_log_prob[valid] / num_pos[valid]

        loss = -pos_log_prob
        if self.scale_by_temperature:
            loss = loss * self.temperature
        return loss.mean()

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim=2, bn=False):  # 修改为 output_dim=2
        super(LogisticRegression, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))  # 输出维度修改为 output_dim
        self.bias = nn.Parameter(torch.Tensor(output_dim))  # 修改 bias 的形状
        self.bias.data.zero_()
        self.reset_parameters()
        self.bn = bn
        if bn:
            self.bn_layer = nn.BatchNorm1d(output_dim)  # BatchNorm 维度也需要修改

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        logit = F.linear(x, self.weight, self.bias)
        if self.bn:
            logit = self.bn_layer(logit)
        return logit  # 返回 logit

if __name__ == '__main__':
    feat=torch.rand((1024,1792))
    cent=torch.rand((1792))
    prob=torch.rand((1024))
    shuf_dis=torch.rand((1024))
    images=torch.rand((1024,3,256,256))
    sample_selection(feat,cent,504,prob,images,mode='sparse_robust',shuf_dis=shuf_dis)