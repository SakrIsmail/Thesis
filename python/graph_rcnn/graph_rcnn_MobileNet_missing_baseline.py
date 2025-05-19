import os
import json
import random
from tqdm import tqdm
import time
import psutil
import gc
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from codecarbon import EmissionsTracker
from torchvision.ops import box_iou
from torchvision.ops import nms
from torch_geometric.nn import GATConv

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


final_output_json='/var/scratch/sismail/data/processed/final_direct_missing.json'
image_directory = '/var/scratch/sismail/data/images'

test_ratio = 0.2
valid_ratio = 0.1
random_seed = 42

with open(final_output_json, 'r') as f:
    annotations = json.load(f)

image_filenames = list(annotations['images'].keys())

random.seed(random_seed)
random.shuffle(image_filenames)

num_test = int(len(image_filenames) * test_ratio)
test_images = image_filenames[:num_test]
train_images = image_filenames[num_test:]
num_valid = int(len(train_images) * valid_ratio)
valid_images = train_images[:num_valid]
train_images = train_images[num_valid:]


train_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in train_images}
}

valid_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in valid_images}
}

test_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in test_images}
}


class BikePartsDetectionDataset(Dataset):
    def __init__(self, annotations_dict, image_dir, augment=True, target_size=(640,640)):
        self.all_parts       = annotations_dict['all_parts']
        self.part_to_idx     = {p: i+1 for i, p in enumerate(self.all_parts)}  # background=0
        self.image_data      = annotations_dict['images']
        self.image_filenames = list(self.image_data.keys())
        self.image_dir       = image_dir
        self.augment         = augment
        self.target_size     = target_size

    def __len__(self):
        return len(self.image_filenames) * (2 if self.augment else 1)

    def apply_augmentation(self, image, boxes):
        # horizontal flip
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            w = image.width
            boxes = boxes.clone()
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
        # color jitters
        if random.random() < 0.8:
            image = transforms.functional.adjust_brightness(image, random.uniform(0.6,1.4))
        if random.random() < 0.8:
            image = transforms.functional.adjust_contrast(image, random.uniform(0.6,1.4))
        if random.random() < 0.5:
            image = transforms.functional.adjust_saturation(image, random.uniform(0.7,1.3))
        return image, boxes

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_filenames)
        do_aug   = self.augment and (idx >= len(self.image_filenames))

        fn = self.image_filenames[real_idx]
        img = Image.open(os.path.join(self.image_dir, fn)).convert('RGB')
        ow, oh = img.size

        parts = self.image_data[fn]['parts']
        boxes, labels, is_missing = [], [], []
        for part in parts:
            bb = part['absolute_bounding_box']
            x0, y0 = bb['left'], bb['top']
            x1, y1 = x0 + bb['width'], y0 + bb['height']
            boxes.append([x0,y0,x1,y1])
            idx = self.part_to_idx[part['part_name']]
            labels.append(idx)
            is_missing.append(0 if part['present'] else 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        is_missing = torch.tensor(is_missing, dtype=torch.int64)

        if do_aug:
            img, boxes = self.apply_augmentation(img, boxes)

        img = transforms.functional.resize(img, self.target_size)
        sx, sy = self.target_size[0]/ow, self.target_size[1]/oh
        boxes[:,[0,2]] *= sx; boxes[:,[1,3]] *= sy
        img = transforms.functional.to_tensor(img)

        return img, {'boxes': boxes, 'labels': labels, 'is_missing': is_missing}



train_dataset = BikePartsDetectionDataset(
    annotations_dict=train_annotations,
    image_dir=image_directory,
    augment=False
)

valid_dataset = BikePartsDetectionDataset(
    annotations_dict=valid_annotations,
    image_dir=image_directory,
    augment=False
)

test_dataset = BikePartsDetectionDataset(
    annotations_dict=test_annotations,
    image_dir=image_directory,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    worker_init_fn=seed_worker,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch))
)

def visualize_and_save_predictions(model, dataset, device, out_dir="output_preds", n_images=5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    for idx in range(min(n_images, len(dataset))):
        img, target = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]

        img_np = img.mul(255).permute(1,2,0).byte().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_np)

        gt_boxes = target['boxes'][target['is_missing']==1].cpu().numpy()
        for x0,y0,x1,y1 in gt_boxes:
            rect = patches.Rectangle((x0,y0), x1-x0, y1-y0,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        pred_boxes = pred['boxes_missing'].cpu().numpy()
        for x0,y0,x1,y1 in pred_boxes:
            rect = patches.Rectangle((x0,y0), x1-x0, y1-y0,
                                     linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')
        plt.savefig(os.path.join(out_dir, f"graph_pred_{idx}.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def evaluate_model(model, loader, all_parts, device, iou_thr=0.5):
    model.eval()
    P = len(all_parts)
    results = []

    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [img.to(device) for img in imgs]
            outs = model(imgs)

            for tgt, out in zip(tgts, outs):
                vec_true = torch.zeros(P, dtype=torch.int64)
                for lbl, miss in zip(tgt['labels'], tgt['is_missing']):
                    if miss.item() == 1:
                        vec_true[lbl.item()-1] = 1

                vec_pred = torch.zeros(P, dtype=torch.int64)
                boxes_p  = out['boxes_missing']
                labels_p = out['labels_missing']
                for b_pred, lbl_pred in zip(boxes_p, labels_p):
                    idxs = [i for i,(l,_) in enumerate(zip(tgt['labels'], tgt['boxes'])) 
                            if l.item()-1 == lbl_pred.item()]
                    if not idxs:
                        continue
                    gt_boxes = tgt['boxes'][idxs]
                    ious     = box_iou(b_pred[None], gt_boxes)
                    if ious.max() >= iou_thr:
                        vec_pred[lbl_pred.item()] = 1

                results.append({
                    'true_missing_parts':      vec_true.cpu().numpy(),
                    'predicted_missing_parts': vec_pred.cpu().numpy()
                })
    return results


def part_level_evaluation(results, all_parts):
    """
    results: list of dicts with keys
        - 'predicted_missing_parts': List[int] of length P (0/1 for each part)
        - 'true_missing_parts':      List[int] of length P (0/1 for each part)
    all_parts:  List[str] of length P, giving the part names in order
    """
    # Stack into (N_images, P) arrays
    Y_true = np.array([r['true_missing_parts']      for r in results])
    Y_pred = np.array([r['predicted_missing_parts'] for r in results])

    # Overall (flattened) metrics
    micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    overall_acc = accuracy_score(Y_true.flatten(), Y_pred.flatten())
    overall_prec = precision_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_rec  = recall_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_f1   = f1_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)

    # Miss-rate and FPPI
    FN = np.logical_and(Y_true==1, Y_pred==0).sum()
    TP = np.logical_and(Y_true==1, Y_pred==1).sum()
    FP = np.logical_and(Y_true==0, Y_pred==1).sum()
    miss_rate = FN/(FN+TP) if (FN+TP)>0 else 0.0
    fppi = FP / len(results)

    print(f"[METRIC] Micro-F1: {micro_f1:.4f}")
    print(f"[METRIC] Macro-F1: {macro_f1:.4f}")
    print(f"[METRIC] Miss Rate: {miss_rate:.4f}")
    print(f"[METRIC] FPPI: {fppi:.4f}")
    print(f"[METRIC] Overall Acc: {overall_acc:.4f}")
    print(f"[METRIC] Precision: {overall_prec:.4f}")
    print(f"[METRIC] Recall: {overall_rec:.4f}")
    print(f"[METRIC] F1: {overall_f1:.4f}")

    # Per-part table
    table = []
    for j, part_name in enumerate(all_parts):
        acc  = accuracy_score(Y_true[:,j], Y_pred[:,j])
        prec = precision_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        rec  = recall_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        f1   = f1_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        table.append([part_name, f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

    print("\n[PER-PART EVALUATION]")
    print(tabulate(table, headers=["Part","Acc","Prec","Rec","F1"], tablefmt="fancy_grid"))
        


class HallucinationFasterRCNN(nn.Module):
    def __init__(self, all_parts, trainable_backbone_layers=3):
        super().__init__()
        P = len(all_parts)

        base_model = fasterrcnn_mobilenet_v3_large_fpn(
            weights='DEFAULT',
            trainable_backbone_layers=trainable_backbone_layers
        )
        backbone = base_model.backbone
        default_anchor_generator = base_model.rpn.anchor_generator

        num_anchors = default_anchor_generator.num_anchors_per_location()[0]
        rpn_head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=num_anchors * P
        )

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=2,
            rpn_anchor_generator=default_anchor_generator,
            rpn_head=rpn_head,
            trainable_backbone_layers=trainable_backbone_layers
        )

        self.P = P


    def forward(self, images, targets=None):
        if self.training:
            new_t = []
            for t in targets:
                labs = []
                for lbl, miss in zip(t['labels'], t['is_missing']):
                    k = lbl.item() - 1
                    cls = 1 + k if miss.item()==0 else 1 + self.P + k
                    labs.append(cls)
                new_t.append({'boxes': t['boxes'], 'labels': torch.tensor(labs, device=t['boxes'].device)})
            return self.model(images, new_t)
        else:
            outs = self.model(images)
            final = []
            for out in outs:
                boxes, labels, scores = out['boxes'], out['labels'], out['scores']
                miss_boxes, miss_scores, miss_labels = [], [], []
                pres_boxes, pres_scores, pres_labels = [], [], []
                for b, l, s in zip(boxes, labels, scores):
                    part_idx = (l-1).item()
                    if 1 <= l <= self.P:
                        pres_boxes.append(b); pres_scores.append(s); pres_labels.append(part_idx)
                    elif self.P+1 <= l <= 2*self.P:
                        miss_boxes.append(b); miss_scores.append(s); miss_labels.append(part_idx)
                final.append({
                    'boxes_present':   torch.stack(pres_boxes)   if pres_boxes else torch.zeros((0,4), device=boxes.device),
                    'scores_present':  torch.tensor(pres_scores,device=boxes.device) if pres_scores else torch.tensor([],device=boxes.device),
                    'labels_present':  torch.tensor(pres_labels,device=boxes.device) if pres_labels else torch.tensor([],device=boxes.device),
                    'boxes_missing':   torch.stack(miss_boxes)   if miss_boxes else torch.zeros((0,4), device=boxes.device),
                    'scores_missing':  torch.tensor(miss_scores,device=boxes.device) if miss_scores else torch.tensor([],device=boxes.device),
                    'labels_missing':  torch.tensor(miss_labels,device=boxes.device) if miss_labels else torch.tensor([],device=boxes.device),
                })
            return final
        

class RelationProposalNetwork(nn.Module):
    def __init__(self, in_c, hidden_c=256):
        super().__init__()
        self.fc1 = nn.Linear(in_c*2 + 4, hidden_c)
        self.fc2 = nn.Linear(hidden_c, 1)

    def forward(self, feats, boxes):
        N, C = feats.size()
        f1 = feats.unsqueeze(1).expand(-1, N, -1)
        f2 = feats.unsqueeze(0).expand(N, -1, -1)
        b1 = boxes.unsqueeze(1).expand(-1, N, -1)
        b2 = boxes.unsqueeze(0).expand(N, -1, -1)
        geom = torch.abs(b1 - b2)
        x = torch.relu(self.fc1(torch.cat([f1, f2, geom], dim=-1)))
        return self.fc2(x).squeeze(-1)

class AttentionalGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_c, hid_c, heads=heads)
        self.gat2 = GATConv(hid_c*heads, out_c, heads=1)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)

class GraphHallucinationRCNN(nn.Module):
    def __init__(self, all_parts, k=3, hidden_c=256):
        """
        Graph-based Faster R-CNN that jointly detects parts and learns co-missing relationships.

        Args:
            all_parts (List[str]): List of part names.
            k (int): Number of top edges per node for graph construction.
            hidden_c (int): Hidden channels for the GCN.
        """
        super().__init__()
        self.detector = HallucinationFasterRCNN(all_parts)
        num_node_classes = 2
        self.k = k


        in_feats = (
            self.detector.model.roi_heads.box_predictor.cls_score.in_features + 4
        )
        self.repn = RelationProposalNetwork(
            in_c=self.detector.model.roi_heads.box_predictor.cls_score.in_features
        )
        self.agcn = AttentionalGCN(in_feats, hidden_c, num_node_classes)

        self.log_sigma_gcn = nn.Parameter(torch.zeros(()))
        self.log_sigma_repnet = nn.Parameter(torch.zeros(()))

    def forward(self, images, targets=None):
        """
        Training: returns (total_loss, loss_dict).
        Inference: returns list of detection dicts with co-missing refinement.
        """
        if self.training:
            loss_dict = self.detector(images, targets)
            box_loss  = sum(loss_dict.values())

            with torch.no_grad():
                det_outs = self.detector(images)
            
            for det, img in zip(det_outs, images):
                det['image'] = img

            gcn_loss, repnet_loss = self.compute_gcn_loss_on_proposals(det_outs, targets)

            total_loss = box_loss + gcn_loss + repnet_loss
            return total_loss, {
                'box_loss':     box_loss,
                'gcn_loss':     gcn_loss,
                'repnet_loss':  repnet_loss,
                'total_loss':   total_loss
            }


        dets = self.detector(images)
        outputs = []
        for img, det in zip(images, dets):
            b_m, s_m, l_m = det['boxes_missing'], det['scores_missing'], det['labels_missing']
            b_p, s_p, l_p = det['boxes_present'], det['scores_present'], det['labels_present']

            boxes_all  = torch.cat([b_p, b_m], dim=0)
            labels_all = torch.cat([l_p, l_m], dim=0)
            scores_det = torch.cat([s_p, s_m], dim=0)

            if boxes_all.numel()==0:
                outputs.append(det); continue

            feats = self._get_roi_feats(img.unsqueeze(0), boxes_all).squeeze(0)
            geom  = self._box_geom(boxes_all, img.shape[-2:])
            nf    = torch.cat([feats, geom], dim=1)

            rel      = self.repn(feats, boxes_all)
            edge_idx = self._make_edge_index(rel)
            logits   = self.agcn(nf, edge_idx)
            probs    = torch.softmax(logits, dim=1)
            miss_pr  = probs[:,1]

            alpha = 0.5
            comb_sc = alpha * scores_det + (1-alpha) * miss_pr

            keep    = comb_sc > 0.3
            b2       = boxes_all[keep]
            sc2      = comb_sc[keep]
            lbl2     = labels_all[keep]
            keep_idx = nms(b2, sc2, iou_threshold=0.5)

            outputs.append({
                'boxes_missing':  b2[keep_idx],
                'scores_missing': sc2[keep_idx],
                'labels_missing': lbl2[keep_idx]
            })
        return outputs

    def compute_gcn_loss_on_proposals(self, det_outs, targets):
        gcn_preds, gcn_labels = [], []
        rep_sum = 0.0

        for det, tgt in zip(det_outs, targets):
            boxes_all  = torch.cat([det['boxes_present'], det['boxes_missing']], dim=0)
            scores_all = torch.cat([det['scores_present'], det['scores_missing']], dim=0)
            labels_all = torch.cat([det['labels_present'], det['labels_missing']], dim=0)

            N = boxes_all.size(0)
            if N < 2:
                continue

            feats = self._get_roi_feats(det['image'].unsqueeze(0), boxes_all)[0]
            geom  = self._box_geom(boxes_all, det['image'].shape[-2:])
            node_feats = torch.cat([feats, geom], dim=1)


            rel = self.repn(feats, boxes_all)

            with torch.no_grad():
                iou_mat = box_iou(boxes_all, tgt['boxes'])
                best_gt = iou_mat.argmax(dim=1)
                miss_vec = tgt['is_missing'][best_gt].float()
            yij = miss_vec.unsqueeze(1) * miss_vec.unsqueeze(0)

            rep_sum += nn.functional.binary_cross_entropy_with_logits(rel, yij, reduction='mean')

            edge_index = self._make_edge_index(rel)
            logits     = self.agcn(node_feats, edge_index)
            gcn_preds.append(logits)
            gcn_labels.append(miss_vec.long())

        if not gcn_preds:
            return torch.tensor(0., device=boxes_all.device), torch.tensor(0., device=boxes_all.device)

        all_logits = torch.cat(gcn_preds, dim=0)
        all_labels = torch.cat(gcn_labels, dim=0)
        gcn_loss   = nn.functional.cross_entropy(all_logits, all_labels)

        return gcn_loss, rep_sum

    def _get_roi_feats(self, img, boxes):
        fmap = self.detector.model.backbone(img)
        roi  = self.detector.model.roi_heads.box_roi_pool(
            fmap, [boxes], [img.shape[-2:]]
        )
        return self.detector.model.roi_heads.box_head(roi)

    def _make_edge_index(self, scores):
        idx = torch.topk(scores, self.k + 1, dim=1).indices[:, 1:]
        src = idx.flatten()
        dst = torch.arange(scores.size(0), device=scores.device)
        dst = dst.unsqueeze(1).expand(-1, self.k).flatten()
        return torch.stack([dst, src], dim=0)

    def _box_geom(self, boxes, shape):
        h, w = shape
        nb = boxes.clone()
        nb[:, [0,2]] /= w
        nb[:, [1,3]] /= h
        return torch.stack([nb[:,0], nb[:,1], nb[:,2]-nb[:,0], nb[:,3]-nb[:,1]], dim=1)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphHallucinationRCNN(train_dataset.all_parts, k=3, hidden_c=256).to(device)
detector_params = list(model.detector.backbone.parameters()) \
                + list(model.detector.rpn.parameters()) \
                + list(model.detector.roi_heads.parameters())
graph_params    = list(model.repn.parameters()) + list(model.agcn.parameters())

opt_det   = torch.optim.AdamW(detector_params, lr=1e-4, weight_decay=1e-4)
opt_graph = torch.optim.AdamW(graph_params,   lr=1e-4, weight_decay=1e-4)

epochs = 50
freeze_epoch = 20
patience = 5
detector_best_macro_f1 = 0
detector_no_improve = 0
joint_best_macro_f1 = 0
joint_no_improve = 0

if torch.cuda.is_available():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

for p in graph_params:
    p.requires_grad = False

for epoch in range(1, epochs+1):
    if epoch < freeze_epoch:
        with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:
            model.detector.train()
            batch_times, gpu_memories, cpu_memories = [], [], []
            with tqdm(train_loader, unit="batch", desc=f"Detector Epoch {epoch}/{freeze_epoch}") as tepoch:
                for images, targets in tepoch:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    start_time = time.time()

                    opt_det.zero_grad()
                    loss_dict = model.detector(images, targets)
                    total_loss = sum(loss_dict.values())
                    total_loss.backward()
                    opt_det.step()

                    end_time = time.time()
                    inference_time = end_time - start_time

                    batch_times.append(inference_time)
                    if torch.cuda.is_available():
                        gpu_mem_used = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
                        gpu_memories.append(gpu_mem_used)
                    else:
                        gpu_mem_used = 0
                    
                    cpu_mem_used = psutil.virtual_memory().used / 1024**2
                    cpu_memories.append(cpu_mem_used)


                    tepoch.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "time (s)": f"{inference_time:.3f}",
                    "GPU Mem (MB)": f"{gpu_mem_used:.0f}",
                    "CPU Mem (MB)": f"{cpu_mem_used:.0f}"
                    })

                    del loss_dict, images, targets
                    gc.collect()
                    if torch.cuda.is_available(): 
                        torch.cuda.empty_cache()

        avg_time = np.mean(batch_times)
        max_gpu = max(gpu_memories) if gpu_memories else 0
        max_cpu = max(cpu_memories)
        energy = tracker.final_emissions_data.energy_consumed
        co2 = tracker.final_emissions
        print(tabulate([
            ['Detector Epoch', epoch],
            ['Detector Loss', f"{total_loss.item():.4f}"],
            ['Avg Batch Time (s)', f"{avg_time:.3f}"],
            ['Max GPU Mem (MB)', f"{max_gpu:.0f}"],
            ['Max CPU Mem (MB)', f"{max_cpu:.0f}"],
            ['Energy (kWh)', f"{energy:.4f}"],
            ['CO2 (kg)', f"{co2:.4f}"]
        ], headers=['Metric','Value'], tablefmt='pretty'))

        model.eval()
        results = evaluate_model(model.detector, valid_loader, train_dataset.all_parts, device)
        Y_true = np.array([r['true_missing_parts']      for r in results])
        Y_pred = np.array([r['predicted_missing_parts'] for r in results])
        macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)

        if macro_f1 > detector_best_macro_f1:
            detector_best_macro_f1 = macro_f1
            detector_no_improve = 0
            torch.save(model.detector.state_dict(), "/var/scratch/sismail/models/graph_rcnn/graphrcnn_detector_missing_baseline_model.pth")
        else:
            detector_no_improve += 1
            if detector_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                freeze_epoch = epoch
                
        continue

    model.detector.load_state_dict(torch.load("/var/scratch/sismail/models/graph_rcnn/graphrcnn_detector_missing_baseline_model.pth"))

    for p in detector_params:
        p.requires_grad = False
    for p in graph_params:
        p.requires_grad = True
        
    with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:
        model.train()
        batch_times, gpu_memories, cpu_memories = [], [], []

        with tqdm(train_loader, unit="batch", desc=f"Joint Epoch {epoch - freeze_epoch + 1}/{epochs - freeze_epoch + 1}") as tepoch:
            for images, targets in tepoch:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                start_time = time.time()
                opt_graph.zero_grad()
                total_loss, loss_dict = model(images, targets)
                total_loss.backward()
                opt_graph.step()
                end_time = time.time()

                inference_time = end_time - start_time
                batch_times.append(inference_time)
                if torch.cuda.is_available():
                    gpu_mem_used = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
                    gpu_memories.append(gpu_mem_used)
                else:
                    gpu_mem_used = 0
                
                cpu_mem_used = psutil.virtual_memory().used / 1024**2
                cpu_memories.append(cpu_mem_used)


                tepoch.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "time (s)": f"{inference_time:.3f}",
                "GPU Mem (MB)": f"{gpu_mem_used:.0f}",
                "CPU Mem (MB)": f"{cpu_mem_used:.0f}"
                })


    avg_time = np.mean(batch_times)
    max_gpu = max(gpu_memories) if gpu_memories else 0
    max_cpu = max(cpu_memories)
    energy = tracker.final_emissions_data.energy_consumed
    co2 = tracker.final_emissions
    print(tabulate([
        ['Joint Epoch', epoch - freeze_epoch + 1],
        ['GNN Loss', f"{total_loss.item():.4f}"],
        ['Avg Batch Time (s)', f"{avg_time:.3f}"],
        ['Max GPU Mem (MB)', f"{max_gpu:.0f}"],
        ['Max CPU Mem (MB)', f"{max_cpu:.0f}"],
        ['Energy (kWh)', f"{energy:.4f}"],
        ['CO2 (kg)', f"{co2:.4f}"]
    ], headers=['Metric','Value'], tablefmt='pretty'))
    
    model.eval()
    print(f"\nEvaluating on validation set after Epoch {epoch}...")
    results = evaluate_model(model, valid_loader, train_dataset.all_parts, device)

    Y_true = np.array([r['true_missing_parts'] for r in results])
    Y_pred = np.array([r['predicted_missing_parts'] for r in results])
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)

    if macro_f1 > joint_best_macro_f1:
        joint_best_macro_f1 = macro_f1
        joint_no_improve = 0
        torch.save(model.state_dict(), "/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_missing_baseline_model.pth")
    else:
        joint_no_improve += 1
        if joint_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    

if torch.cuda.is_available():
    nvmlShutdown()


model.load_state_dict(torch.load("/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_missing_baseline_model.pth", map_location=device))
model.to(device)

model.eval()

results_per_image = evaluate_model(model, valid_loader, valid_dataset.all_parts, device)
# print(results_per_image)

part_level_evaluation(results_per_image, train_dataset.all_parts)

visualize_and_save_predictions(model, valid_dataset, device, out_dir="/home/sismail/Thesis/visualisations/", n_images=5)


results_per_image = evaluate_model(model, test_loader, test_dataset.all_parts, device)

part_level_evaluation(results_per_image, train_dataset.all_parts)
