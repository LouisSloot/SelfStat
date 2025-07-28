from utils import *
from collections import deque
import torchvision.transforms as T
import numpy as np
from scipy.optimize import linear_sum_assignment

### MAJOR TODO: -improve reID logic, including get_embedding 
###             -consider constantly updating emb_ref for each player

class IdentityManager:
    def __init__(self, num_ids, sv_ids, conf_thresh = 0.3):
        self.num_ids = num_ids
        self.sv_id_lookup = self.build_sv_id_lookup(sv_ids)
        self.conf_thresh = conf_thresh
        self.embeddings = dict() # {pID: embedding reference}
        self.last_pos = dict() # {pID: DEQUE( recent bbox's )}
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def build_sv_id_lookup(self, sv_ids):
        # TODO: Error handle if len(sv_ids) != self.num_ids
        sv_id_lookup = dict()
        for pID in range(self.num_ids):
            sv_id_lookup[pID] = sv_ids[pID]
        return sv_id_lookup

    def build_last_pos(self, selected_boxes):
        for pID, box in zip(range(self.num_ids), selected_boxes):
            self.last_pos[pID] = deque([box], maxlen = 5)
    
    def get_sv_id(self, pID):
        # TODO: Error handle an invalid pID
        return self.sv_id_lookup[pID]

    def get_embedding(self, crop):
        # will improve with some reID model; naive for now
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img_tensor = self.transform(rgb_crop)  # Tensor: [C, H, W]
        return img_tensor.flatten()  # Flattened 1D tensor for cosine similarity
    
    def build_embedding_refs(self, crops):
        if len(crops) != self.num_ids:
            print(f"Number of boxes ({len(crops)}) and players ({self.num_ids}) are not equal in build_embeddings.")

        curr_pID = 0
        for crop in crops:
            emb = self.get_embedding(crop)
            self.embeddings[curr_pID] = emb
            curr_pID += 1

    def identify(self, boxes, frame):
        """ Returns a dict mapping box: pID using Hungarian algorithm for optimal assignment. """
        if not boxes:
            return dict()
            
        # limit boxes to avoid crashes - TODO: handle this more gracefully
        boxes = boxes[:self.num_ids]
        
        # 2D: rows = boxes, cols = pIDs
        cost_matrix = []
        for box in boxes:
            fit_scores = self.calc_fit_scores(box, frame)
            costs = [1.0 - score for score in fit_scores]
            cost_matrix.append(costs)
        
        cost_matrix = np.array(cost_matrix)
        
        if len(boxes) < self.num_ids:
            # pad with high cost dummy rows
            dummy_rows = np.full((self.num_ids - len(boxes), self.num_ids), 1.0)
            cost_matrix = np.vstack([cost_matrix, dummy_rows])
        
        # Hungarian matching
        box_indices, player_indices = linear_sum_assignment(cost_matrix)
        
        assigned_ids = dict()
        
        for box_idx, player_idx in zip(box_indices, player_indices):
            if box_idx < len(boxes):  # skip dummy boxes
                box = boxes[box_idx]
                cost = cost_matrix[box_idx, player_idx]
                fit_score = 1.0 - cost
                
                if fit_score > self.conf_thresh:
                    assigned_ids[box] = player_idx
        
        self.update_last_pos(assigned_ids)
        return assigned_ids

    def calc_fit_scores(self, box, frame):
        crop = crop_frame(box, frame)
        emb = self.get_embedding(crop)
        fit_scores = [0] * self.num_ids

        for pID in range(self.num_ids):
            curr_score = 0
            emb_ref = self.embeddings[pID]
            sim = cos_sim(emb, emb_ref)
            sim_min, sim_max = -1, 1
            sim_norm = normalize(sim, sim_min, sim_max)

            last_pos = self.last_pos[pID][-1]
            if last_pos is None:
                curr_score = sim_norm
            
            else:
                xyxy = get_corners(box)
                xyxy_ref = get_corners(last_pos)
                iou = findIOU(xyxy, xyxy_ref)
                # IOU already in [0,1]
                weight_sim, weight_iou = 0.7, 0.3
                # weights sum to 1, so score is normalized already
                curr_score = (sim_norm * weight_sim) + (iou * weight_iou)
            
            fit_scores[pID] = curr_score

        return fit_scores
    
    def update_last_pos(self, assigned_ids):
        for box, pID in assigned_ids.items():
            self.last_pos[pID].append(box)