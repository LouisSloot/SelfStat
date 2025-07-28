from utils import *
from collections import deque
import torchvision.transforms as T
import numpy as np
from scipy.optimize import linear_sum_assignment

### MAJOR TODO: -improve reID logic, including get_embedding 
###             -consider constantly updating emb_ref for each player

class IdentityManager:
    """
    Manages player identity tracking across video frames using appearance and position features.
    Combines embedding similarity with motion prediction for robust re-identification.
    """
    def __init__(self, num_ids, sv_ids, conf_thresh = 0.3):
        """
        Initialize identity manager with player count and supervised IDs.
        
        Args:
            num_ids (int): Number of players to track
            sv_ids (list): User-provided labels for each player
            conf_thresh (float): Minimum confidence threshold for ID assignment
        """
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
        """
        Create mapping from numerical player IDs to user-provided supervised labels.
        
        Args:
            sv_ids (list): User-provided string labels for each player
            
        Returns:
            dict: Mapping from pID (int) to supervised label (str)
        """
        # TODO: Error handle if len(sv_ids) != self.num_ids
        sv_id_lookup = dict()
        for pID in range(self.num_ids):
            sv_id_lookup[pID] = sv_ids[pID]
        return sv_id_lookup

    def build_last_pos(self, selected_boxes):
        """
        Initialize position history deques with initial labeled bounding boxes.
        
        Args:
            selected_boxes (list): Initial bounding boxes from user labeling
        """
        for pID, box in zip(range(self.num_ids), selected_boxes):
            self.last_pos[pID] = deque([box], maxlen = 5)
    
    def get_sv_id(self, pID):
        """
        Retrieve user-provided supervised label for a given player ID.
        
        Args:
            pID (int): Numerical player ID
            
        Returns:
            str: User-provided label for this player
        """
        # TODO: Error handle an invalid pID
        return self.sv_id_lookup[pID]

    def get_embedding(self, crop):
        """
        Extract appearance embedding from player crop using CNN features.
        
        Args:
            crop (np.ndarray): Cropped image region containing player
            
        Returns:
            torch.Tensor: Flattened feature vector for similarity comparison
        """
        # will improve with some reID model; naive for now
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img_tensor = self.transform(rgb_crop)  # Tensor: [C, H, W]
        return img_tensor.flatten()  # Flattened 1D tensor for cosine similarity
    
    def build_embedding_refs(self, crops):
        """
        Build reference embeddings for each player from initial labeled crops.
        
        Args:
            crops (list): List of cropped player images from reference frame
        """
        if len(crops) != self.num_ids:
            print(f"Number of boxes ({len(crops)}) and players ({self.num_ids}) are not equal in build_embeddings.")

        curr_pID = 0
        for crop in crops:
            emb = self.get_embedding(crop)
            self.embeddings[curr_pID] = emb
            curr_pID += 1

    def identify(self, boxes, frame):
        """
        Assign player IDs to detected bounding boxes using Hungarian algorithm for optimal matching.
        Relies on calc_fit_scores method to find globally optimal assignments.
        
        Args:
            boxes (list): List of detected bounding boxes from current frame
            frame (np.ndarray): Current video frame for appearance analysis
            
        Returns:
            dict: Mapping from bounding box to assigned player ID (pID)
                 Only includes assignments above confidence threshold
        """
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
    
    def pred_next_pos(self, pID):
        """
        Predict next bounding box position using velocity calculated from position history.
        Uses average velocity across recent positions to extrapolate future location.
        
        Args:
            pID (int): Player ID for which to predict position
            
        Returns:
            tuple or None: Predicted bounding box as (x1, y1, x2, y2),
                          or None if insufficient position history
        """
        positions = list(self.last_pos[pID])

        if len(positions) < 2:
            return positions[-1] if positions else None
        
        # pixles / frame
        x_velos = []
        y_velos = []
        
        for i in range(len(positions) - 1):
            curr_box = positions[i]
            next_box = positions[i + 1]
            
            curr_cx, curr_cy = get_center(curr_box)
            next_cx, next_cy = get_center(next_box)
            
            x_velos.append(next_cx - curr_cx)
            y_velos.append(next_cy - curr_cy)
        
        avg_x_velo = sum(x_velos) / len(x_velos)
        avg_y_velo = sum(y_velos) / len(y_velos)
        
        last_box = positions[-1]
        last_cx, last_cy = get_center(last_box)
        last_w, last_h = get_wh(last_box)
        
        pred_cx = last_cx + avg_x_velo
        pred_cy = last_cy + avg_y_velo
        
        pred_x1 = int(pred_cx - (last_w / 2))
        pred_y1 = int(pred_cy - (last_h / 2))
        pred_x2 = int(pred_cx + (last_w / 2))
        pred_y2 = int(pred_cy + (last_h / 2))
        
        return (pred_x1, pred_y1, pred_x2, pred_y2)


    def calc_fit_scores(self, box, frame):
        """
        Calculate similarity scores between a detection and all known players.
        Combines appearance similarity with predicted position overlap.
        
        Args:
            box: Bounding box detection to evaluate
            frame (np.ndarray): Current video frame
            
        Returns:
            list: Similarity scores for each player ID [0-1]
        """
        crop = crop_frame(box, frame)
        emb = self.get_embedding(crop)
        fit_scores = [0] * self.num_ids

        for pID in range(self.num_ids):
            curr_score = 0
            emb_ref = self.embeddings[pID]
            sim = cos_sim(emb, emb_ref)
            sim_min, sim_max = -1, 1
            sim_norm = normalize(sim, sim_min, sim_max)

            pred_pos = self.pred_next_pos(pID) # already xyxy format
            
            if pred_pos is None:
                curr_score = sim_norm
            
            else:
                xyxy = get_corners(box)
                iou = findIOU(xyxy, pred_pos)
                # IOU already in [0,1]
                weight_sim, weight_iou = 0.7, 0.3
                # weights sum to 1, so score is normalized already
                curr_score = (sim_norm * weight_sim) + (iou * weight_iou)
            
            fit_scores[pID] = curr_score

        return fit_scores
    
    def update_last_pos(self, assigned_ids):
        """
        Update position history with newly assigned bounding boxes.
        
        Args:
            assigned_ids (dict): Mapping from bounding box to assigned player ID
        """
        for box, pID in assigned_ids.items():
            self.last_pos[pID].append(box)