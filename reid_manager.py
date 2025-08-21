from utils import *
from collections import deque
import torchvision.transforms as T
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torchreid import models
import cv2

### MAJOR TODO: - reID still insufficient ... more to be done with embeddings?
###             maybe finetune OSNet, color matching, other ideas...

class IdentityManager:
    """
    Manages player identity tracking across video frames using appearance and position features.
    Combines embedding similarity with motion prediction for robust re-identification.
    """
    def __init__(self, num_ids, sv_ids, conf_thresh = 0.3, update_rate = 0.1):
        """
        Initialize identity manager with player count and supervised IDs.
        
        Args:
            num_ids (int): Number of players to track
            sv_ids (list): User-provided labels for each player
            conf_thresh (float): Minimum confidence threshold for ID assignment
            update_rate (float): Learning rate for embedding updates (0.0-1.0)
        """
        self.num_ids = num_ids
        self.sv_id_lookup = self.build_sv_id_lookup(sv_ids)
        self.conf_thresh = conf_thresh
        self.update_rate = update_rate
        self.embeddings = dict() # {pID: current embedding reference}
        self.embedding_history = dict() # {pID: deque of recent embeddings}
        self.color_histograms = dict() # {pID: current color histogram reference}
        self.color_history = dict() # {pID: deque of recent color histograms}
        self.last_pos = dict() # {pID: DEQUE( recent bbox's )}
        
        if torch.backends.mps.is_available(): # i am developing on a mac
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.reid_model = models.build_model(
            name='osnet_x0_25',
            num_classes=1000,  # ignored since we use features
            pretrained=True
        )
        self.reid_model.eval()
        self.reid_model.to(self.device)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # OSNet's preferred input size
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
        Extract appearance embedding from player crop using OSNet ReID model.
        
        Args:
            crop (np.ndarray): Cropped image region containing player
            
        Returns:
            torch.Tensor: L2-normalized feature vector for similarity comparison
        """
        if crop is None or crop.size == 0:
            return torch.zeros(512)
        
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(rgb_crop).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            features = self.reid_model(img_tensor)
            features = F.normalize(features, p=2, dim=1)
        
        return features.squeeze(0).cpu()
    
    def get_color_histogram(self, crop, bins=32):
        """
        Extract color histogram features from player crop for color-based matching.
        Uses HSV color space for better illumination invariance.
        
        Args:
            crop (np.ndarray): Cropped image region containing player
            bins (int): Number of histogram bins per channel
            
        Returns:
            np.ndarray: Normalized color histogram features
        """
        if crop is None or crop.size == 0:
            return np.zeros(bins * 3)  # H, S, V channels
        
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        h_hist = cv2.calcHist([hsv_crop], [0], None, [bins], [0, 180])  # Hue: 0-180
        s_hist = cv2.calcHist([hsv_crop], [1], None, [bins], [0, 256])  # Saturation: 0-255
        v_hist = cv2.calcHist([hsv_crop], [2], None, [bins], [0, 256])  # Value: 0-255
        
        # normalize histograms
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)
        
        color_features = np.concatenate([h_hist, s_hist, v_hist])
        
        return color_features
    
    def calc_color_sim(self, hist1, hist2):
        """
        Calculate similarity between two color histograms using correlation.
        
        Args:
            hist1 (np.ndarray): First color histogram
            hist2 (np.ndarray): Second color histogram
            
        Returns:
            float: Color similarity score [0-1]
        """
        if hist1 is None or hist2 is None:
            return 0.0
            
        corr= cv2.compareHist(hist1.astype(np.float32), 
                                    hist2.astype(np.float32), 
                                    cv2.HISTCMP_CORREL)
        corr_min, corr_max = -1, 1
        
        corr = normalize(corr, corr_min, corr_max)
        return corr
    
    def build_embedding_refs(self, crops):
        """
        Build initial reference embeddings for each player from labeled crops.
        Also initializes embedding history for iterative updates.
        
        Args:
            crops (list): List of cropped player images from reference frame
        """
        if len(crops) != self.num_ids:
            print(f"Number of boxes ({len(crops)}) and players ({self.num_ids}) are not equal in build_embeddings.")

        curr_pID = 0
        for crop in crops:
            emb = self.get_embedding(crop)
            color_hist = self.get_color_histogram(crop)
            
            self.embeddings[curr_pID] = emb
            self.color_histograms[curr_pID] = color_hist
            
            # init history with the ref features
            self.embedding_history[curr_pID] = deque([emb.clone()], maxlen=5)
            self.color_history[curr_pID] = deque([color_hist.copy()], maxlen=5)
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
        self.update_embeddings(assigned_ids, frame)

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
            return get_corners(positions[0]) if positions else None
        
        # pixels / frame
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
        Uses embedding history for robust multi-reference matching and combines with position.
        
        Args:
            box: Bounding box detection to evaluate
            frame (np.ndarray): Current video frame
            
        Returns:
            list: Similarity scores for each player ID [0-1]
        """
        crop = crop_frame(box, frame)
        emb_candidate = self.get_embedding(crop)
        fit_scores = [0] * self.num_ids

        for pID in range(self.num_ids):
            
            color_hist_candidate = self.get_color_histogram(crop)
            
            score_appearance = self.calc_appearance_score(emb_candidate, pID)
            score_color = self.calc_color_score(color_hist_candidate, pID)
            score_pos = self.calc_position_score(box, pID)
            
            scores = [score_appearance, score_color, score_pos]
            weights = [0.8, 0.05, 0.1]
            score_ovr = sum([score * weight for (score, weight) in zip(scores, weights)] )
            
            fit_scores[pID] = score_ovr

        return fit_scores
    
    def calc_appearance_score(self, emb_candidate, pID):
        """
        Calculate appearance similarity using embedding history for robust matching.
        
        Args:
            candidate_emb (torch.Tensor): Embedding of candidate detection
            pID (int): Player ID to compare against
            
        Returns:
            float: Normalized appearance similarity score [0-1]
        """
        if pID not in self.embedding_history:
            return 0.0
        
        history = list(self.embedding_history[pID])
        if not history:
            return 0.0
        
        sims = []
        
        for emb_hist in history:
            sim = cos_sim(emb_candidate, emb_hist)
            sims.append(sim)
        
        sim_avg = sum(sims) / len(sims)
        sim_min, sim_max = -1, 1
        score_appearance = normalize(sim_avg, sim_min, sim_max)

        return score_appearance
    
    def calc_color_score(self, color_hist_candidate, pID):
        """
        Calculate color similarity using color histogram history for robust matching.
        
        Args:
            color_hist_candidate (np.ndarray): Color histogram of candidate detection
            pID (int): Player ID to compare against
            
        Returns:
            float: Normalized color similarity score [0-1]
        """
        if pID not in self.color_history:
            return 0.0
        
        history = list(self.color_history[pID])
        if not history:
            return 0.0
        
        sims = []
        
        for color_hist in history:
            sim = self.calc_color_sim(color_hist_candidate, color_hist)
            sims.append(sim)
        
        # Average similarity across history
        sim_avg = sum(sims) / len(sims)
        
        return sim_avg

    def calc_position_score(self, box, pID):
        pred_pos = self.pred_next_pos(pID)
                
        if pred_pos is None:
            return None
        else:
            xyxy = get_corners(box)
            iou = findIOU(xyxy, pred_pos)
        return iou

    def update_embeddings(self, assigned_ids, frame):
        """
        Update player embeddings using exponential moving average from new detections.
        
        Args:
            assigned_ids (dict): Mapping from bounding box to assigned player ID
            frame (np.ndarray): Current video frame for cropping
        """
        for box, pID in assigned_ids.items():
            crop = crop_frame(box, frame)
            new_embedding = self.get_embedding(crop)
            new_color_hist = self.get_color_histogram(crop)
            
            # Update embedding with exponential moving average
            current_emb = self.embeddings[pID]
            updated_emb = (1 - self.update_rate) * current_emb + self.update_rate * new_embedding
            self.embeddings[pID] = F.normalize(updated_emb, p=2, dim=0)
            
            # Update color histogram with exponential moving average
            current_color = self.color_histograms[pID]
            updated_color = (1 - self.update_rate) * current_color + self.update_rate * new_color_hist
            self.color_histograms[pID] = updated_color
            
            # Update histories
            self.embedding_history[pID].append(updated_emb.clone())
            self.color_history[pID].append(updated_color.copy())
    
    def update_last_pos(self, assigned_ids):
        """
        Update position history with newly assigned bounding boxes.
        
        Args:
            assigned_ids (dict): Mapping from bounding box to assigned player ID
        """
        for box, pID in assigned_ids.items():
            self.last_pos[pID].append(box)