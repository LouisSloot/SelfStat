from utils import *
from collections import deque
import random

class IdentityManager:
    def __init__(self, num_ids, sim_thresh = 0.5):
        self.num_ids = num_ids
        self.sim_thresh = sim_thresh
        self.embeddings = dict() # {pID: embedding}
        self.last_pos = self.build_last_pos() # {pID: DEQUE( recent bbox's )}

    def build_last_pos(self):
        last_pos = dict()
        for pID in range(self.num_ids):
            last_pos[pID] = deque()
        return last_pos

    def get_embedding(self, crop):
        # will improve with some reID model; naive for now
        emb = crop.flatten() / 255.0
        return emb
    
    def build_embeddings(self, crops):
        if len(crops) != self.num_ids:
            print(f"Number of boxes ({len(crops)}) and players ({self.num_ids}) are not equal in build_embeddings.")

        curr_pID = 0
        for crop in crops:
            emb = self.get_embedding(crop)
            self.embeddings[curr_pID] = emb
            curr_pID += 1

    def identify(self, boxes, frame):
        """ Returns a dict mapping box_num: ID. """
        assigned_ids = dict() # {box: ID}
        id_to_fit = dict() # {ID: (box, fit_scores)}
        scores = [self.calc_fit_scores(box, frame) for box in boxes]

        Q = deque(zip(boxes, scores))
        
        while Q: # still a box left to identify
            box, fit_scores = Q.popleft()
            best_score = max(fit_scores)
            best_pID = fit_scores.index(best_score)
            
            if best_pID not in id_to_fit.keys():
                id_to_fit[best_pID] = (box, fit_scores)
            
            else:
                other_box, other_fit_scores = id_to_fit[best_pID]
                other_score = other_fit_scores[best_pID]

                if best_score > other_score:
                    id_to_fit[best_pID] = (box, fit_scores)
                    other_fit_scores[best_pID] = -1 # kill the score for this ID
                    Q.append((other_box, other_fit_scores))
                
                else:
                    fit_scores[best_pID] = -1 # same as above
                    Q.append((box, fit_scores))

        for pID in id_to_fit.keys():
            assigned_box = id_to_fit[pID][0]
            assigned_ids[assigned_box] = pID
        
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

            last_pos = self.last_pos[pID]
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

    def add_pos(self, xyxy, id):
        self.last_pos[id].append(xyxy)
        # "5" frames is arbitrary... maybe only need most recent pos?
        if len(self.last_pos[id]) > 5: 
            self.last_pos[id].popleft()