from utils import *
from collections import deque

class IdentityManager:
    def __init__(self, num_ids, sim_thresh = 0.5):
        self.num_ids = num_ids
        self.sim_thresh = sim_thresh
        self.embeddings = dict() # {pID: embedding}
        self.last_pos = self.build_last_pos() # {pID: DEQEU( recent bbox's )}

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
        ids_taken = set()
        id_to_best = dict() # {id: (box, emb) of best fit thus far}

        for box in boxes:
            crop = crop_frame(box, frame)
            emb = self.get_embedding(crop)
            best_sim = -1
            best_match = None

            for pID, emb_ref in self.embeddings.items():
                curr_sim = cos_sim(emb, emb_ref)

                if curr_sim > best_sim:

                    if pID not in ids_taken:
                        best_match = pID
                        best_sim = curr_sim
                        id_to_best[pID] = (box, emb)

                    else: # pID already taken -- need to tiebreak
                        contested_box, contested_emb = id_to_best[pID]
                        result = self.tiebreak_id(emb, box, 
                                                  contested_emb, contested_box,
                                                  pID)
                        if result: # curr was a better fit than contested
                            best_match = pID
                            best_sim = curr_sim
                            id_to_best[pID] = (box, emb)
                            # TODO: give the box that just got beaten a new ID

            if best_match is not None:
                ids_taken.add(best_match)
                id_to_best[best_match] = (box, emb)

    def tiebreak(self, emb1, box1, emb2, box2, pID):
        """ Returns True if emb1, box1 is a better match for the pID. """
        ### Future additions: Past Velocity/Acceleration + color matching(?)
        emb_ref = self.embeddings[pID]
        last_pos = self.last_pos[pID]

        xyxy1, xyxy2 = get_corners(box1), get_corners(box2)
        sim1, sim2 = cos_sim(emb1, emb_ref), cos_sim(emb2, emb_ref)

        if last_pos is None:
            return (sim1 > sim2)
        
        else:
            sim_min, sim_max = -1, 1
            sim1_norm = normalize(sim1, sim_min, sim_max)
            sim2_norm = normalize(sim2, sim_min, sim_max)

            xyxy_ref = get_corners(last_pos)
            iou1, iou2 = findIOU(xyxy1, xyxy_ref), findIOU(xyxy2, xyxy_ref)
            # IOU already in [0,1]
            weight_sim, weight_iou = 0.65, 1

            score_1 = (sim1_norm * weight_sim) + (iou1 * weight_iou)
            score_2 = (sim2_norm * weight_sim) + (iou2 * weight_iou)
            return score_1 > score_2

    def add_pos(self, xyxy, id):
        self.last_pos[id].append(xyxy)
        # "5" frames is arbitrary... maybe only need most recent pos?
        if len(self.last_pos[id]) > 5: 
            self.last_pos[id].popleft()
