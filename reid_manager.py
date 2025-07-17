
class IdentityManager:
    def __init__(self, num_ids, sim_thresh = 0.5):
        self.num_ids = num_ids
        self.sim_thresh = sim_thresh
        self.embeddings = dict() # {pID: embedding}
        self.last_pos = dict() # {pID: [recent bbox positions]}
        self.next_pID = 0

    def get_embedding(self, crop):
        # will update with some reID model
        emb = crop.flatten() / 255.0
        return emb
    
    def identify(self, crop):
        emb = self.get_embedding(crop)
        best_sim = -1
